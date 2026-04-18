import regex as re 
from collections import Counter, defaultdict
import multiprocessing as mp
import os
from typing import BinaryIO
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# wrapper class for max heap with custom tiebreak rules
class Entry:
    __slots__ = ("count", "pair")
    def __init__(self, count, pair):
        self.count = count
        self.pair = pair
    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count   # higher count = "smaller" = popped first
        return self.pair > other.pair          # ties broken by larger pair

def pre_tokenize_and_build_counts(chunk, pattern, PAT):
    # remove special tokens, pre-tokenize and gather count
    segments = re.split(pattern, chunk)
    pre_token_counts = dict(Counter(
                        m.group() for segment in segments for m in re.finditer(PAT, segment)))
    byte_pre_token_counts = {
        tuple(bytes([b]) for b in token.encode('utf-8')): count 
        for token, count in pre_token_counts.items()
    }
    
    return byte_pre_token_counts

# worker has to be top-level function so it works with multiprocessing
def worker(args):
    path, start, end, pattern = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # pre-tokenize and build counts
    return pre_tokenize_and_build_counts(chunk, pattern, PAT)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # build the initial vocab
    vocab = {i: bytes([i]) for i in range(256)}
    for i in range(len(special_tokens)):
        vocab[len(vocab)+i] = special_tokens[i].encode('utf-8') 

    pattern = "|".join(re.escape(token) for token in special_tokens)
    
    def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    # parallelizing pre-tokenization
    with open(input_path, "rb") as f:
        num_processes = os.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    jobs = [(input_path, s, e, pattern) for s, e in zip(boundaries[:-1], boundaries[1:])]

    with mp.Pool(processes=num_processes) as pool:
        all_byte_counts = pool.map(worker, jobs)
    
    # merge into one global Counter
    word_counts = Counter()
    for count in all_byte_counts:
        word_counts.update(count)

    # initialize pair_counts, pair_to_words dictionaries, do it once
    def get_pair_count_init(word_counts):
        pair_counts = defaultdict(int)
        pair_to_words = defaultdict(set) # a pair can appear in multiple parts of one word
        for byte_tuple, count in word_counts.items():
            for pair in zip(byte_tuple, byte_tuple[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + count
                pair_to_words[pair].add(byte_tuple)
        return pair_counts, pair_to_words

    pair_counts, pair_to_words = get_pair_count_init(word_counts)

    # initialize heap
    heap = [Entry(c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    
    # merge and update all three data structures
    def merge_fast(pair, word_counts, pair_counts, pair_to_words):
        update_set = list(pair_to_words[pair])
        merged_token = pair[0]+pair[1]

        # do word level update to prevent corner cases
        for w in update_set:
            c = word_counts[w]

            # remove every old pair
            for p in zip(w, w[1:]):
                pair_counts[p] -= c
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                else: 
                    heapq.heappush(heap, Entry(pair_counts[p], p)) # push a fresh pair, old pair with incorrect counts get stale and stay as garbage
                pair_to_words[p].discard(w)
                if not pair_to_words[p]:
                    del pair_to_words[p]
            
            # form new word
            w_new = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == pair[0] and w[i+1] == pair[1]:
                    w_new.append(merged_token)
                    i += 2
                else:
                    w_new.append(w[i])
                    i += 1
            w_new_tuple = tuple(w_new)

            # add new pair
            for p in zip(w_new_tuple, w_new_tuple[1:]):
                pair_counts[p] = pair_counts.get(p, 0) + c 
                pair_to_words[p].add(w_new_tuple)
                heapq.heappush(heap, Entry(pair_counts[p], p))

            # update word count
            word_counts[w_new_tuple] = word_counts.get(w_new_tuple, 0) + c 
            del word_counts[w]

        # assert sum(pair_counts.values()) == sum((len(w)-1)*c for w, c in word_counts.items()) # use this only for testing, it's very slow

    
    # merge process
    init_vocab_size = len(vocab)
    num_merges = vocab_size - init_vocab_size
    merges = []

    for i in range(num_merges):
        #pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        while True:
            entry = heapq.heappop(heap)
            if pair_counts.get(entry.pair, 0) == entry.count: # check for staleness
                pair = entry.pair
                break
            # else stale, discard and keep popping
        idx = init_vocab_size + i
        merge_fast(pair, word_counts, pair_counts, pair_to_words)
        merges.append(pair)
        vocab[idx] = pair[0] + pair[1]

    return vocab, merges
