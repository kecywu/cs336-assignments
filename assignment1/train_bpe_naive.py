import regex as re 
from collections import Counter

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # open file
    with open(input_path, "r") as f:
        text = f.read()
    

    # build the initial vocab
    vocab = {i: bytes([i]) for i in range(256)}
    for i in range(len(special_tokens)):
        vocab[len(vocab)+i] = special_tokens[i].encode('utf-8') 

    # remove special tokens
    pattern = "|".join(re.escape(token) for token in special_tokens)
    chunks = re.split(pattern, text)

    # pre-tokenize and gather counts
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_token_counts = dict(Counter(
                        m.group() for chunk in chunks for m in re.finditer(PAT, chunk))) # merges also don't happen across pre-token boundaries

    byte_pre_token_counts = {
        tuple(bytes([b]) for b in token.encode('utf-8')): count 
        for token, count in pre_token_counts.items()
    }

    # pair counts weighted by pre-token frequencies
    def get_pair_count(counts):
        pair_counts = {}
        for byte_tuple, count in counts.items():
            for pair in zip(byte_tuple, byte_tuple[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        return pair_counts
    
    # merge and update byte_pre_token_counts
    def merge(pair, counts):
        byte_list = list(counts.keys())
        for byte_tuple in byte_list:
            new_byte = []
            i = 0
            while i < len(byte_tuple):
                if i < len(byte_tuple) - 1 and byte_tuple[i] == pair[0] and byte_tuple[i+1] == pair[1]:
                    new_byte.append(pair[0]+pair[1])
                    i += 2
                else:
                    new_byte.append(byte_tuple[i])
                    i += 1
            counts[tuple(new_byte)] = counts.pop(byte_tuple)
    
    # merge process
    init_vocab_size = len(vocab)
    num_merges = vocab_size - init_vocab_size
    merges = []

    for i in range(num_merges):
        pair_counts = get_pair_count(byte_pre_token_counts)
        pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        idx = init_vocab_size + i
        merge(pair, byte_pre_token_counts)
        merges.append(pair)
        vocab[idx] = pair[0] + pair[1]

    return vocab, merges
