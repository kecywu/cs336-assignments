import pickle 
import time 
import resource

from train_bpe_fast import train_bpe

INPUT_PATH = "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt" 
#INPUT_PATH = "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"  # use a small debug file first
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]
OUTPUT_DIR = "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/"

if __name__ == "__main__":
    t0 = time.time()

    vocab, merges = train_bpe(
        input_path=INPUT_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    elapsed = time.time() - t0

    # --- memory and time usage ---
    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"training took {elapsed:.1f}s")
    print(f"peak memory: {peak_bytes / 1024**3:.2f} GB")

    # --- serialize vocab + merges ---
    with open(OUTPUT_DIR + "tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(OUTPUT_DIR + "tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    # --- longest token ---
    longest = max(vocab.values(), key=len)
    print(f"longest token ({len(longest)} bytes): {longest!r}")