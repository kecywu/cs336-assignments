import numpy as np
import os 
from cs336_assignments.assignment1.tokenizer import Tokenizer 


def tokenization(filepath, tokenizer, filename, basepath="/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/"):

    with open(filepath, "r", encoding="utf-8") as f:
        ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)

    arr = np.array(ids, dtype=np.uint16) # unsigned 16-bit, goes from 0 to 65535, larger than vocab size, saves disk space than int32 and int64
    outpath = os.path.join(basepath, f"{filename}.npy")
    np.save(outpath, arr)

# load input
file_path_stories_valid = "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
file_path_owt_valid =  "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/owt_valid.txt"
file_path_stories_train = "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
file_path_owt_train =  "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/owt_train.txt"
special_tokens = ["<|endoftext|>"]

vocab_stories = "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/tinystories_vocab.pkl"
merges_stories = "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/tinystories_merges.pkl"

vocab_owt = "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/owt_vocab.pkl"
merges_owt = "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/owt_merges.pkl"

tokenizer_stories = Tokenizer.from_files(vocab_stories, merges_stories, special_tokens)
tokenizer_owt = Tokenizer.from_files(vocab_owt, merges_owt, special_tokens)

# tokenize
tokenization(file_path_stories_valid, tokenizer_stories, "tinystories_token_valid")
print("Done with tinystories valid!")

tokenization(file_path_stories_train, tokenizer_stories, "tinystories_token_train")
print("Done with tinystories train!")

tokenization(file_path_owt_valid, tokenizer_owt, "owt_token_valid")
print("Done with owt valid!")

tokenization(file_path_owt_train, tokenizer_owt, "owt_token_train")
print("Done with owt train!")
