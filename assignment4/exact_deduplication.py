from collections import Counter
import os
import hashlib

def hash_line(line, hash_method="md5"):

    line_bytes = line.encode("utf-8")

    if hash_method == "md5":
        return hashlib.md5(line_bytes).hexdigest()
    elif hash_method == "sha-256":
        return hashlib.sha256(line_bytes).hexdigest()
    else:
        raise ValueError(f"Unsupported hash method: '{hash_method}'. Supported methods are 'md5' and 'sha-256'.")



def exact_dedup(input_files, output_dir, hash_method="md5"):

    freq = Counter()

    # build frequency table
    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f: 
                line_key = hash_line(line, hash_method)
                freq[line_key] = freq.get(line_key, 0) + 1

    # rewriting files
    for file in input_files:
        file_name = os.path.basename(file)
        out_path = os.path.join(output_dir, file_name)

        with open(file, "r", encoding="utf-8") as infile, open(out_path, "w", encoding="utf-8") as outfile:
            for line in infile:
                line_key = hash_line(line, hash_method)
                if freq.get(line_key, 0) == 1:
                    outfile.write(line)