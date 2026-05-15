import unicodedata
import os
import mmh3
from collections import defaultdict
from itertools import combinations

def normalize_text(text):

    if not text:
        return "" 
    
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    text = text.lower()
    text = " ".join(text.split())

    return text 

def load_and_clean_files(input_files):

    clean_files = []

    for file in input_files:
        if not os.path.exists(file):
            continue 
        with open(file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        cleaned_file = normalize_text(raw_text)
        clean_files.append(cleaned_file)
    
    return clean_files

def generate_ngrams(text, ngrams):

    words = text.split()

    if len(words) < ngrams:
        return {tuple(words)}
    else:
        return {tuple(words[i:i+ngrams]) for i in range(len(words)-ngrams+1)}
    
def compute_minhash_signature(text, num_hashes, ngrams):

    signature = [float('inf')] * num_hashes 
    ngrams_set = generate_ngrams(text, ngrams)

    for ngram in ngrams_set:
        ngram_str = " ".join(ngram)
        ngram_bytes = ngram_str.encode("utf-8")

        for seed in range(num_hashes):
            hash_val = mmh3.hash(ngram_bytes, seed=seed, signed=False)
            if hash_val < signature[seed]:
                signature[seed] = hash_val

    return signature

def get_lsh_buckets(signatures, num_bands):

    buckets = defaultdict(list)
    r = len(signatures[0]) // num_bands
    
    for doc_id, signature in enumerate(signatures):
        for i in range(num_bands):
            band = tuple(signature[i*r:(i+1)*r])
            bucket_key = (i, band)
            buckets[bucket_key].append(doc_id)

    return buckets

def identify_candidate_pairs(buckets):
    candidates = set()

    for doc_ids in buckets.values():
        if len(doc_ids) < 2:
            continue 

        for a, b in combinations(doc_ids, 2):
            candidates.add((min(a, b), max(a, b)))
    
    return candidates

def get_duplicates(candidates, jaccard_threshold, files, ngrams):

    duplicates = set()

    for candidate in candidates:
        a,b = candidate[0], candidate[1]
        ngrams_a = set(generate_ngrams(files[a], ngrams))
        ngrams_b = set(generate_ngrams(files[b], ngrams))

        intersection_size = len(ngrams_a.intersection(ngrams_b))
        union_size = len(ngrams_a.union(ngrams_b))
        jaccard_similarity = intersection_size / union_size

        if jaccard_similarity > jaccard_threshold:
            duplicates.add(candidate)
    
    return duplicates

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x 
    
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a


def get_docs_to_keep(duplicates, n):

    uf = UnionFind(n)
    for duplicate in duplicates:
        uf.union(duplicate[0], duplicate[1])
    
    seen_clusters = set()
    keep = set()

    for i in range(n):
        root = uf.find(i)

        if root not in seen_clusters:
            seen_clusters.add(root)
            keep.add(i)

    return keep

def minhash_dedup(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_dir):

    # load file and normalize text
    clean_files = load_and_clean_files(input_files)
    
    # compute minhash signature for each cleaned document
    signatures = []
    for file in clean_files:
        signatures.append(compute_minhash_signature(file, num_hashes, ngrams))

    # use LSH to identify candidate duplicates
    buckets = get_lsh_buckets(signatures, num_bands)
    candidates = identify_candidate_pairs(buckets)

    # update duplicates based on Jaccard similarity
    duplicates = get_duplicates(candidates, jaccard_threshold, clean_files, ngrams)

    # cluster duplicated documents
    docs_to_keep = get_docs_to_keep(duplicates, len(input_files))
    
    # remove and write to output directory
    for doc in docs_to_keep:
        file_name = os.path.basename(input_files[doc])
        out_path = os.path.join(output_dir, file_name)

        with open(input_files[doc], "r", encoding="utf-8") as infile, open(out_path, "w", encoding="utf-8") as outfile:
            outfile.write(infile.read())

