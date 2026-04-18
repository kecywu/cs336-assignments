
import regex as re
import pickle

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab 
        self.merges = merges
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        #when special tokens are substrings of each other and are ordered in the regex pattern — longer tokens should be matched first. 
        if self.special_tokens:
            sorted_tokens = sorted(special_tokens, key=len, reverse=True)
            self.pattern = "|".join(re.escape(token) for token in sorted_tokens)
        else:
            self.pattern = None

        self.ids = {v:k for k,v in self.vocab.items()}
        self.merge_priority = {merge: i for i, merge in enumerate(self.merges)}
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)

        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)
    
    def _get_pairs(self, token_list):
        pairs = []
        for pair in zip(token_list, token_list[1:]):
            pairs.append(pair)

        return pairs
        
    def _apply_merges(self, token_list):

        while len(token_list) >= 2:
            pairs = self._get_pairs(token_list)
            pair = min(pairs, key=lambda p: self.merge_priority.get(p, float("inf")))
            if pair not in self.merge_priority: # dictionary is O(1) look up
                break 
            new_token_list = []
            i = 0
            while i < len(token_list):
                if i < len(token_list) - 1 and token_list[i] == pair[0] and token_list[i+1] == pair[1]:
                    new_token_list.append(pair[0]+pair[1])
                    i += 2
                else:
                    new_token_list.append(token_list[i])
                    i += 1
            token_list = new_token_list

        return token_list 
    
    def encode(self, text):
        ids = []
        if self.pattern is not None:
            chunks = re.split(f"({self.pattern})", text) # Wrapping the pattern in a capturing group preserves them
        else:
            chunks = [text]

        for chunk in chunks:
            if not chunk:
                continue 
            if self.special_tokens and chunk in self.special_tokens:
                ids.append(self.ids[chunk.encode('utf-8')])
            else:
                pre_tokens = [m.group() for m in re.finditer(self.PAT, chunk)]
                pre_token_list = [[bytes([b]) for b in token.encode('utf-8')] for token in pre_tokens]

                for token_list in pre_token_list:
                    token_list = self._apply_merges(token_list) # merge
                    for token in token_list:
                        ids.append(self.ids[token]) # convert to id

        return ids 
    
    def encode_iterable(self, iterable):
        for chunk in iterable:
            yield from self.encode(chunk)
    
    def decode(self, ids):
        tokens = b"".join(self.vocab[id] for id in ids)
        text = tokens.decode("utf-8", errors="replace")
        
        return text