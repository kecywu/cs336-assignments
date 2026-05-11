from nltk.tokenize import word_tokenize

def gopher_quality_filter(text):
    
    tokenized_text = word_tokenize(text, preserve_line=True)
    if len(tokenized_text) < 50 or len(tokenized_text) > 100000:
        return False
    
    mean_word_length = sum(len(word) for word in tokenized_text) / len(tokenized_text)
    if mean_word_length < 3 or mean_word_length > 10:
        return False
    
    lines = text.splitlines()
    if lines:
        ellipsis_count = sum(1 for line in lines if line.strip().endswith("..."))
        if (ellipsis_count / len(lines)) > 0.30:
            return False
    
    alpha_word_count = sum(1 for word in tokenized_text if any(c.isalpha() for c in word))
    if (alpha_word_count / len(tokenized_text)) < 0.80:
        return False

    return True