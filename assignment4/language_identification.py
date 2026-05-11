import fasttext


def language_identification(text):

    model = fasttext.load_model("/Users/liukunwu/Library/CloudStorage/Dropbox/GitHub/cs336_assignments/assignment4-data/data/classifiers/lid.176.bin")
    text = " ".join(text.split())
    labels, probs = model.predict(text, k=1)
    label = labels[0].removeprefix('__label__')
    prob = probs[0]

    return label, prob