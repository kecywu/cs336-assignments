import fasttext

def classify_nsfw(text):

    model = fasttext.load_model("/Users/liukunwu/Library/CloudStorage/Dropbox/GitHub/cs336_assignments/assignment4-data/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
    text = " ".join(text.split())
    labels, probs = model.predict(text, k=1)
    label = labels[0].removeprefix('__label__')

    return label, probs[0]


def classify_toxic_speech(text):

    model = fasttext.load_model("/Users/liukunwu/Library/CloudStorage/Dropbox/GitHub/cs336_assignments/assignment4-data/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")
    text = " ".join(text.split())
    labels, probs = model.predict(text, k=1)
    label = labels[0].removeprefix('__label__')

    return label, probs[0]

