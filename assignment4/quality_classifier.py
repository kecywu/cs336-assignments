from pathlib import Path
import fasttext

fixtures = Path("/Users/liukunwu/Library/CloudStorage/Dropbox/GitHub/cs336_assignments/assignment4-data/tests/fixtures")

low_quality_cc = (fixtures / "low_quality_cc.txt").read_text()
high_quality_wiki = (fixtures / "high_quality_wiki_reference.txt").read_text()

def one_line(text):
    return " ".join(text.split())

train_text = "\n".join([
      f"__label__cc {one_line(low_quality_cc)}",
      f"__label__wiki {one_line(high_quality_wiki)}",
  ]) + "\n"

Path("/Users/liukunwu/Library/CloudStorage/Dropbox/GitHub/cs336_assignments/assignment4-data/data/quality_train.txt").write_text(train_text)

model = fasttext.train_supervised(
      input="data/quality_train.txt",
      epoch=25,
      lr=0.5,
      wordNgrams=2,
  )

model.save_model("/Users/liukunwu/Library/CloudStorage/Dropbox/GitHub/cs336_assignments/assignment4-data/data/classifiers/quality_classifier.bin")


def classify_quality(text):
    text = " ".join(text.split())
    model = fasttext.load_model("/Users/liukunwu/Library/CloudStorage/Dropbox/GitHub/cs336_assignments/assignment4-data/data/classifiers/quality_classifier.bin")
    labels, probs = model.predict(text, k=1)
    label = labels[0].removeprefix("__label__")
    
    return label, probs[0]