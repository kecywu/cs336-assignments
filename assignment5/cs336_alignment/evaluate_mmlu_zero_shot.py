from mlx_vlm import load, generate
import mlx.core as mx
import csv
from pathlib import Path
from cs336_alignment.zero_shot_baseline import parse_mmlu_response

# load MMLU examples
def load_mmlu(path):
    path = Path(path)
    subject = path.stem.removesuffix("_test").replace("_", " ")

    examples = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            question, option_a, option_b, option_c, option_d, answer = row

            examples.append({
                  "subject": subject,
                  "question": question,
                  "options": [option_a, option_b, option_c, option_d],
                  "answer": answer,
              })
    return examples

# format prompt
def format_mmlu_prompt(example):
    prompt = f"""Answer the following multiple choice question about {example["subject"]}. Respond with a single
      sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the
      correct answer (i.e., A, B, C or D).

  Question: {example["question"]}
  A. {example["options"][0]}
  B. {example["options"][1]}
  C. {example["options"][2]}
  D. {example["options"][3]}
  Answer:"""
    return prompt

# generate output 
def generate_output(model, prompt):
    response = generate(model,processor,prompt,max_tokens=16,temperature=0.0,verbose=False)
    return response.text 

# parse and evaluate
def evaluate(model_output, example):
    answer = parse_mmlu_response(model_output)
    return int(answer == example["answer"])

# evaluate 20 examples
MODEL_ID = "mlx-community/gemma-4-e4b-4bit"
model, processor = load(MODEL_ID)
N = 20
path = "/Users/liukunwu/Library/CloudStorage/Dropbox/GitHub/cs336_assignments/assignment5-alignment/data/mmlu/test/econometrics_test.csv"
examples = load_mmlu(path)

score = 0
for i in range(N):
    prompt = format_mmlu_prompt(examples[i])
    output = generate_output(model, prompt)
    print(output)
    score += evaluate(output, examples[i])

print(f"Got {score} questions correct out of {N} questions.")
