from mlx_vlm import load, generate
import mlx.core as mx
import json
from pathlib import Path
from cs336_alignment.zero_shot_baseline import parse_gsm8k_response

# load GSM8k examples
def load_gsm8k(path):
    path = Path(path)
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    return examples

# format prompt
def format_gsm8k_prompt(example):
    prompt = f"""{example["question"]}
  Answer:"""
    return prompt

# generate output 
def generate_output(model, prompt):
    response = generate(model,processor,prompt,max_tokens=128,temperature=0.0,verbose=False)
    return response.text 

# parse and evaluate
def evaluate(model_output, example):
    answer = parse_gsm8k_response(model_output)
    gold_answer = example["answer"].split("####")[-1].strip().replace(",", "")
    return int(answer == gold_answer)

# evaluate 20 examples
MODEL_ID = "mlx-community/gemma-4-e4b-4bit"
#MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
model, processor = load(MODEL_ID)
N = 20
path = "/Users/liukunwu/Library/CloudStorage/Dropbox/GitHub/cs336_assignments/assignment5-alignment/data/gsm8k/train.jsonl"
examples = load_gsm8k(path)

score = 0
for i in range(N):
    prompt = format_gsm8k_prompt(examples[i])
    output = generate_output(model, prompt)
    print(output)
    score += evaluate(output, examples[i])

print(f"Got {score} questions correct out of {N} questions.")
