import torch 
import numpy as np
from cs336_basics import data_loading, rope, transformer_lm, adamw, cross_entropy, checkpointing, learning_rate_schedule, gradient_clipping
import os
import wandb
import time
import math

"""
A training script that does the following:
1. Ability to configure and control the various model and optimizer hyperparameters.
2. Memory-efficient loading of large training and validation datasets with np.memmap.
3. Serializing checkpoints to a user-provided path.
4. Periodically logging training and validation performance
"""

# Set up device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Training on device: {device}")

# Configure hyperparameters
run_name = "test_tinystories"
train_path = "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/tinystories_token_train.npy"
val_path = "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/tinystories_token_valid.npy"
train_dataset = np.load(train_path, mmap_mode="r")
val_dataset = np.load(val_path, mmap_mode="r")
out_path = f"/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/results/{run_name}"
os.makedirs(out_path, exist_ok=True)

config = {
    "batch_size" : 32,
    "context_length" : 256,
    "d_model" : 512,
    "d_ff" : 1344,
    "num_heads" : 16, 
    "num_layers" : 4,
    "vocab_size" : 10000,
    "theta" : 10000,
    "max_l2_norm" : 1.0, 
    "max_learning_rate" : 0.001,
    "min_learning_rate" : 0.0001, 
    "warmup_iters" : 5,
    "cosine_cycle_iters" : 10,
    "num_train_steps" : 10,
    "eval_every" : 2,
    "checkpoint_every" : 10,
    "num_val_batches" : 2 # 100-500, Karpathy uses 200
}

# initialize logging
wandb.init(project="cs336-assignment1", name=run_name, config=config)
wandb.define_metric("step")
wandb.define_metric("train/*", step_metric="step")
wandb.define_metric("val/*", step_metric="step")
wandb.define_metric("perf/*", step_metric="step")
wandb.define_metric("tokens_processed", step_metric="step")

# initialize hyperparameters
batch_size = config["batch_size"]
context_length = config["context_length"]
d_model = config["d_model"]
d_ff = config["d_ff"]
num_heads = config["num_heads"]
num_layers = config["num_layers"]
vocab_size = config["vocab_size"]
theta = config["theta"]
max_l2_norm = config["max_l2_norm"]
max_learning_rate = config["max_learning_rate"]
min_learning_rate = config["min_learning_rate"]
warmup_iters = config["warmup_iters"]
cosine_cycle_iters = config["cosine_cycle_iters"]
num_train_steps = config["num_train_steps"]
eval_every = config["eval_every"]
checkpoint_every = config["checkpoint_every"]
num_val_batches = config["num_val_batches"]

# initialize model, optimizer
pos_encoder = rope.RotaryPositionalEmbedding(theta, d_model // num_heads, context_length, device)
model = transformer_lm.Transformer(
    d_model,
    num_heads,
    d_ff,
    vocab_size,
    context_length,
    num_layers,
    pos_encoder,
    device=device,
)
# speed up training on mps
# if device.type == "mps":
#     model = torch.compile(model, backend="aot_eager")
optimizer = adamw.AdamW(model.parameters())

# training loop
start = time.time()

for step in range(1, num_train_steps + 1):
    model.train()

    input_seq, target_seq = data_loading.data_loading(
        train_dataset,
        batch_size,
        context_length,
        device,
    )

    optimizer.zero_grad()
    """
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
      predictions = model(input_seq)
      loss = cross_entropy.cross_entropy(predictions, target_seq)
    """
    predictions = model(input_seq)
    loss = cross_entropy.cross_entropy(predictions, target_seq)
    loss.backward()

    grad_norm = gradient_clipping.gradient_clipping(list(model.parameters()), max_l2_norm)
    lr = learning_rate_schedule.learning_rate_schedule(step, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()

    train_ppl = torch.exp(loss).item()
    tokens_processed = step * batch_size * context_length

    wandb.log({"train/lr": lr, 
               "train/loss": loss.item(), 
               "train/grad_norm": grad_norm.item(),
               "step": step,
               "train/perplexity" : train_ppl,
               "perf/cum_time_sec" : time.time() - start,
               "tokens_processed" : tokens_processed})

    if step % eval_every == 0:
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for _ in range(num_val_batches):
                val_input, val_target = data_loading.data_loading(
                    val_dataset,
                    batch_size,
                    context_length,
                    device,
                )

                """
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    val_predictions = model(val_input)
                    loss_val = cross_entropy.cross_entropy(val_predictions, val_target)
                """

                val_predictions = model(val_input)
                loss_val = cross_entropy.cross_entropy(val_predictions, val_target)
                val_loss += loss_val.item()

        avg_val_loss = val_loss / num_val_batches
        val_ppl = math.exp(avg_val_loss)

        print(f"Step [{step}/{num_train_steps}] | Total Time: {time.time() - start}s | Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f}")

        wandb.log({
            "val/loss": avg_val_loss,
            "step": step,
            "val/perplexity" : val_ppl
        })

    if step % checkpoint_every == 0:
        checkpointing.save_checkpoint(
            model,
            optimizer,
            step,
            os.path.join(out_path, f"checkpoint_step_{step}.pt"),
        )
    
wandb.finish()
print("Training Complete!")
