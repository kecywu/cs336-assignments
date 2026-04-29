import torch 
import numpy as np
from cs336_basics import data_loading, rope, transformer_lm, adamw, cross_entropy, checkpointing, learning_rate_schedule, gradient_clipping
import os
import wandb

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
train_path = "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/tinystories_token_train.npy"
val_path = "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/data/tinystories_token_valid.npy"
train_dataset = np.load(train_path, mmap_mode="r")
val_dataset = np.load(val_path, mmap_mode="r")
out_path = "/Users/liukunwu/Documents/GitHub/cs336_assignments/assignment1-basics/results/"
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
    "max_l2_norm" : 100, 
    "max_learning_rate" : 0.01,
    "min_learning_rate" : 0.0001, 
    "warmup_iters" : 100,
    "cosine_cycle_iters" : 900,
    "num_train_steps" : 1000,
    "eval_every" : 100,
    "checkpoint_every" : 100,
    "num_val_batches" : 10
}
wandb.init(project="cs336_transformer", name="run-with-tinystories", config=config)

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
optimizer = adamw.AdamW(model.parameters())

# training loop
for step in range(1, num_train_steps + 1):
    model.train()

    input_seq, target_seq = data_loading.data_loading(
        train_dataset,
        batch_size,
        context_length,
        device,
    )

    optimizer.zero_grad()
    predictions = model(input_seq)
    loss = cross_entropy.cross_entropy(predictions, target_seq)
    loss.backward()

    gradient_clipping.gradient_clipping(model.parameters(), max_l2_norm)
    lr = learning_rate_schedule.learning_rate_schedule(step, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()

    wandb.log({"train/lr": lr, "train/loss": loss.item(), "step": step})

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

                val_predictions = model(val_input)
                loss_val = cross_entropy.cross_entropy(val_predictions, val_target)
                val_loss += loss_val.item()

        avg_val_loss = val_loss / num_val_batches

        print(f"Step [{step}/{num_train_steps}] | Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f}")

        wandb.log({
            "val/loss": avg_val_loss,
            "step": step,
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
