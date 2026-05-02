import torch 
from cs336_basics.softmax import softmax_temp_scaling


"""
A decoder with the following features:
1. Generate completions for a user-provided prompt until hitting an <|endoftext|> token.
2. Allow the user to control the maximum number of generated tokens.
3. Given a desired temperature value, apply softmax temperature scaling to the predicted next-
token distributions before sampling.
4. Top-p sampling given a user-specified threshold value.
"""

def decode(tokens, max_gen_tokens, tau, p, tokenizer, transformer):

    seq = tokens  # initialize

    transformer.eval()

    with torch.no_grad():
        for _ in range(max_gen_tokens):
            input_seq = seq[-transformer.context_length:]
            logits = transformer(input_seq)[-1, :]
            probs = softmax_temp_scaling(logits, -1, tau)

            # top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cum_probs > p
            remove_mask[1:] = remove_mask[:-1].clone()
            remove_mask[0] = False
            sorted_indices_remove = sorted_indices[remove_mask]
            probs[sorted_indices_remove] = 0
            probs = probs / torch.sum(probs, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1, replacement=True)
            next_token_id = next_token.item()

            eos_id = tokenizer.encode("<|endoftext|>")[0]
            if next_token_id == eos_id:
                break
            
            seq = torch.cat([seq, next_token])

    return seq

