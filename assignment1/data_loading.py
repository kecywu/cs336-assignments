import torch 
import numpy as np

def data_loading(x, batch_size, context_length, device):

    max_start_ind = len(x) - context_length 
    start_indices = np.random.randint(0, max_start_ind, size=batch_size)

    batch = np.stack([x[i : i + context_length] for i in start_indices])
    target = np.stack([x[i+1 : i+1+context_length] for i in start_indices])

    device = torch.device(device)
    batch = torch.from_numpy(batch).to(device)
    target = torch.from_numpy(target).to(device)

    return batch, target


