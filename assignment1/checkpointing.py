import torch 

def save_checkpoint(model, optimizer, iteration, out):

    obj = {}
    obj["iteration"] = iteration 
    obj["model_state_dict"] = model.state_dict()
    obj["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(obj, out)


def load_checkpoint(src, model, optimizer):

    obj = torch.load(src)
    model.load_state_dict(obj["model_state_dict"])
    optimizer.load_state_dict(obj["optimizer_state_dict"])

    return obj["iteration"]