import torch

def batch_memory(batch):
    total_bytes = 0
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            total_bytes += val.numel() * val.element_size()
    return total_bytes / (1024 ** 2)  # in MB