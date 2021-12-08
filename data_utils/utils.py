import numpy as np 
import torch

def collate_fn(samples):
    xs = []
    scores = []
    for sample in samples:
        xs.append(sample[0])
        scores.append(sample[1])

    xs = np.array(xs)
    scores = np.array(scores)

    return torch.tensor(xs).float(), torch.tensor(scores).float()