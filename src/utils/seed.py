import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)        # Python randomness
    np.random.seed(seed)    # NumPy randomness
    torch.manual_seed(seed) # CPU tensors
    torch.cuda.manual_seed_all(seed) # GPU tensors
