import torch
import numpy as np

def setup_seed_21():
    seed = 21
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
