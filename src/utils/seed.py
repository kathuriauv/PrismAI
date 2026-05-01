import torch
import numpy as np
import random
import os
def set_global_seed(seed: int = 42):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f" Global seed is locked to: {seed}")

if __name__ == "__main__":
    set_global_seed(42)