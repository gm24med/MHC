import os
import random

import numpy as np
import torch

from .logging import get_logger


def set_seed(seed: int = 42, verbose: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if verbose:
        logger = get_logger("mhc.seed")
        logger.info("Random seed set to: %s", seed)
