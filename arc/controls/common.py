import numpy as np
import random


def get_oscillation_patterns(lag):
    kernel = [1] + [0] * (lag - 1) + [1] + [0] * (lag - 1)
    return [list(np.roll(kernel, i)) for i in range(lag)]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
