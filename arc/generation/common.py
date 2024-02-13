import numpy as np


def get_oscillation_patterns(lag):
    kernel = [1] + [0] * (lag - 1) + [1] + [0] * (lag - 1)
    return [list(np.roll(kernel, i)) for i in range(lag)]
