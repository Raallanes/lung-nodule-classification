import numpy as np


def normalize_min_max(value):
    min_value = np.min(value)
    normalized = (value - min_value) / (np.max(value) - min_value)
    return normalized
