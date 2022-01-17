import numpy as np

def count(arr):
    unique, counts = np.unique(arr, return_counts=True)
    count_dict = dict(zip(unique, counts))
    return count_dict
