import numpy as np
from tqdm import tqdm

def get_tqdm_bar(iterator, desc):
    return tqdm(iterator, desc=desc, bar_format="{l_bar}{bar}| Update: {n_fmt}/{total_fmt} - {unit} - Elapsed: {elapsed}")

def one_hot_encode(labels):
    oh_labels = np.zeros((labels.shape[0], 10))
    for i in range(labels.shape[0]):
        oh_labels[i, labels[i]] = 1
    return oh_labels

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{round(self.avg, 4)}"

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
