import numpy as np
def flip_x_axis(data, labels, echogram=None):
    if np.random.randint(2):
        data = np.flip(data, 2).copy()
        labels = np.flip(labels, 1).copy()
    return data, labels, echogram