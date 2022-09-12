import numpy as np

def binary_classification(data, labels, echogram):
    if np.any(labels == 1):
        new_label = 1
    else:
        new_label = 0
    return data, new_label, echogram


def binary_classification_with_ignore_values(data, labels, echogram, ignore_value=-100):

    # Todo: Not sure how well this function performs - not tested

    if np.any(labels == 1):
        new_label = 1
    else:
        if np.any(labels == ignore_value):
            new_label = ignore_value
        else:
            new_label = 0
    return data, new_label, echogram