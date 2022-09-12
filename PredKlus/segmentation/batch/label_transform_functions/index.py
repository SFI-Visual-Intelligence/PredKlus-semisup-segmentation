import numpy as np


def index(data, labels, echogram, ignore_val=-100):
    '''
    Re-assign labels to successive integers (1, 2, ..., n) and ignore value (-100). Sandeel==1.
    :param data:
    :param labels:
    :param echogram:
    :param ignore_val:
    :return:
    '''

    label_types = [0, 1, 12, 27, 5027, 6007, 6008, 6009, 6010, 9999]

    # Fish types that are set to 'ignore value': "0-group sandeel" (5027) and "Possible sandeel" (6009)
    ignores = (labels == 5027) | (labels == 6009) | (labels == ignore_val)

    label_types_reduced = [a for a in label_types if a not in (0, 27, 5027, 6009, ignore_val)]
    label_types_reduced.sort()

    new_labels = np.zeros(labels.shape)
    new_labels[ignores] = ignore_val
    new_labels[labels == 27] = 1

    i = 2
    for k in label_types_reduced:
        new_labels[labels == k] = i
        i += 1

    return data, new_labels, echogram