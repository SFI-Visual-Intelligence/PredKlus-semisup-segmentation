import numpy as np


def index_0_1_27(data, labels, echogram=None, ignore_val=0):
    '''
    Re-assign labels to: Background==0, Sandeel==1, Other==2 - all remaining are set to ignore_value.
    :param data:
    :param labels:
    :param echogram:
    :param ignore_val:
    :return:
    '''

    new_labels = np.zeros(labels.shape)
    # new_labels.fill(ignore_val)
    # new_labels[np.where(labels == 0)] = 0
    new_labels[np.where(labels == 27)] = 1
    new_labels[np.where(labels == 1)] = 2
    missed = len(np.where(new_labels == -100)[0])
    if missed > 0:
        print('Unconverted classs: %d pixels' % missed)
        print(np.where(new_labels == -100))
    return data, new_labels, echogram