import numpy as np
import copy

def drop_bg_pixels(data, labels, echogram=None, args=None):
    '''
    Re-assign labels to: Background==0, Sandeel==1, Other==2 - all remaining are set to ignore_value.
    :param data:
    :param labels:
    :param echogram:
    :param ignore_val:
    :return:
    '''
    new_labels = copy.copy(labels)
    subsample_bg = args.subsample_bg_pixels
    bg_idxes = np.where(new_labels==0)
    new_labels[bg_idxes] = -1
    n_bg_pixels = len(bg_idxes[0])
    new_idx = np.arange(0, n_bg_pixels, subsample_bg)
    bg_new_idxes = tuple([bg[new_idx] for bg in bg_idxes])
    new_labels[bg_new_idxes] = 0
    return data, new_labels, echogram