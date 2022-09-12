import numpy as np
from scipy.ndimage.morphology import binary_closing

def relabel_with_threshold_morph_close(
        data, labels, echogram=None,
        frequencies=[18, 38, 120, 200],
        threshold_freq=200,
        threshold_val=[1e-7, 1e-4],
        ignore_val=None,
        ignore_zero_inside_bbox=True):
    '''
    Refine existing labels based on thresholding with respect to pixel values in image.
    :param data: (numpy.array) Image (C, H, W)
    :param labels: (numpy.array) Labels corresponding to image (H, W)
    :param echogram: (Echogram object) Echogram
    :param threshold_freq: (int) Image frequency channel that is used for thresholding
    :param threshold_val: (float) Threshold value that is applied to image for assigning new labels
    :param ignore_val: (int) Ignore value (specific label value) instructs loss function not to compute gradients for these pixels
    :param ignore_zero_inside_bbox: (bool) labels==1 that is relabeled to 0 are set to ignore_value if True, 0 if False
    :return: data, new_labels, echogram
    '''

    closing = np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0]
    ])

    if ignore_val == None:
        ignore_val = 0

    # Set new label for all pixels inside bounding box that are below threshold value
    if ignore_zero_inside_bbox:
        label_below_threshold = ignore_val
    else:
        label_below_threshold = 0

    # Get refined label masks
    freq_idx = frequencies.index(threshold_freq)

    # Relabel
    new_labels = labels.copy()

    mask_threshold = (labels != 0) & (labels != ignore_val) & (data[freq_idx, :, :] > threshold_val[0]) & (data[freq_idx, :, :] < threshold_val[1])
    mask_threshold_closed = binary_closing(mask_threshold, structure=closing)
    mask = (labels != 0) & (labels != ignore_val) & (mask_threshold_closed == 0)

    new_labels[mask] = label_below_threshold
    new_labels[labels == ignore_val] = ignore_val

    return data, new_labels, echogram


def relabel_with_threshold_morph_close_no_upper(
        data, labels, echogram,
        frequencies=[18, 38, 120, 200],
        threshold_freq=200,
        threshold_val=1e-7,
        ignore_val=-100,
        ignore_zero_inside_bbox=True):
    '''
    Refine existing labels based on thresholding with respect to pixel values in image.
    :param data: (numpy.array) Image (C, H, W)
    :param labels: (numpy.array) Labels corresponding to image (H, W)
    :param echogram: (Echogram object) Echogram
    :param threshold_freq: (int) Image frequency channel that is used for thresholding
    :param threshold_val: (float) Threshold value that is applied to image for assigning new labels
    :param ignore_val: (int) Ignore value (specific label value) instructs loss function not to compute gradients for these pixels
    :param ignore_zero_inside_bbox: (bool) labels==1 that is relabeled to 0 are set to ignore_value if True, 0 if False
    :return: data, new_labels, echogram
    '''

    closing = np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0]
    ])

    if ignore_val == None:
        ignore_val = 0

    # Set new label for all pixels inside bounding box that are below threshold value
    if ignore_zero_inside_bbox:
        label_below_threshold = ignore_val
    else:
        label_below_threshold = 0

    # Get refined label masks
    freq_idx = frequencies.index(threshold_freq)

    # Relabel
    new_labels = labels.copy()

    mask_threshold = (labels != 0) & (labels != ignore_val) & (data[freq_idx, :, :] > threshold_val)
    mask_threshold_closed = binary_closing(mask_threshold, structure=closing)
    mask = (labels != 0) & (labels != ignore_val) & (mask_threshold_closed == 0)

    new_labels[mask] = label_below_threshold
    new_labels[labels == ignore_val] = ignore_val

    return data, new_labels, echogram