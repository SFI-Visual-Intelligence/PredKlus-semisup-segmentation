import numpy as np

def relabel_with_threshold(
        data, labels, echogram,
        frequencies=[18, 38, 120, 200],
        threshold_freq=200,
        threshold_val=[1e-7, 1e-4],
        ignore_val=-100,
        ignore_zero_inside_bbox=True):
    '''
    Refine existing labels based on thresholding with respect to pixel values in image.
    :param data: (numpy.array) Image (H, W, C)
    :param labels: (numpy.array) Labels corresponding to image (H, W)
    :param echogram: (Echogram object) Echogram
    :param threshold_freq: (int) Image frequency channel that is used for thresholding
    :param threshold_val: (float) Threshold value that is applied to image for assigning new labels
    :param ignore_val: (int) Ignore value (specific label value) instructs loss function not to compute gradients for these pixels
    :param ignore_zero_inside_bbox: (bool) labels==1 that is relabeled to 0 are set to ignore_value if True, 0 if False
    :return: data, new_labels, echogram
    '''

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
    new_labels[
        (labels != 0) &
        (
            (data[freq_idx, :, :] < threshold_val[0]) |
            (data[freq_idx, :, :] > threshold_val[1])
        )
    ] = label_below_threshold
    new_labels[labels == ignore_val] = ignore_val

    return data, new_labels, echogram