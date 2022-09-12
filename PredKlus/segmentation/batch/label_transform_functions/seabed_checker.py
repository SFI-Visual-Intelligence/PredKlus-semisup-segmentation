import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

def seabed_checker(data, labels, echogram=None, ignore_val=0):
    '''
    Re-assign labels to: Background==0, Sandeel==1, Other==2 - all remaining are set to ignore_value.
    :param data:
    :param labels:
    :param echogram:
    :param ignore_val:
    :return:
    '''
    tss = np.asarray(data)
    tss_av = np.average(tss, axis=-3)
    lss = np.asarray(labels)

    dim_tss_av = np.shape(tss_av)
    dim_lss = np.shape(tss_av)

    if len(dim_tss_av) == 2:
        tss_av = np.expand_dims(tss_av, 0)
    if len(dim_lss) == 2:
        lss = np.expand_dims(lss, 0)

    kernel = np.ones((3, 3), np.uint8)  # note this is a horizontal kernel
    new_labels = []
    for j, (img, label) in enumerate(zip(tss_av, lss)):
        d_im = cv2.dilate(np.float32(img > 0.01), kernel, iterations=1)
        e_im = cv2.erode(d_im, kernel, iterations=1)
        no_boundary = np.argwhere(np.sum(e_im, axis=0) == 0).ravel()
        if len(no_boundary) < 20:
            while not len(np.argwhere(np.sum(e_im, axis=0) == 0).ravel()) == 0:
                no_boundary = np.argwhere(np.sum(e_im, axis=0) == 0).ravel()
                for col in no_boundary:
                    e_im[:, col] = e_im[:, col - 1]
            for i in range(256):
                index = np.where(e_im[:, i] == 1)
                if len(index[0]) == 0:
                    continue
                label[max(index[0]):, i] = -1
        new_labels.append(label)
    new_labels = np.asarray(new_labels)
    new_labels = np.squeeze(new_labels)
    return data, new_labels, echogram


