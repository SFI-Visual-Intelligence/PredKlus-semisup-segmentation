import numpy as np
from scipy import ndimage

def sobel(data, labels, echogram=None, frequencies=None):
    '''
    Re-assign labels to: Background==0, Sandeel==1, Other==2 - all remaining are set to ignore_value.
    :param data:
    :param labels:
    :param echogram:
    :param ignore_val:
    :return:
    '''

    # new_labels = np.zeros(labels.shape)
    # new_labels[np.where(labels == 27)] = 1
    # new_labels[np.where(labels == 1)] = 2
    # missed = len(np.where(new_labels == -100)[0])
    # if missed > 0:
    #     print('Unconverted classs: %d pixels' % missed)
    #     print(np.where(new_labels == -100))

    sobel_data_l = []
    for i in range(len(data)):
        sobel_data_temp = []
        for j in range(len(data[i])):
            sobel_data_temp.append(ndimage.sobel(data[i][j]))
        sobel_data_l.append(np.asarray(sobel_data_temp))
    sobel_data = np.asarray(sobel_data_l)
    sobel_data = sobel_data/np.max(sobel_data)
    return sobel_data, labels, echogram, frequencies