
from segmentation.data.normalization import db


def db_with_limits(data, labels, echogram=None, frequencies=None):
    data = db(data)
    data[data>0] = 0
    data[data<-75] = -75
    data[:, labels==-1] = -75
    data = data + 75
    return data, labels, echogram, frequencies
