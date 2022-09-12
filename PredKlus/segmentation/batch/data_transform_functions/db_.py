
from segmentation.data.normalization import db


def db_(data, labels, echogram, frequencies):
    return db(data), labels, echogram, frequencies
