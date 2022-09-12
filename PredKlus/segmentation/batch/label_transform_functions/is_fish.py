
def is_fish(data, labels, echogram):
    labels = (labels != 0).astype('int')
    return data, labels, echogram
