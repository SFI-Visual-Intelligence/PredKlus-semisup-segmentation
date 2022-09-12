import numpy as np

def is_sandeel(data, labels, echogram, other_fish_as_separate_class=True, other_fish_to_ignore=False, ignore_val=-100):

    if other_fish_as_separate_class and other_fish_to_ignore:
        print('is_sandeel function: kwargs other_fish_as_separate_class and other_fish_to_ignore are both given as True.')
        print('Both cannot be True at the same time.')
        return None

    if other_fish_to_ignore:
        # Fish types that are set to 'ignore value': All other than "Sandeel" (27)
        ignores = (labels != 0) & (labels != 27)
    else:
        # Fish types that are set to 'ignore value': "0-group sandeel" (5027) and "Possible sandeel" (6009)
        ignores = (labels == 5027) | (labels == 6009) | (labels == ignore_val)

    new_labels = np.zeros(labels.shape)
    new_labels[labels == 27] = 1
    if other_fish_as_separate_class:
        new_labels[(labels != 0) & (labels != 27) & (labels != ignore_val)] = 2
    new_labels[ignores] = ignore_val

    return data, new_labels, echogram