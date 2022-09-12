import numpy as np
from segmentation._not_in_use.preprocess_params import preprocess_params

def sigmoid_log(data, labels, echogram):
    '''
    Transforms the input data: 'data = sigmoid(log(data))', with parametrized Sigmoid function for each frequency channel.
    This is implemented as the equivalent expression: 'data = 1 / (1 + k * data**a)', where 'k' and 'a' are parameters.
    :param data:
    :param labels:
    :param echogram:
    :return:
    '''


    if data.shape[0] != 4:
        print('sigmoid_log function requires 4 input frequency channels. ', data.shape[0], ' were given.')
        return None
    else:
        eps = 1e-25
        k = np.array(preprocess_params()['k']).reshape(4, 1, 1)
        a = np.array(preprocess_params()['a']).reshape(4, 1, 1)
        data += eps
        data = 1 / (1 + k * np.power(data, a))
        return data, labels, echogram