import numpy as np

def add_noise(data, labels, echogram=None):

    # Apply random noise to crop with probability p = 0.5
    if np.random.randint(2):

        # Change pixel value in 5% of the pixels
        change_pixel_value = np.random.binomial(1, 0.05, data.shape)

        # Pixels that are changed:
        # 50% are increased (multiplied by random number in [1, 10]
        # 50% are reduced (multiplied by random number in [0, 1]
        increase_pixel_value = np.random.binomial(1, 0.5, data.shape)

        data *= (1 - change_pixel_value) + \
                change_pixel_value * \
                (
                        increase_pixel_value * np.random.uniform(1, 10, data.shape) +
                        (1 - increase_pixel_value) * np.random.uniform(0, 1, data.shape)
                )

    return data, labels, echogram