import numpy as np

from segmentation.utils.np import getGrid, linear_interpolation, nearest_interpolation

class Dataset():

    def __init__(self, samplers, window_size, frequencies,
                 n_samples = 1000,
                 sampler_probs=None,
                 augmentation_function=None,
                 label_transform_function=None,
                 data_transform_function=None):
        """
        A dataset is used to draw random samples
        :param samplers: The samplers used to draw samples
        :param window_size: expected window size
        :param n_samples:
        :param frequencies:
        :param sampler_probs:
        :param augmentation_function:
        :param label_transform_function:
        :param data_transform_function:
        """

        self.samplers = samplers
        self.window_size = window_size
        self.n_samples = n_samples
        self.frequencies = frequencies
        self.sampler_probs = sampler_probs
        self.augmentation_function = augmentation_function
        self.label_transform_function = label_transform_function
        self.data_transform_function = data_transform_function

        # Normalize sampling probabillities
        if self.sampler_probs is None:
            self.sampler_probs = np.ones(len(samplers))
        self.sampler_probs = np.array(self.sampler_probs)
        self.sampler_probs = np.cumsum(self.sampler_probs).astype(float)
        self.sampler_probs /= np.max(self.sampler_probs)

    def __getitem__(self, index):
        #Select which sampler to use
        i = np.random.rand()
        sampler = self.samplers[np.where(i < self.sampler_probs)[0][0]]

        #Draw coordinate and echogram with sampler
        center_location, echogram = sampler.get_sample()

        # Adjust coordinate by random shift in y and x direction
        center_location[0] += np.random.randint(-self.window_size[0]//2, self.window_size[0]//2 + 1)
        center_location[1] += np.random.randint(-self.window_size[1]//2, self.window_size[1]//2 + 1)

        #Get data/labels-patches
        data, labels = get_crop(echogram, center_location, self.window_size, self.frequencies)


        # Apply augmentation
        if self.augmentation_function is not None:
            data, labels, echogram = self.augmentation_function(data, labels, echogram)

        # Apply label-transform-function
        if self.label_transform_function is not None:
            data, labels, echogram = self.label_transform_function(data, labels, echogram)

        # Apply data-transform-function
        if self.data_transform_function is not None:
            data, labels, echogram, frequencies = self.data_transform_function(data, labels, echogram, self.frequencies)

        labels = labels.astype('int16')
        return data, labels

    def __len__(self):
        return self.n_samples

class Dataset_single():

    def __init__(self, data_in, labels_in,
                 augmentation_function=None,
                 label_transform_function=None,
                 data_transform_function=None):
        """
        A dataset is used to draw random samples
        :param samplers: The samplers used to draw samples
        :param window_size: expected window size
        :param n_samples:
        :param frequencies:
        :param sampler_probs:
        :param augmentation_function:
        :param label_transform_function:
        :param data_transform_function:
        """
        self.data_in = data_in
        self.labels_in = labels_in
        self.n_samples = len(labels_in)
        self.augmentation_function = augmentation_function
        self.label_transform_function = label_transform_function
        self.data_transform_function = data_transform_function

    def __getitem__(self, index):

        data = self.data_in[index]
        labels = self.labels_in[index]
        # Apply augmentation
        if self.augmentation_function is not None:
            data, labels, _ = self.augmentation_function(data, labels)

        # Apply label-transform-function
        if self.label_transform_function is not None:
            data, labels, _ = self.label_transform_function(data, labels)

        # Apply data-transform-function
        if self.data_transform_function is not None:
            data, labels, _, _ = self.data_transform_function(data, labels)

        labels = labels.astype('int16')
        return data, labels

    def __len__(self):
        return self.n_samples

class Dataset_single_tensor():

    def __init__(self, data_in, labels_in,
                 augmentation_function=None,
                 label_transform_function=None,
                 data_transform_function=None):
        """
        A dataset is used to draw random samples
        :param samplers: The samplers used to draw samples
        :param window_size: expected window size
        :param n_samples:
        :param frequencies:
        :param sampler_probs:
        :param augmentation_function:
        :param label_transform_function:
        :param data_transform_function:
        """
        self.data_in = data_in
        self.labels_in = labels_in
        self.n_samples = len(labels_in)
        self.augmentation_function = augmentation_function
        self.label_transform_function = label_transform_function
        self.data_transform_function = data_transform_function

    def __getitem__(self, index):

        data = self.data_in[index]
        labels = self.labels_in[index]
        # Apply augmentation
        if self.augmentation_function is not None:
            data, labels, _ = self.augmentation_function(data, labels)

        # Apply label-transform-function
        if self.label_transform_function is not None:
            data, labels, _ = self.label_transform_function(data, labels)

        # Apply data-transform-function
        if self.data_transform_function is not None:
            data, labels, _, _ = self.data_transform_function(data, labels)

        return data, labels

    def __len__(self):
        return self.n_samples


def get_crop(echogram, center_location, window_size, freqs):
    """
    Returns a crop of data around the pixels specified in the center_location.

    """
    # Get grid sampled around center_location
    grid = getGrid(window_size) + np.expand_dims(np.expand_dims(center_location, 1), 1)

    channels = []
    for f in freqs:

        # Interpolate data onto grid
        memmap = echogram.data_memmaps(f)[0]
        data = linear_interpolation(memmap, grid, boundary_val=0, out_shape=window_size)
        del memmap

        # Set non-finite values (nan, positive inf, negative inf) to zero
        if np.any(np.invert(np.isfinite(data))):
            data[np.invert(np.isfinite(data))] = 0

        channels.append(np.expand_dims(data, 0))
    channels = np.concatenate(channels, 0)

    labels = nearest_interpolation(echogram.label_memmap(), grid, boundary_val=-100, out_shape=window_size)

    return channels, labels
