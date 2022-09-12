import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from segmentation import paths
from segmentation.batch.dataset import Dataset_single
from segmentation.batch.data_transform_functions.remove_nan_inf import remove_nan_inf
from segmentation.batch.data_transform_functions.db_with_limits import db_with_limits
from segmentation.batch.label_transform_functions.index_0_1_27 import index_0_1_27
from segmentation.batch.label_transform_functions.relabel_with_threshold_morph_close import relabel_with_threshold_morph_close
from segmentation.batch.label_transform_functions.seabed_checker import seabed_checker
from segmentation.batch.combine_functions import CombineFunctions

def sampling_echograms_same_length(args):
    path_to_echograms = paths.path_to_echograms()
    data_multiple_tr_labeled = []
    label_multiple_tr_labeled = []
    """
    DO NOT ERASE
    # year_tr = ['2016', '2017'] : 200 patches
    # year_te = ['2019']: 60 patches
    """
    data_tr_test = torch.load(os.path.join(path_to_echograms, 'data_tr_TEST_200.pt'))
    label_tr_test = torch.load(os.path.join(path_to_echograms, 'label_tr_TEST_200.pt'))
    data_multiple_te = torch.load(os.path.join(path_to_echograms, 'data_te_TEST_60.pt'))
    label_multiple_te = torch.load(os.path.join(path_to_echograms, 'label_te_TEST_60.pt'))
    split_idx = int(args.problem_setting[2]/100 * len(data_tr_test))
    print('Labeled ratio: {0} %\nNum.LBD/num.TOT: : {1}/{2}'.format(args.problem_setting[2], split_idx, len(data_tr_test)))
    data_multiple_tr_labeled_raw, data_multiple_tr_unlabeled = data_tr_test[:split_idx], data_tr_test
    label_multiple_tr_labeled_raw, label_multiple_tr_unlabeled = label_tr_test[:split_idx], label_tr_test

    while (len(data_multiple_tr_labeled) < len(data_multiple_tr_unlabeled)):
        data_multiple_tr_labeled += data_multiple_tr_labeled_raw
        label_multiple_tr_labeled += label_multiple_tr_labeled_raw
    if not len(data_multiple_tr_labeled) == len(data_multiple_tr_unlabeled):
        data_multiple_tr_labeled = data_multiple_tr_labeled[:len(data_multiple_tr_unlabeled)]
        label_multiple_tr_labeled = label_multiple_tr_labeled[:len(label_multiple_tr_unlabeled)]

    '''##################################'''
    label_transform = CombineFunctions([index_0_1_27, relabel_with_threshold_morph_close, seabed_checker])
    label_transform_te = CombineFunctions([index_0_1_27, seabed_checker])
    data_transform = CombineFunctions([remove_nan_inf, db_with_limits])

    dataset_train_labeled = Dataset_single(
        data_multiple_tr_labeled,
        label_multiple_tr_labeled,
        augmentation_function=None,
        label_transform_function=label_transform,
        data_transform_function=data_transform)

    dataloader_train_labeled = DataLoader(dataset_train_labeled,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers,
                                          worker_init_fn=np.random.seed,
                                          drop_last=False,
                                          pin_memory=True)

    dataset_test = Dataset_single(
        data_multiple_te,
        label_multiple_te,
        augmentation_function=None,
        label_transform_function=label_transform_te,
        data_transform_function=data_transform)

    dataloader_test = DataLoader(dataset_test,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=args.num_workers,
                                          worker_init_fn=np.random.seed,
                                          drop_last=False,
                                          pin_memory=True)

    dataset_train_unlabeled = Dataset_single(
        data_multiple_tr_unlabeled,
        label_multiple_tr_unlabeled,
        augmentation_function=None,
        label_transform_function=label_transform,
        data_transform_function=data_transform)

    dataloader_train_unlabeled = DataLoader(dataset_train_unlabeled,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers,
                                          worker_init_fn=np.random.seed,
                                          drop_last=False,
                                          pin_memory=True)

    return dataloader_train_labeled, dataloader_train_unlabeled, dataloader_test

