import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import argparse

import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import matplotlib.pyplot as plt

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, '..', '..'))

import segmentation.models.unet_bn_sequential_db as models
from segmentation import clustering
from segmentation.patch_sampling import sampling_echograms_same_length
from segmentation.train_modules import supervised_and_self_supervised_train, generate_pseudolabels, prediction, prediction_earlystop
from segmentation.confusion_matrix import plot_conf, plot_conf_best, plot_macro, plot_macro_best
from segmentation.pytorchtools import EarlyStopping


def parse_args():
    current_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
    parser.add_argument('--problem_setting', type=list, default=['test', 'semi', 40], help='full/test, sup/semi, 10/25')
    # paths
    parser.add_argument('--exp', type=str, default=current_dir, help='path to exp folder')
    parser.add_argument('--resume',
                        default=os.path.join(current_dir, 'checkpoint.pth.tar'), type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--exp_check', type=str, default=os.path.join(current_dir, 'checkpoints'), help='path to exp folder')
    parser.add_argument('--pred_test', type=str, default=os.path.join(current_dir, 'pred_test'), help='path to exp folder')
    # Earlystop
    parser.add_argument('--earlystoppath', type=str, default=os.path.join(current_dir, 'checkpoints', 'early_stop_check.pt'), help='path to exp folder')
    parser.add_argument('--patience', type=int, default=100, help='earlystop_patience')
    # patch
    parser.add_argument('--window_dim', type=int, default=256, help='patch_window_size')
    parser.add_argument('--window_size', type=list, default=[256, 256], help='patch_window_size')
    parser.add_argument('--in_channels', type=int, default=4, help='frequency channels')
    parser.add_argument('--label_types', type=list, default=[1, 27], help='fishtypes')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--n_clusters', type=int, default=512, help='nmb_custers for k-means per patch. The actual K is a multiplication of this and batch_size')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--num_patches_clustering', default=30, type=int, help='num_patches for gpu clustering')
    # model
    parser.add_argument('--n_classes', type=int, default=3, help='nmb_categories')
    parser.add_argument('--fd', type=int, default=64, help='feature length before the conv_final')
    parser.add_argument('--pca', default=32, type=int, help='pca dimension (default: 128)')
    parser.add_argument('--use_given_label', default=False, type=bool, help='subsampling pixels for deepclustering. USe pred if False')
    parser.add_argument('--class_weight', default=[1, 1, 1], type=list, help='bg/se/ot weight for cross entropy loss')
    # optimizer
    parser.add_argument('--lr_Adam', default=3e-5, type=float, help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--lr_reduction', type=float, default=0.5, help='learning_rate_reduction')
    parser.add_argument('--momentum', type=float, default=0.95, help='momentum')
    # training
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers pytorch')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--save_epoch', default=50, type=int,
                        help='save features every epoch number (default: 20)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of total epochs to run (default: 200)')
    # count
    parser.add_argument('--display_count', type=int, default=5,
                        help='display iterations for every <display_count> numbers')
    parser.add_argument('--verbose', type=bool, default=False, help='chatty')
    parser.add_argument('--f1_avg', type=str, default='weighted', help='the way computing f1-score')
    return parser.parse_args(args=[])

def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device)
    criterion_test = nn.CrossEntropyLoss(ignore_index=-1)

    '''
    ##########################################
    ##########################################
    # Model definition
    ##########################################
    ##########################################'''
    model = models.UNet(n_classes=args.n_classes, in_channels=args.in_channels)
    model.conv_final = None
    model = model.double()
    model.to(device)
    cudnn.benchmark = True
    optimizer_body =  optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr_Adam, betas=(0.9, 0.999), weight_decay=10 ** args.wd)

    '''conv_final: trainable one'''
    model.conv_final = models.conv1x1(args.fd, args.n_classes)
    model.conv_final = model.conv_final.double()
    model.conv_final.to(device)
    optimizer_final = optim.Adam(filter(lambda x: x.requires_grad, model.conv_final.parameters()), lr=args.lr_Adam, betas=(0.9, 0.999), weight_decay=10 ** args.wd)
    model.conv_final = None
    '''
    ########################################
    ########################################
    Create echogram sampling index
    ########################################
    ########################################'''

    print('Sample echograms.')
    dataloader_train_labeled, dataloader_train_unlabeled, dataloader_test = sampling_echograms_same_length(args)

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.n_clusters, args.pca) # default args.pca = None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top located layer parameters from checkpoint
            copy_checkpoint_state_dict = checkpoint['state_dict'].copy()
            checkpoint['state_dict'] = copy_checkpoint_state_dict
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_body.load_state_dict(checkpoint['optimizer_body'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
    if not os.path.isdir(args.exp_check):
        os.makedirs(args.exp_check)

    # creating prediction_test
    if not os.path.isdir(args.pred_test):
        os.makedirs(args.pred_test)

    if os.path.isfile(os.path.join(args.exp, 'early_stopping.pth.tar')):
        early_stopping = torch.load(os.path.join(args.exp, 'early_stopping.pth.tar'))
    else:
        early_stopping = EarlyStopping(patience=args.patience, verbose=False, delta=0, path=args.earlystoppath)

    if os.path.isfile(os.path.join(args.exp, 'records_tr_lab_epoch.pth.tar')):
        records_tr_lab_epoch = torch.load(os.path.join(args.exp, 'records_tr_lab_epoch.pth.tar'))
    else:
        records_tr_lab_epoch = {'epoch': [],
                                'lab_sup_loss_epoch': [],
                                'BG_tr_lab_pixel_count': [],
                                'SE_tr_lab_pixel_count': [],
                                'OT_tr_lab_pixel_count': []
                                }

    if os.path.isfile(os.path.join(args.exp, 'records_te_epoch.pth.tar')):
        records_te_epoch = torch.load(os.path.join(args.exp, 'records_te_epoch.pth.tar'))
    else:
        records_te_epoch = {'epoch': [],
                            'test_loss_epoch': [],
                            'BG_accu_epoch': [],
                            'SE_accu_epoch': [],
                            'OT_accu_epoch': [],
                            'BG_te_pixel_count': [],
                            'SE_te_pixel_count': [],
                            'OT_te_pixel_count': [],
                            'te_cluster_pixel_count_epoch': [],
                            'prob_mat': [],
                            'mat': [],
                            'f1_score': [],
                            'kappa': [],
                            'fpr': [],
                            'tpr': [],
                            'roc_auc': [],
                            'roc_auc_macro': [],
                            }

    for epoch in range(args.start_epoch, args.epochs):
        '''
        #######################
        #######################
        MAIN TRAINING
        #######################
        #######################'''

        print('#####################  Start training at Epoch %d ################'% epoch)

        '''
        ##################################### 
        ##################################### 
        ########## Pseudo-labeling ##########
        ##################################### 
        #####################################'''

        inputs, pseudolabels, deepcluster =\
            generate_pseudolabels(dataloader_train_unlabeled, model, deepcluster, device, args)

        tr_cluster_pixel_count_save = np.bincount(pseudolabels.ravel(), minlength=args.n_clusters)
        selfloss_weight = np.reciprocal(tr_cluster_pixel_count_save/np.mean(tr_cluster_pixel_count_save))
        criterion_self = nn.CrossEntropyLoss(weight=torch.Tensor(selfloss_weight).double().to(device))

        '''
        ########################################################################## 
        ########################################################################## 
        ########## SELF_SUPEVISED + SUPERVISED (Mini-batch alternation) ##########
        ########################################################################## 
        ##########################################################################'''
        torch.cuda.empty_cache()
        lab_sup_loss_save, \
        lab_sup_output_save, \
        lab_label_save, \
        tr_lab_pixel_count_save, \
        self_loss_save, \
        self_output_save, \
        self_pseudolabel_save = \
            supervised_and_self_supervised_train(dataloader_train_labeled, model, inputs, pseudolabels, criterion_self, optimizer_body, optimizer_final, epoch, device, args)


        print('TR-LAB MAIN TRAIN pixel count:', tr_lab_pixel_count_save)
        records_tr_lab_epoch['epoch'].append(epoch)
        records_tr_lab_epoch['lab_sup_loss_epoch'].append(lab_sup_loss_save)
        records_tr_lab_epoch['BG_tr_lab_pixel_count'].append(tr_lab_pixel_count_save[0])
        records_tr_lab_epoch['SE_tr_lab_pixel_count'].append(tr_lab_pixel_count_save[1])
        records_tr_lab_epoch['OT_tr_lab_pixel_count'].append(tr_lab_pixel_count_save[2])
        torch.save(records_tr_lab_epoch, os.path.join(args.exp, 'records_tr_lab_epoch.pth.tar'))

        print('#####################  LABELED PART TRAINED ################')

        '''
        ################################ 
        ################################ 
        ########## TEST PHASE ##########
        ################################ 
        ################################ 
        '''
        print('#####################  TEST START ################')
        torch.cuda.empty_cache()

        test_loss, \
        prob_mat, \
        mat, \
        f1_score,\
        kappa, \
        predictions, \
        pred_clusters, \
        predictions_mat,\
        labels, \
        te_pixel_count_save, \
        te_cluster_pixel_count_save, \
        fpr, \
        tpr, \
        roc_auc, \
        roc_auc_macro = prediction(dataloader_test, model, criterion_test, epoch, device, args)

        plot_macro(fpr, tpr, roc_auc, epoch, args)
        bg_accu, se_accu, ot_accu = prob_mat.diagonal()
        plot_conf(epoch, prob_mat, mat, f1_score, kappa, args)

        print('TE pixel count:', te_pixel_count_save)
        records_te_epoch['epoch'].append(epoch)
        records_te_epoch['test_loss_epoch'].append(test_loss)
        records_te_epoch['te_cluster_pixel_count_epoch'].append(te_cluster_pixel_count_save)
        records_te_epoch['BG_te_pixel_count'].append(te_pixel_count_save[0])
        records_te_epoch['SE_te_pixel_count'].append(te_pixel_count_save[1])
        records_te_epoch['OT_te_pixel_count'].append(te_pixel_count_save[2])
        records_te_epoch['prob_mat'].append(prob_mat)
        records_te_epoch['mat'].append(mat)
        records_te_epoch['f1_score'].append(f1_score)
        records_te_epoch['kappa'].append(kappa)
        records_te_epoch['BG_accu_epoch'].append(bg_accu)
        records_te_epoch['SE_accu_epoch'].append(se_accu)
        records_te_epoch['OT_accu_epoch'].append(ot_accu)
        records_te_epoch['fpr'].append(fpr)
        records_te_epoch['tpr'].append(tpr)
        records_te_epoch['roc_auc'].append(roc_auc)
        records_te_epoch['roc_auc_macro'].append(roc_auc_macro)
        torch.save(records_te_epoch, os.path.join(args.exp, 'records_te_epoch.pth.tar'))

        torch.save(pred_clusters, os.path.join(args.pred_test, '%d_test_cluster.pth.tar' % epoch))
        torch.save(predictions, os.path.join(args.pred_test, '%d_test_pred.pth.tar' % epoch))
        torch.save(predictions_mat, os.path.join(args.pred_test, '%d_test_pred_softmax.pth.tar' % epoch))
        torch.save(labels, os.path.join(args.pred_test, '%d_test_label.pth.tar' % epoch))

        # save running checkpoint
        model.conv_final = None
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_body': optimizer_body.state_dict(),
                    },
                   os.path.join(args.exp, 'checkpoint.pth.tar'))

        if epoch % args.save_epoch == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer_body': optimizer_body.state_dict(),
                        },
                       os.path.join(args.exp_check, '%d_checkpoint.pth.tar' % epoch))


if __name__ == '__main__':
    args = parse_args()
    main(args=args)
