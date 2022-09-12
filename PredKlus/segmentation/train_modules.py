import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

import gc

import numpy as np
import segmentation.models.unet_bn_sequential_db as models
import torch.optim as optim
import torch.nn.functional as F

import os
import time
import copy

from segmentation.util import AverageMeter, Logger, UnifLabelSampler
from segmentation.confusion_matrix import IB_conf_mat, IB_roc_curve_macro, plot_macro, conf_mat, roc_curve_macro
from segmentation.batch.dataset import Dataset_single
from segmentation.models.ib_explainer import idxtobool

def cps_semi_supervised_train(dataloader_train_labeled, dataloader_train_unlabeled, model, opt_l, opt_r, device, args):
    torch.cuda.empty_cache()
    running_loss = AverageMeter()
    tr_l_pixel_count_save = np.ones(args.n_classes)
    tr_r_pixel_count_save = np.ones(args.n_classes)
    model.train()
    sup_loader = iter(dataloader_train_labeled)

    with torch.autograd.set_detect_anomaly(True):
        for i in range(len(sup_loader)):
            imgs, gts = sup_loader.next()
            imgs_var = torch.autograd.Variable(imgs.to(device))
            gts_var = torch.autograd.Variable(gts.to(device, non_blocking=True))
            gts_var = gts_var.long()
            _, pred_l = model(imgs_var, step=1)
            _, pred_r = model(imgs_var, step=2)

            _, max_l = torch.max(pred_l, dim=1)
            _, max_r = torch.max(pred_r, dim=1)

            with torch.no_grad():
                max_l_cpu = copy.copy(max_l).cpu().numpy()
                max_r_cpu = copy.copy(max_r).cpu().numpy()
                tr_l_pixel_count = np.bincount(max_l_cpu.ravel(), minlength=args.n_classes)
                tr_r_pixel_count = np.bincount(max_r_cpu.ravel(), minlength=args.n_classes)
                tr_l_pixel_count_save += tr_l_pixel_count
                tr_r_pixel_count_save += tr_r_pixel_count

                predloss_l_weight = np.reciprocal(tr_l_pixel_count_save / np.mean(tr_l_pixel_count_save))
                predloss_r_weight = np.reciprocal(tr_r_pixel_count_save / np.mean(tr_r_pixel_count_save))

            crit_sup_l = nn.CrossEntropyLoss(weight=torch.Tensor(predloss_l_weight).double().to(device),
                                             ignore_index=-1)
            crit_sup_r = nn.CrossEntropyLoss(weight=torch.Tensor(predloss_r_weight).double().to(device),
                                             ignore_index=-1)

            loss_sup_l = crit_sup_l(pred_l, gts_var)
            loss_sup_r = crit_sup_r(pred_r, gts_var)
            cps_loss = crit_sup_r(pred_l, max_r) + crit_sup_l(pred_r, max_l)

            opt_l.zero_grad()
            opt_r.zero_grad()
            loss = loss_sup_l + loss_sup_r + cps_loss
            loss.backward()
            opt_l.step()
            opt_r.step()

            running_loss.update(loss.item())

        if not args.problem_setting[2] == 100:
            print('gogo')
            unsup_loader = iter(dataloader_train_unlabeled)
            for j in range(len(unsup_loader)):
                unsup_imgs, _ = unsup_loader.next()
                unsup_imgs_var = torch.autograd.Variable(unsup_imgs.to(device))
                _, pred_unsup_l = model(unsup_imgs_var, step=1)
                _, pred_unsup_r = model(unsup_imgs_var, step=2)


                _, max_unsup_l = torch.max(pred_unsup_l, dim=1)
                _, max_unsup_r = torch.max(pred_unsup_r, dim=1)

                with torch.no_grad():
                    max_unsup_l_cpu = copy.copy(max_unsup_l).cpu().numpy()
                    max_unsup_r_cpu = copy.copy(max_unsup_r).cpu().numpy()
                    tr_unsup_l_pixel_count = np.bincount(max_unsup_l_cpu.ravel(), minlength=args.n_classes)
                    tr_unsup_r_pixel_count = np.bincount(max_unsup_r_cpu.ravel(), minlength=args.n_classes)
                    tr_l_pixel_count_save += tr_unsup_l_pixel_count
                    tr_r_pixel_count_save += tr_unsup_r_pixel_count

                    predloss_l_weight = np.reciprocal(tr_l_pixel_count_save / np.mean(tr_l_pixel_count_save))
                    predloss_r_weight = np.reciprocal(tr_r_pixel_count_save / np.mean(tr_r_pixel_count_save))

                crit_unsup_l = nn.CrossEntropyLoss(weight=torch.Tensor(predloss_l_weight).double().to(device),
                                                 ignore_index=-1)
                crit_unsup_r = nn.CrossEntropyLoss(weight=torch.Tensor(predloss_r_weight).double().to(device),
                                                 ignore_index=-1)

                cps_loss = crit_unsup_r(pred_unsup_l, max_unsup_r) + crit_unsup_l(pred_unsup_r, max_unsup_l)

                opt_l.zero_grad()
                opt_r.zero_grad()
                cps_loss.backward()
                opt_l.step()
                opt_r.step()

                running_loss.update(cps_loss.item())
    return running_loss.avg

def cps_prediction(dataloader_test, model, crit, epoch, device, args):
    running_loss_test = AverageMeter()
    labels = []
    predictions = []
    predictions_mat = []

    torch.cuda.empty_cache()
    te_pred_pixel_count_save = torch.zeros(args.n_classes).double().to(device)
    model.eval()
    with torch.no_grad():
        for i, (input_tensor, label) in enumerate(dataloader_test):
            # Load test data and transfer from numpy to pytorch
            inputs_test = input_tensor.double().to(device)
            labels_test = label.long().to(device)
            pred_mat = model(inputs_test) # including softmax

            # Evaluate test data
            loss_test = crit(pred_mat, labels_test)

            # Update loss count for test set
            running_loss_test.update(loss_test.item())
            pred_class = torch.argmax(pred_mat, dim=1)
            te_pred_pixel_count = torch.bincount(pred_class.flatten(), minlength=args.n_classes)
            te_pred_pixel_count_save += te_pred_pixel_count
            predictions_mat.extend(pred_mat.cpu().numpy())
            predictions.extend(pred_class.cpu().numpy())
            labels.extend(labels_test.cpu().numpy())

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    predictions_mat = np.asarray(predictions_mat)

    keep_test_idx = np.where(labels > -1)
    labels_vec = labels[keep_test_idx]
    predictions_vec = predictions[keep_test_idx]
    predictions_mat_sampled = predictions_mat[keep_test_idx[0], :, keep_test_idx[1], keep_test_idx[2]]
    fpr, tpr, roc_auc, roc_auc_macro = roc_curve_macro(labels_vec, predictions_mat_sampled)
    prob_mat, mat, f1_score, kappa = conf_mat(ylabel=labels_vec, ypred=predictions_vec, args=args)
    acc_bg, acc_se, acc_ot = prob_mat.diagonal()

    print('\n\n#############################################')
    print('#############################################')
    print('###############   TEST   ###################')
    print(
        'Epoch {0:3d} \t  Accuracy bg[0]: {1:.3f} \t Accuracy se[1]: {2:.3f} \t Accuracy ot[2]: {3:.3f}, Loss {4:.3f}'.format(
            epoch, acc_bg, acc_se, acc_ot, running_loss_test.avg))
    print('#############################################')
    print('#############################################\n\n')
    return running_loss_test.avg, prob_mat, mat, f1_score, kappa, predictions, predictions_mat, labels, te_pred_pixel_count_save.cpu().numpy(), fpr, tpr, roc_auc, roc_auc_macro


def cps_prediction_2019(dataloader_test, model, device):
    predictions = []

    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for i, (input_tensor, _) in enumerate(dataloader_test):
            # Load test data and transfer from numpy to pytorch
            inputs_test = input_tensor.double().to(device)
            # labels_test = label.long().to(device)
            model.conv_final = None
            pred_mat = model(inputs_test)

            # Update loss count for test set
            pred_class = torch.argmax(pred_mat, dim=1)
            predictions.extend(pred_class.cpu().numpy())
    predictions = np.asarray(predictions)
    return predictions



def cps_prediction_earlystop(dataloader_test, model, crit, epoch, device, args):
    running_loss_test = AverageMeter()
    labels = []
    predictions = []
    predictions_mat = []

    torch.cuda.empty_cache()
    te_pred_pixel_count_save = torch.zeros(args.n_classes).double().to(device)
    model.eval()
    with torch.no_grad():
        for i, (input_tensor, label) in enumerate(dataloader_test):
            # Load test data and transfer from numpy to pytorch
            inputs_test = input_tensor.double().to(device)
            labels_test = label.long().to(device)
            model.conv_final = None
            pred_mat = model(inputs_test)
            loss_test = crit(pred_mat, labels_test)

            # Update loss count for test set
            running_loss_test.update(loss_test.item())
            pred_class = torch.argmax(pred_mat, dim=1)
            te_pred_pixel_count = torch.bincount(pred_class.flatten(), minlength=args.n_classes)
            te_pred_pixel_count_save += te_pred_pixel_count
            predictions_mat.extend(pred_mat.cpu().numpy())
            predictions.extend(pred_class.cpu().numpy())
            labels.extend(labels_test.cpu().numpy())

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    predictions_mat = np.asarray(predictions_mat)

    keep_test_idx = np.where(labels > -1)
    labels_vec = labels[keep_test_idx]
    predictions_vec = predictions[keep_test_idx]
    predictions_mat_sampled = predictions_mat[keep_test_idx[0], :, keep_test_idx[1], keep_test_idx[2]]
    fpr, tpr, roc_auc, roc_auc_macro = roc_curve_macro(labels_vec, predictions_mat_sampled)
    prob_mat, mat, f1_score, kappa = conf_mat(ylabel=labels_vec, ypred=predictions_vec, args=args)
    acc_bg, acc_se, acc_ot = prob_mat.diagonal()

    print('\n\n#############################################')
    print('#############################################')
    print('###############   TEST   ###################')
    print(
        'Epoch {0:3d} \t  Accuracy bg[0]: {1:.3f} \t Accuracy se[1]: {2:.3f} \t Accuracy ot[2]: {3:.3f}, Loss {4:.3f}'.format(
            epoch, acc_bg, acc_se, acc_ot, running_loss_test.avg))
    print('#############################################')
    print('#############################################\n\n')
    return running_loss_test.avg, prob_mat, mat, f1_score, kappa, predictions, predictions_mat, labels, te_pred_pixel_count_save.cpu().numpy(), fpr, tpr, roc_auc, roc_auc_macro


#
# def explain_input_single_channel(x, Z_hat, args):
#     assert args.gumble_sample > 0
#     Z_hat0 = Z_hat.view(Z_hat.size(0), Z_hat.size(1),
#                         int(np.sqrt(Z_hat.size(-1))),
#                         int(np.sqrt(Z_hat.size(-1))))
#     ## Upsampling
#     if args.chunk_size > 1:
#         mask = F.interpolate(Z_hat0,
#                                scale_factor = (args.chunk_size, args.chunk_size),
#                                mode = 'nearest')
#
#     ## feature selection
#     newsize = [x.size(0), args.gumble_sample]
#     newsize.extend(list(map(lambda x: x, x.size()[2:])))
#     masked_input = torch.mul(x.expand(torch.Size(newsize)), mask) # torch.Size([batch_size, num_sample, 256, 256])
#
#     newsize2 = [-1, 1]
#     newsize2.extend(newsize[2:])
#     masked_input = masked_input.view(torch.Size(newsize2))
#     return masked_input, mask
#
# def prior(var_size):
#     p = torch.ones(var_size[1]) / var_size[1]
#     p = p.view(1, var_size[1])
#     p_prior = p.expand(var_size)  # [batch-size, k, feature dim]
#
#     return p_prior
#
# def model_prediction(x, Z_hat, model, args):
#     assert args.gumble_sample > 0
#
#     Z_hat0 = Z_hat.view(Z_hat.size(0), Z_hat.size(1),
#                         int(np.sqrt(Z_hat.size(-1))),
#                         int(np.sqrt(Z_hat.size(-1))))
#
#     ## Upsampling
#     if args.chunk_size > 1:
#         Z_hat0 = F.interpolate(Z_hat0,
#                                scale_factor=(args.chunk_size, args.chunk_size),
#                                mode='nearest')
#
#     ## feature selection
#     newsize = [x.size(0), args.gumble_sample]
#     newsize.extend(list(map(lambda x: x, x.size()[2:])))
#     net = torch.mul(x.expand(torch.Size(newsize)), Z_hat0)  # torch.Size([batch_size, num_sample, d])
#
#     ## decode
#     newsize2 = [-1, 1]
#     newsize2.extend(newsize[2:])
#     net = net.view(torch.Size(newsize2))
#     pred = model(net)
#     pred = pred.view(-1, args.num_sample, pred.size(-1))
#     pred = pred.mean(1)
#     return pred
#
#
# def IB_supervised_train_exponly(dataloader_train_labeled, expr, model, opt_expr, opt_body, epoch, device, args):
#     # measure elapsed time
#     running_total_loss_labeled = AverageMeter()
#     running_kld_loss_labeled = AverageMeter()
#     running_class_loss_labeled = AverageMeter()
#     batch_time = AverageMeter()
#     masked_input_save = []
#     mask_save = []
#     input_save = []
#     pred_save = []
#     label_save = []
#
#     expr.train()
#     model.eval()
#     info_criterion = nn.KLDivLoss(reduction='sum')
#     tr_lab_pixel_count_save = np.zeros(args.n_classes)
#
#     with torch.autograd.set_detect_anomaly(True):
#         for i, (input_tensor, label) in enumerate(dataloader_train_labeled):
#             end = time.time()
#             label[label == -1] = 0  # seabed ignore
#             label_var = torch.autograd.Variable(label.to(device))
#
#             running_total_loss_batch = AverageMeter()
#             running_kld_loss_batch = AverageMeter()
#             running_class_loss_batch = AverageMeter()
#
#             mask_batch = []
#             masked_input_batch = []
#             pred_batch = []
#
#             for ch in range(args.total_channels):
#                 input_single = input_tensor[:, ch, :, :]
#                 input_single = input_single.view(args.batch_size, 1, args.window_dim, args.window_dim)
#                 input_single = input_single.double()
#                 input_var = torch.autograd.Variable(input_single.to(device))
#                 '''
#                 ##########################################
#                 EXPLAINER: by single channel
#                 ##########################################
#                 '''
#                 ## gumble softamx sampling (reparameterization trick)
#                 log_p_i = expr(input_var).double()
#                 log_p_i_ = log_p_i.view(log_p_i.size(0), 1, 1, -1)
#                 log_p_i_ = log_p_i_.expand(log_p_i_.size(0), args.gumble_sample, args.ib_k, log_p_i_.size(-1))
#                 C_dist = RelaxedOneHotCategorical(args.tau, log_p_i_)
#                 Z_hat = torch.max(C_dist.sample(), -2)[0]  # [batch-size, multi-shot, d, d]
#
#                 ## without sampling
#                 Z_fixed_size = log_p_i.unsqueeze(1).size()
#                 _, Z_fixed_idx = log_p_i.unsqueeze(1).topk(args.ib_k, dim=-1)  # batch * 1 * k
#                 Z_hat_fixed = idxtobool(Z_fixed_idx, Z_fixed_size)
#                 Z_hat_fixed = Z_hat_fixed.type(torch.float).to(device)
#
#
#                 '''
#                 ##########################################
#                 MASKED PREDICTION
#                 ##########################################
#                 '''
#                 masked_input, mask = explain_input_single_channel(input_var, Z_hat, args)
#                 logit = model(masked_input.double())
#                 logit = logit.view(-1, args.gumble_sample, args.n_classes, args.window_dim, args.window_dim)
#                 logit = logit.mean(1)
#
#                 masked_input_fixed, mask_fixed = explain_input_single_channel(input_var, Z_hat_fixed, args)
#                 logit_fixed = model(masked_input_fixed.double())
#                 logit_fixed = logit_fixed.view(-1, args.gumble_sample, args.n_classes, args.window_dim, args.window_dim)
#                 logit_fixed = logit_fixed.mean(1)
#                 pred_fixed = torch.argmax(logit_fixed, dim=1)
#
#                 with torch.no_grad():
#                     output_t = copy.copy(logit)
#                     softmax_output = F.softmax(output_t, dim=1)
#                     argmax_output = torch.argmax(softmax_output, dim=1)
#                     sup_output = argmax_output.cpu().numpy()
#                     tr_lab_pixel_count = np.bincount(sup_output.ravel(), minlength=args.n_classes)
#                     tr_lab_pixel_count_save += tr_lab_pixel_count
#                     predloss_weight = np.reciprocal(tr_lab_pixel_count_save / np.mean(tr_lab_pixel_count_save))
#                     predloss_weight = predloss_weight / np.sum(predloss_weight)
#
#                 class_criterion = nn.CrossEntropyLoss(weight=torch.Tensor(predloss_weight).double().to(device), reduction='sum')
#                 masked_input = masked_input.view(-1, args.gumble_sample, args.window_dim, args.window_dim)
#
#                 log_p_i_prior = prior(var_size=log_p_i.size()).double()
#                 log_p_i_prior = log_p_i_prior.to(device)
#
#                 class_loss = class_criterion(logit, label_var.long()).div(math.log(2)) / args.batch_size
#                 info_loss = args.ib_k * info_criterion(log_p_i, log_p_i_prior) / args.batch_size
#                 total_loss = torch.log(class_loss) + args.beta * info_loss
#
#                 opt_body.zero_grad()
#                 opt_expr.zero_grad()
#                 total_loss.backward()
#                 # opt_body.step()
#                 opt_expr.step()
#
#                 running_total_loss_batch.update(total_loss.item())
#                 running_kld_loss_batch.update(info_loss.item())
#                 running_class_loss_batch.update(class_loss.item())
#
#                 if args.gumble_sample == 1:
#                     mask = mask.view(-1, args.window_dim, args.window_dim)
#                     masked_input = masked_input.view(-1, args.window_dim, args.window_dim)
#
#                 mask_batch.append(mask.cpu().numpy())
#                 masked_input_batch.append(masked_input.cpu().numpy())
#                 pred_batch.append(softmax_output.cpu().numpy())
#
#             mask_batch = np.swapaxes(mask_batch, 0, 1)
#             masked_input_batch = np.swapaxes(masked_input_batch, 0, 1)
#             pred_batch = np.swapaxes(pred_batch, 0, 1)
#
#             mask_save.extend(mask_batch)
#             masked_input_save.extend(masked_input_batch)
#             input_save.extend(input_tensor.detach().cpu().numpy())
#             pred_save.extend(pred_batch)
#             label_save.extend(label.cpu().numpy())
#             batch_time.update(time.time() - end)
#
#             running_total_loss_labeled.update(running_total_loss_batch.avg)
#             running_kld_loss_labeled.update(running_kld_loss_batch.avg)
#             running_class_loss_labeled.update(running_class_loss_batch.avg)
#
#             if (i % args.display_count) == 0:
#                 print('TRAIN: Epoch {0} \t'
#                       ' {1} / {2} \t'
#                       'total_loss:{running_total_loss_labeled.val:.3f}  \t'
#                       'info_loss:{running_kld_loss_labeled.val:.3f}  \t'
#                       'class_loss:{running_class_loss_labeled.val:.3f}  \t'
#                       ' Batch Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
#                       .format(epoch, i, len(dataloader_train_labeled),
#                               running_total_loss_labeled=running_total_loss_labeled,
#                               running_kld_loss_labeled=running_kld_loss_labeled,
#                               running_class_loss_labeled=running_class_loss_labeled,
#                               batch_time=batch_time))
#
#     if epoch % args.save_epoch == 0:
#         torch.save({'epoch': epoch,
#                         'state_dict': model.state_dict(),
#                         'optimizer_body': opt_body.state_dict(),
#                         },
#                        os.path.join(args.exp_check, '%d_checkpoint.pth.tar' % epoch))
#         torch.save({'epoch': epoch,
#                     'state_dict': model.state_dict(),
#                     'optimizer_body': opt_body.state_dict(),
#                     },
#                    os.path.join(args.exp, 'checkpoint.pth.tar'))
#         torch.save({'epoch': epoch,
#                     'state_dict': expr.state_dict(),
#                     'optimizer_body': opt_expr.state_dict(),
#                     },
#                    os.path.join(args.exp_check, '%d_expr_checkpoint.pth.tar' % epoch))
#         torch.save({'epoch': epoch,
#                     'state_dict': expr.state_dict(),
#                     'optimizer_body': opt_expr.state_dict(),
#                     },
#                    os.path.join(args.exp, 'expr_checkpoint.pth.tar'))
#     else:
#         torch.save({'epoch': epoch,
#                     'state_dict': model.state_dict(),
#                     'optimizer_body': opt_body.state_dict(),
#                     },
#                    os.path.join(args.exp, 'checkpoint.pth.tar'))
#         torch.save({'epoch': epoch,
#                     'state_dict': expr.state_dict(),
#                     'optimizer_body': opt_expr.state_dict(),
#                     },
#                    os.path.join(args.exp, 'expr_checkpoint.pth.tar'))
#
#     return running_total_loss_labeled.avg, running_kld_loss_labeled.avg, running_class_loss_labeled.avg, masked_input_save, mask_save, pred_save, input_save, label_save
#
# def IB_supervised_train(dataloader_train_labeled, expr, model, opt_expr, opt_body, epoch, device, args):
#     # measure elapsed time
#     running_total_loss_labeled = AverageMeter()
#     running_kld_loss_labeled = AverageMeter()
#     running_class_loss_labeled = AverageMeter()
#     batch_time = AverageMeter()
#     masked_input_save = []
#     mask_save = []
#     input_save = []
#     pred_save = []
#     label_save = []
#
#     expr.train()
#     model.train()
#     info_criterion = nn.KLDivLoss(reduction='sum')
#     tr_lab_pixel_count_save = np.zeros(args.n_classes)
#
#     with torch.autograd.set_detect_anomaly(True):
#         for i, (input_tensor, label) in enumerate(dataloader_train_labeled):
#             end = time.time()
#             label[label == -1] = 0  # seabed ignore
#             label_var = torch.autograd.Variable(label.to(device))
#
#             running_total_loss_batch = AverageMeter()
#             running_kld_loss_batch = AverageMeter()
#             running_class_loss_batch = AverageMeter()
#
#             mask_batch = []
#             masked_input_batch = []
#             pred_batch = []
#
#             for ch in range(args.total_channels):
#                 input_single = input_tensor[:, ch, :, :]
#                 input_single = input_single.view(args.batch_size, 1, args.window_dim, args.window_dim)
#                 input_single = input_single.double()
#                 input_var = torch.autograd.Variable(input_single.to(device))
#                 '''
#                 ##########################################
#                 EXPLAINER: by single channel
#                 ##########################################
#                 '''
#                 ## gumble softamx sampling (reparameterization trick)
#                 log_p_i = expr(input_var).double()
#                 log_p_i_ = log_p_i.view(log_p_i.size(0), 1, 1, -1)
#                 log_p_i_ = log_p_i_.expand(log_p_i_.size(0), args.gumble_sample, args.ib_k, log_p_i_.size(-1))
#                 C_dist = RelaxedOneHotCategorical(args.tau, log_p_i_)
#                 Z_hat = torch.max(C_dist.sample(), -2)[0]  # [batch-size, multi-shot, d, d]
#
#                 ## without sampling
#                 Z_fixed_size = log_p_i.unsqueeze(1).size()
#                 _, Z_fixed_idx = log_p_i.unsqueeze(1).topk(args.ib_k, dim=-1)  # batch * 1 * k
#                 Z_hat_fixed = idxtobool(Z_fixed_idx, Z_fixed_size)
#                 Z_hat_fixed = Z_hat_fixed.type(torch.float).to(device)
#
#
#                 '''
#                 ##########################################
#                 MASKED PREDICTION
#                 ##########################################
#                 '''
#                 masked_input, mask = explain_input_single_channel(input_var, Z_hat, args)
#                 logit = model(masked_input.double())
#                 logit = logit.view(-1, args.gumble_sample, args.n_classes, args.window_dim, args.window_dim)
#                 logit = logit.mean(1)
#
#                 masked_input_fixed, mask_fixed = explain_input_single_channel(input_var, Z_hat_fixed, args)
#                 logit_fixed = model(masked_input_fixed.double())
#                 logit_fixed = logit_fixed.view(-1, args.gumble_sample, args.n_classes, args.window_dim, args.window_dim)
#                 logit_fixed = logit_fixed.mean(1)
#                 pred_fixed = torch.argmax(logit_fixed, dim=1)
#
#                 with torch.no_grad():
#                     output_t = copy.copy(logit)
#                     softmax_output = F.softmax(output_t, dim=1)
#                     argmax_output = torch.argmax(softmax_output, dim=1)
#                     sup_output = argmax_output.cpu().numpy()
#                     tr_lab_pixel_count = np.bincount(sup_output.ravel(), minlength=args.n_classes)
#                     tr_lab_pixel_count_save += tr_lab_pixel_count
#                     predloss_weight = np.reciprocal(tr_lab_pixel_count_save / np.mean(tr_lab_pixel_count_save))
#                     predloss_weight = predloss_weight / np.sum(predloss_weight)
#
#                 class_criterion = nn.CrossEntropyLoss(weight=torch.Tensor(predloss_weight).double().to(device), reduction='sum')
#                 masked_input = masked_input.view(-1, args.gumble_sample, args.window_dim, args.window_dim)
#
#                 log_p_i_prior = prior(var_size=log_p_i.size()).double()
#                 log_p_i_prior = log_p_i_prior.to(device)
#
#                 class_loss = class_criterion(logit, label_var.long()).div(math.log(2)) / args.batch_size
#                 info_loss = args.ib_k * info_criterion(log_p_i, log_p_i_prior) / args.batch_size
#                 total_loss = torch.log(class_loss) + args.beta * info_loss
#
#                 opt_body.zero_grad()
#                 opt_expr.zero_grad()
#                 total_loss.backward()
#                 opt_body.step()
#                 opt_expr.step()
#
#                 running_total_loss_batch.update(total_loss.item())
#                 running_kld_loss_batch.update(info_loss.item())
#                 running_class_loss_batch.update(class_loss.item())
#
#                 if args.gumble_sample == 1:
#                     mask = mask.view(-1, args.window_dim, args.window_dim)
#                     masked_input = masked_input.view(-1, args.window_dim, args.window_dim)
#
#                 mask_batch.append(mask.cpu().numpy())
#                 masked_input_batch.append(masked_input.cpu().numpy())
#                 pred_batch.append(softmax_output.cpu().numpy())
#
#             mask_batch = np.swapaxes(mask_batch, 0, 1)
#             masked_input_batch = np.swapaxes(masked_input_batch, 0, 1)
#             pred_batch = np.swapaxes(pred_batch, 0, 1)
#
#             mask_save.extend(mask_batch)
#             masked_input_save.extend(masked_input_batch)
#             input_save.extend(input_tensor.detach().cpu().numpy())
#             pred_save.extend(pred_batch)
#             label_save.extend(label.cpu().numpy())
#             batch_time.update(time.time() - end)
#
#             running_total_loss_labeled.update(running_total_loss_batch.avg)
#             running_kld_loss_labeled.update(running_kld_loss_batch.avg)
#             running_class_loss_labeled.update(running_class_loss_batch.avg)
#
#             if (i % args.display_count) == 0:
#                 print('TRAIN: Epoch {0} \t'
#                       ' {1} / {2} \t'
#                       'total_loss:{running_total_loss_labeled.val:.3f}  \t'
#                       'info_loss:{running_kld_loss_labeled.val:.3f}  \t'
#                       'class_loss:{running_class_loss_labeled.val:.3f}  \t'
#                       ' Batch Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
#                       .format(epoch, i, len(dataloader_train_labeled),
#                               running_total_loss_labeled=running_total_loss_labeled,
#                               running_kld_loss_labeled=running_kld_loss_labeled,
#                               running_class_loss_labeled=running_class_loss_labeled,
#                               batch_time=batch_time))
#
#     if epoch % args.save_epoch == 0:
#         torch.save({'epoch': epoch,
#                         'state_dict': model.state_dict(),
#                         'optimizer_body': opt_body.state_dict(),
#                         },
#                        os.path.join(args.exp_check, '%d_checkpoint.pth.tar' % epoch))
#         torch.save({'epoch': epoch,
#                     'state_dict': model.state_dict(),
#                     'optimizer_body': opt_body.state_dict(),
#                     },
#                    os.path.join(args.exp, 'checkpoint.pth.tar'))
#         torch.save({'epoch': epoch,
#                     'state_dict': expr.state_dict(),
#                     'optimizer_body': opt_expr.state_dict(),
#                     },
#                    os.path.join(args.exp_check, '%d_expr_checkpoint.pth.tar' % epoch))
#         torch.save({'epoch': epoch,
#                     'state_dict': expr.state_dict(),
#                     'optimizer_body': opt_expr.state_dict(),
#                     },
#                    os.path.join(args.exp, 'expr_checkpoint.pth.tar'))
#     else:
#         torch.save({'epoch': epoch,
#                     'state_dict': model.state_dict(),
#                     'optimizer_body': opt_body.state_dict(),
#                     },
#                    os.path.join(args.exp, 'checkpoint.pth.tar'))
#         torch.save({'epoch': epoch,
#                     'state_dict': expr.state_dict(),
#                     'optimizer_body': opt_expr.state_dict(),
#                     },
#                    os.path.join(args.exp, 'expr_checkpoint.pth.tar'))
#
#     return running_total_loss_labeled.avg, running_kld_loss_labeled.avg, running_class_loss_labeled.avg, masked_input_save, mask_save, pred_save, input_save, label_save
#
#
# def IB_supervised_test(dataloader_test_labeled, expr, model, epoch, device, args):
#     # measure elapsed time
#     running_total_loss_labeled = AverageMeter()
#     running_kld_loss_labeled = AverageMeter()
#     running_class_loss_labeled = AverageMeter()
#     batch_time = AverageMeter()
#     masked_input_save = []
#     mask_save = []
#     pred_save = []
#     input_save = []
#     label_save = []
#
#     expr.eval()
#     model.eval()
#     info_criterion = nn.KLDivLoss(reduction='sum')
#     te_pixel_count_save = np.zeros(args.n_classes)
#
#     with torch.no_grad():
#         for i, (input_tensor, label) in enumerate(dataloader_test_labeled):
#             end = time.time()
#             label[label == -1] = 0  # seabed ignore
#             label_var = label.long().to(device)
#             running_total_loss_batch = AverageMeter()
#             running_kld_loss_batch = AverageMeter()
#             running_class_loss_batch = AverageMeter()
#
#             mask_batch = []
#             masked_input_batch = []
#             pred_batch = []
#
#             for ch in range(args.total_channels):
#                 input_single = input_tensor[:, ch, :, :]
#                 input_single = input_single.view(args.batch_size, 1, args.window_dim, args.window_dim)
#                 input_single = input_single.double()
#                 input_var = input_single.double().to(device)
#                 '''
#                 ##########################################
#                 EXPLAINER: by single channel
#                 ##########################################
#                 '''
#                 ## gumble softamx sampling (reparameterization trick)
#                 log_p_i = expr(input_var).double()
#                 log_p_i_ = log_p_i.view(log_p_i.size(0), 1, 1, -1)
#                 log_p_i_ = log_p_i_.expand(log_p_i_.size(0), args.gumble_sample, args.ib_k, log_p_i_.size(-1))
#                 C_dist = RelaxedOneHotCategorical(args.tau, log_p_i_)
#                 Z_hat = torch.max(C_dist.sample(), -2)[0]  # [batch-size, multi-shot, d, d]
#                 '''
#                 ##########################################
#                 MASKED PREDICTION
#                 ##########################################
#                 '''
#                 masked_input, mask = explain_input_single_channel(input_var, Z_hat, args)
#                 logit = model(masked_input.double())
#                 logit = logit.view(-1, args.gumble_sample, args.n_classes, args.window_dim, args.window_dim)
#                 logit = logit.mean(1)
#
#                 with torch.no_grad():
#                     output_t = copy.copy(logit)
#                     softmax_output = F.softmax(output_t, dim=1)
#                     argmax_output = torch.argmax(softmax_output, dim=1)
#                     sup_output = argmax_output.cpu().numpy()
#                     te_pixel_count = np.bincount(sup_output.ravel(), minlength=args.n_classes)
#                     te_pixel_count_save += te_pixel_count
#                     predloss_weight = np.reciprocal(te_pixel_count_save / np.mean(te_pixel_count_save))
#                     predloss_weight = predloss_weight / np.sum(predloss_weight)
#
#                 class_criterion = nn.CrossEntropyLoss(weight=torch.Tensor(predloss_weight).double().to(device), reduction='sum')
#                 masked_input = masked_input.view(-1, args.gumble_sample, args.window_dim, args.window_dim)
#
#                 log_p_i_prior = prior(var_size=log_p_i.size()).double()
#                 log_p_i_prior = log_p_i_prior.to(device)
#
#                 class_loss = class_criterion(logit, label_var.long()).div(math.log(2)) / args.batch_size
#                 info_loss = args.ib_k * info_criterion(log_p_i, log_p_i_prior) / args.batch_size
#                 total_loss = class_loss + args.beta * info_loss
#
#                 running_total_loss_batch.update(total_loss.item())
#                 running_kld_loss_batch.update(info_loss.item())
#                 running_class_loss_batch.update(class_loss.item())
#
#                 if args.gumble_sample == 1:
#                     mask = mask.view(-1, args.window_dim, args.window_dim)
#                     masked_input = masked_input.view(-1, args.window_dim, args.window_dim)
#
#                 mask_batch.append(mask.cpu().numpy())
#                 masked_input_batch.append(masked_input.cpu().numpy())
#                 pred_batch.append(softmax_output.cpu().numpy())
#
#             mask_batch = np.swapaxes(mask_batch, 0, 1)
#             masked_input_batch = np.swapaxes(masked_input_batch, 0, 1)
#             pred_batch = np.swapaxes(pred_batch, 0, 1)
#
#             mask_save.extend(mask_batch)
#             masked_input_save.extend(masked_input_batch)
#             input_save.extend(input_tensor.detach().cpu().numpy())
#             pred_save.extend(pred_batch)
#             label_save.extend(label.cpu().numpy())
#             batch_time.update(time.time() - end)
#
#             running_total_loss_labeled.update(running_total_loss_batch.avg)
#             running_kld_loss_labeled.update(running_kld_loss_batch.avg)
#             running_class_loss_labeled.update(running_class_loss_batch.avg)
#
#             if (i % args.display_count) == 0:
#                 print('TEST: Epoch {0} \t'
#                       ' {1} / {2} \t'
#                       'total_loss:{running_total_loss_labeled.val:.3f}  \t'
#                       'info_loss:{running_kld_loss_labeled.val:.3f}  \t'
#                       'class_loss:{running_class_loss_labeled.val:.3f}  \t'
#                       ' Batch Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
#                       .format(epoch, i, len(dataloader_test_labeled),
#                               running_total_loss_labeled=running_total_loss_labeled,
#                               running_kld_loss_labeled=running_kld_loss_labeled,
#                               running_class_loss_labeled=running_class_loss_labeled,
#                               batch_time=batch_time))
#     return running_total_loss_labeled.avg, running_kld_loss_labeled.avg, running_class_loss_labeled.avg, masked_input_save, mask_save, pred_save, input_save, label_save
#
# def IB_prediction(epoch, label_save, pred_save, args):
#     # masked_input_save, mask_save, pred_save, label_save
#     roc_outs = IB_roc_curve_macro(label_save, pred_save, args=args)
#     conf_outs = IB_conf_mat(ylabel=label_save, ypred_four=pred_save, args=args)
#
#     print('###############   TEST   ###################')
#     for ch in range(args.total_channels):
#         [acc_bg, acc_se, acc_ot] = conf_outs[ch]['prob_mat'].diagonal()
#         print(
#             'Epoch {0:3d} \t Channel {1:1d} \t  Acc_bg: {2:.3f} \t Acc_se: {3:.3f} \t Acc_ot: {4:.3f}'.format(
#                 epoch, ch, acc_bg, acc_se, acc_ot))
#     print('#############################################')
#     return roc_outs, conf_outs
#
#
# def prediction(dataloader_test, model, crit, epoch, device, args):
#     running_loss_test = AverageMeter()
#     labels = []
#     pred_clusters = []
#     predictions = []
#     predictions_mat = []
#
#     torch.cuda.empty_cache()
#     te_pred_pixel_count_save = torch.zeros(args.n_classes).double().to(device)
#     te_cluster_pixel_count_save = torch.zeros(args.n_clusters).double().to(device)
#     model.eval()
#     with torch.no_grad():
#         for i, (input_tensor, label) in enumerate(dataloader_test):
#             # Load test data and transfer from numpy to pytorch
#             inputs_test = input_tensor.double().to(device)
#             labels_test = label.long().to(device)
#             model.conv_final = None
#             feature_test = model(inputs_test)
#             '''
#             ####################################################
#             ####################################################
#             # pred_cluster
#             ###################################################
#             ####################################################'''
#             model.conv_final = models.conv1x1(args.fd, args.n_clusters)
#             model.conv_final.weight.data.normal_(0, 0.01)
#             model.conv_final.bias.data.zero_()
#             model.conv_final = model.conv_final.double()
#             model.conv_final.to(device)
#             cluster_final_save = os.path.join(args.exp, 'cluster_final.pth.tar')
#             if os.path.isfile(cluster_final_save):
#                 category_layer_param = torch.load(cluster_final_save)
#                 model.conv_final.load_state_dict(category_layer_param)
#
#             pred_cluster = model.conv_final(feature_test)
#             pred_cluster = torch.argmax(F.softmax(pred_cluster, dim=1), dim=1)
#             te_cluster_pixel_count = torch.bincount(pred_cluster.flatten(), minlength=args.n_clusters)
#             te_cluster_pixel_count_save += te_cluster_pixel_count
#             pred_clusters.extend(pred_cluster.cpu().numpy())
#
#             '''
#             ####################################################
#             ####################################################
#             # pred_class
#             ###################################################
#             ####################################################'''
#             model.conv_final = None
#             model.conv_final = models.conv1x1(args.fd, args.n_classes)
#             model.conv_final.weight.data.normal_(0, 0.01)
#             model.conv_final.bias.data.zero_()
#             model.conv_final = model.conv_final.double()
#             model.conv_final.to(device)
#
#             # param load to conv_final
#             conv_final_save = os.path.join(args.exp, 'conv_final.pth.tar')
#             if os.path.isfile(conv_final_save):
#                 category_layer_param = torch.load(conv_final_save)
#                 model.conv_final.load_state_dict(category_layer_param)
#             else:
#                 print('\n\n########## NO available conv_final parameters. ##############\n\n')
#
#             # Evaluate test data
#             pred_class = model.conv_final(feature_test)
#             loss_test = crit(pred_class, labels_test)
#
#             # Update loss count for test set
#             running_loss_test.update(loss_test.item())
#             pred_mat = F.softmax(pred_class, dim=1)
#             pred_class = torch.argmax(pred_mat, dim=1)
#             te_pred_pixel_count = torch.bincount(pred_class.flatten(), minlength=args.n_classes)
#             te_pred_pixel_count_save += te_pred_pixel_count
#             predictions_mat.extend(pred_mat.cpu().numpy())
#             predictions.extend(pred_class.cpu().numpy())
#             labels.extend(labels_test.cpu().numpy())
#
#     labels = np.asarray(labels)
#     predictions = np.asarray(predictions)
#     predictions_mat = np.asarray(predictions_mat)
#     pred_clusters = np.asarray(pred_clusters)
#
#     keep_test_idx = np.where(labels > -1)
#     labels_vec = labels[keep_test_idx]
#     predictions_vec = predictions[keep_test_idx]
#     predictions_mat_sampled = predictions_mat[keep_test_idx[0], :, keep_test_idx[1], keep_test_idx[2]]
#     fpr, tpr, roc_auc, roc_auc_macro = roc_curve_macro(labels_vec, predictions_mat_sampled)
#     prob_mat, mat, f1_score, kappa = conf_mat(ylabel=labels_vec, ypred=predictions_vec, args=args)
#     acc_bg, acc_se, acc_ot = prob_mat.diagonal()
#
#     print('\n\n#############################################')
#     print('#############################################')
#     print('###############   TEST   ###################')
#     print(
#         'Epoch {0:3d} \t  Accuracy bg[0]: {1:.3f} \t Accuracy se[1]: {2:.3f} \t Accuracy ot[2]: {3:.3f}, Loss {4:.3f}'.format(
#             epoch, acc_bg, acc_se, acc_ot, running_loss_test.avg))
#     print('#############################################')
#     print('#############################################\n\n')
#     return running_loss_test.avg, prob_mat, mat, f1_score, kappa, predictions, pred_clusters, predictions_mat, labels, te_pred_pixel_count_save.cpu().numpy(), te_cluster_pixel_count_save.cpu().numpy(), fpr, tpr, roc_auc, roc_auc_macro
#
#
#
#
#
# def supervised_train_loader_weighted(dataloader_train_labeled, model, opt_body, opt_final, epoch, device, args):
#     # measure elapsed time
#     running_sup_loss_labeled = AverageMeter()
#     batch_time = AverageMeter()
#     sup_output_save = []
#     label_save = []
#
#     model.conv_final = None
#     model.conv_final = models.conv1x1(args.fd, args.n_classes)
#     model.conv_final.weight.data.normal_(0, 0.01)
#     model.conv_final.bias.data.zero_()
#     model.conv_final = model.conv_final.double()
#     model.conv_final.to(device)
#
#     # param load to conv_final
#     conv_final_save = os.path.join(args.exp, 'conv_final.pth.tar')
#     if os.path.isfile(conv_final_save):
#         category_layer_param = torch.load(conv_final_save)
#         model.conv_final.load_state_dict(category_layer_param)
#     else:
#         print('\n\n########## NO available conv_final parameters. ##############\n\n')
#
#     tr_lab_pixel_count_save = np.zeros(args.n_classes)
#     model.train()
#     with torch.autograd.set_detect_anomaly(True):
#         for i, (input_tensor, label) in enumerate(dataloader_train_labeled):
#             end = time.time()
#             input_var = torch.autograd.Variable(input_tensor.to(device))
#             label[label == -1] = 0  # seabed ignore
#             output = model(input_var)
#
#             with torch.no_grad():
#                 output_t = copy.copy(output)
#                 sup_output = torch.argmax(F.softmax(output_t, dim=1), dim=1)
#                 sup_output = sup_output.cpu().numpy()
#                 tr_lab_pixel_count = np.bincount(sup_output.ravel(), minlength=args.n_classes)
#                 tr_lab_pixel_count_save += tr_lab_pixel_count
#                 predloss_weight = np.reciprocal(tr_lab_pixel_count_save / np.mean(tr_lab_pixel_count_save))
#                 predloss_weight = predloss_weight/np.sum(predloss_weight)
#
#             crit = nn.CrossEntropyLoss(weight=torch.Tensor(predloss_weight).double().to(device))
#             label_var = torch.autograd.Variable(label.to(device))
#             sup_loss = crit(output, label_var.long())
#
#             opt_final.zero_grad()
#             opt_body.zero_grad()
#             sup_loss.backward()
#             opt_final.step()
#             opt_body.step()
#
#             running_sup_loss_labeled.update(sup_loss.item())
#             sup_output_save.extend(sup_output)
#             label_save.extend(label.cpu().numpy())
#             batch_time.update(time.time() - end)
#
#             if (i % args.display_count) == 0:
#                 print('LABELED: Epoch {0} \t'
#                       ' {1} / {2} \t'
#                       'sup_loss:{running_sup_loss_labeled.val:.3f}  \t'
#                       'sup_loss_avg:{running_sup_loss_labeled.avg:.3f} \t'
#                       ' Batch Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
#                       .format(epoch, i, len(dataloader_train_labeled),
#                               running_sup_loss_labeled=running_sup_loss_labeled,
#                               batch_time=batch_time))
#
#     torch.save(model.conv_final.state_dict(), os.path.join(args.exp, 'conv_final.pth.tar'))
#     torch.save(model.conv_final.state_dict(), os.path.join(args.exp_check, '%d_conv_final.pth.tar' % epoch))
#     model.conv_final = None
#     return running_sup_loss_labeled.avg, sup_output_save, label_save, tr_lab_pixel_count_save
#
# def generate_pseudolabels(dataloader_unlabeled, model, deepcluster, device, args):
#     model.eval()
#     # forward
#     with torch.no_grad():
#         inputs = []
#         outputs = []
#         for i, (input_tensor, label) in enumerate(dataloader_unlabeled):
#             model.conv_final = None
#             input_tensor.double()
#             input_var = torch.autograd.Variable(input_tensor.to(device))
#             output = model(input_var)
#             outputs.extend(output.cpu().numpy())
#             inputs.extend(input_tensor.cpu().numpy())
#     inputs = np.stack(inputs, axis=0)
#     outputs = np.stack(outputs, axis=0)
#     dim_output = np.shape(outputs)
#     pseudolabels = np.zeros((dim_output[0], dim_output[2], dim_output[3]))
#
#     '''Cluster vectors of length 64 (first few patches of args.num_patches_clustering)'''
#     iter_count = int(np.ceil(len(outputs) / args.num_patches_clustering))
#     torch.cuda.empty_cache()
#     for i in range(iter_count):
#         embeddings = outputs[args.num_patches_clustering * i:args.num_patches_clustering * (i + 1)]
#         if i == 0:
#             _, _ = deepcluster.cluster(embeddings.transpose(0, 2, 3, 1).reshape(-1, 64), verbose=args.verbose)
#         else:
#             _ = deepcluster.assign(embeddings.transpose(0, 2, 3, 1).reshape(-1, 64), verbose=args.verbose)
#         pseudolabels[args.num_patches_clustering * i:args.num_patches_clustering * i + len(embeddings)] = \
#             deepcluster.images_dist_lists[0].reshape(len(embeddings), dim_output[2], dim_output[3])
#     return inputs, pseudolabels.astype(int), deepcluster
#
# # def selfsupervised_train_loader(model, inputs, pseudolabels, crit, opt_body, opt_final, epoch, device, args):
# #     dataset_selfsupervised = Dataset_single(
# #         inputs,
# #         pseudolabels,
# #         augmentation_function=None,
# #         label_transform_function=None,
# #         data_transform_function=None)
# #
# #     dataloader_train_pseudolabeled = DataLoader(dataset_selfsupervised,
# #                                           batch_size=args.batch_size,
# #                                           shuffle=True,
# #                                           num_workers=args.num_workers,
# #                                           worker_init_fn=np.random.seed,
# #                                           drop_last=False,
# #                                           pin_memory=True)
# #     torch.cuda.empty_cache()
# #
# #     # measure elapsed time
# #     running_self_loss = AverageMeter()
# #     batch_time = AverageMeter()
# #     self_output_save = []
# #     self_pseudolabel_save = []
# #
# #     model.conv_final = None
# #     model.conv_final = models.conv1x1(args.fd, args.n_clusters)
# #     model.conv_final.weight.data.normal_(0, 0.01)
# #     model.conv_final.bias.data.zero_()
# #     model.conv_final = model.conv_final.double()
# #     model.conv_final.to(device)
# #
# #     # param load to conv_final
# #     cluster_final_save = os.path.join(args.exp, 'cluster_final.pth.tar')
# #     if os.path.isfile(cluster_final_save):
# #         category_layer_param = torch.load(cluster_final_save)
# #         model.conv_final.load_state_dict(category_layer_param)
# #
# #     model.train()
# #     for i, (input_tensor, pseudo_label) in enumerate(dataloader_train_pseudolabeled):
# #         end = time.time()
# #         input_var = torch.autograd.Variable(input_tensor.to(device))
# #         label_var = torch.autograd.Variable(pseudo_label.to(device, non_blocking=True))
# #         output = model(input_var)
# #         selfsupervised_loss = crit(output, label_var.long())
# #
# #         # compute gradient and do SGD step
# #         opt_body.zero_grad()
# #         opt_final.zero_grad()
# #         selfsupervised_loss.backward()
# #         opt_final.step()
# #         opt_body.step()
# #
# #         output_softmax = torch.argmax(F.softmax(output, dim=1), dim=1)
# #         running_self_loss.update(selfsupervised_loss.item())
# #         self_output_save.extend(output_softmax.cpu().numpy())
# #         self_pseudolabel_save.extend(pseudo_label.cpu().numpy())
# #         batch_time.update(time.time() - end)
# #
# #         if (i % args.display_count) == 0:
# #             torch.cuda.empty_cache()
# #             print('PSEUDOLABELED: Epoch {0} \t'
# #                   ' {1} / {2} \t'
# #                   'self_loss:{running_self_loss.val:.3f}  \t'
# #                   'self_loss_avg:{running_self_loss.avg:.3f} \t'
# #                   ' Batch Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
# #                   .format(epoch, i, len(dataloader_train_pseudolabeled),
# #                           running_self_loss=running_self_loss,
# #                           batch_time=batch_time))
# #
# #     torch.save(model.conv_final.state_dict(), os.path.join(args.exp, 'cluster_final.pth.tar'))
# #     torch.save(model.conv_final.state_dict(), os.path.join(args.exp_check, '%d_cluster_final.pth.tar' % epoch))
# #     return running_self_loss.avg, self_output_save, self_pseudolabel_save
#
# def prediction(dataloader_test, model, crit, epoch, device, args):
#     running_loss_test = AverageMeter()
#     labels = []
#     pred_clusters = []
#     predictions = []
#     predictions_mat = []
#
#     torch.cuda.empty_cache()
#     te_pred_pixel_count_save = torch.zeros(args.n_classes).double().to(device)
#     te_cluster_pixel_count_save = torch.zeros(args.n_clusters).double().to(device)
#     model.eval()
#     with torch.no_grad():
#         for i, (input_tensor, label) in enumerate(dataloader_test):
#             # Load test data and transfer from numpy to pytorch
#             inputs_test = input_tensor.double().to(device)
#             labels_test = label.long().to(device)
#             model.conv_final = None
#             feature_test = model(inputs_test)
#             '''
#             ####################################################
#             ####################################################
#             # pred_cluster
#             ###################################################
#             ####################################################'''
#             model.conv_final = models.conv1x1(args.fd, args.n_clusters)
#             model.conv_final.weight.data.normal_(0, 0.01)
#             model.conv_final.bias.data.zero_()
#             model.conv_final = model.conv_final.double()
#             model.conv_final.to(device)
#             cluster_final_save = os.path.join(args.exp, 'cluster_final.pth.tar')
#             if os.path.isfile(cluster_final_save):
#                 category_layer_param = torch.load(cluster_final_save)
#                 model.conv_final.load_state_dict(category_layer_param)
#
#             pred_cluster = model.conv_final(feature_test)
#             pred_cluster = torch.argmax(F.softmax(pred_cluster, dim=1), dim=1)
#             te_cluster_pixel_count = torch.bincount(pred_cluster.flatten(), minlength=args.n_clusters)
#             te_cluster_pixel_count_save += te_cluster_pixel_count
#             pred_clusters.extend(pred_cluster.cpu().numpy())
#
#             '''
#             ####################################################
#             ####################################################
#             # pred_class
#             ###################################################
#             ####################################################'''
#             model.conv_final = None
#             model.conv_final = models.conv1x1(args.fd, args.n_classes)
#             model.conv_final.weight.data.normal_(0, 0.01)
#             model.conv_final.bias.data.zero_()
#             model.conv_final = model.conv_final.double()
#             model.conv_final.to(device)
#
#             # param load to conv_final
#             conv_final_save = os.path.join(args.exp, 'conv_final.pth.tar')
#             if os.path.isfile(conv_final_save):
#                 category_layer_param = torch.load(conv_final_save)
#                 model.conv_final.load_state_dict(category_layer_param)
#             else:
#                 print('\n\n########## NO available conv_final parameters. ##############\n\n')
#
#             # Evaluate test data
#             pred_class = model.conv_final(feature_test)
#             loss_test = crit(pred_class, labels_test)
#
#             # Update loss count for test set
#             running_loss_test.update(loss_test.item())
#             pred_mat = F.softmax(pred_class, dim=1)
#             pred_class = torch.argmax(pred_mat, dim=1)
#             te_pred_pixel_count = torch.bincount(pred_class.flatten(), minlength=args.n_classes)
#             te_pred_pixel_count_save += te_pred_pixel_count
#             predictions_mat.extend(pred_mat.cpu().numpy())
#             predictions.extend(pred_class.cpu().numpy())
#             labels.extend(labels_test.cpu().numpy())
#
#     labels = np.asarray(labels)
#     predictions = np.asarray(predictions)
#     predictions_mat = np.asarray(predictions_mat)
#     pred_clusters = np.asarray(pred_clusters)
#
#     keep_test_idx = np.where(labels > -1)
#     labels_vec = labels[keep_test_idx]
#     predictions_vec = predictions[keep_test_idx]
#     predictions_mat_sampled = predictions_mat[keep_test_idx[0], :, keep_test_idx[1], keep_test_idx[2]]
#     fpr, tpr, roc_auc, roc_auc_macro = roc_curve_macro(labels_vec, predictions_mat_sampled)
#     prob_mat, mat, f1_score, kappa = conf_mat(ylabel=labels_vec, ypred=predictions_vec, args=args)
#     acc_bg, acc_se, acc_ot = prob_mat.diagonal()
#
#     print('\n\n#############################################')
#     print('#############################################')
#     print('###############   TEST   ###################')
#     print(
#         'Epoch {0:3d} \t  Accuracy bg[0]: {1:.3f} \t Accuracy se[1]: {2:.3f} \t Accuracy ot[2]: {3:.3f}, Loss {4:.3f}'.format(
#             epoch, acc_bg, acc_se, acc_ot, running_loss_test.avg))
#     print('#############################################')
#     print('#############################################\n\n')
#     return running_loss_test.avg, prob_mat, mat, f1_score, kappa, predictions, pred_clusters, predictions_mat, labels, te_pred_pixel_count_save.cpu().numpy(), te_cluster_pixel_count_save.cpu().numpy(), fpr, tpr, roc_auc, roc_auc_macro
#
# def prediction_2019(dataloader_test, model, crit, epoch, device, args):
#     pred_clusters = []
#     predictions = []
#
#     torch.cuda.empty_cache()
#     model.eval()
#     with torch.no_grad():
#         for i, (input_tensor, label) in enumerate(dataloader_test):
#             # Load test data and transfer from numpy to pytorch
#             inputs_test = input_tensor.double().to(device)
#             labels_test = label.long().to(device)
#             model.conv_final = None
#             feature_test = model(inputs_test)
#             '''
#             ####################################################
#             ####################################################
#             # pred_cluster
#
#             ###################################################
#             ####################################################'''
#             model.conv_final = models.conv1x1(args.fd, args.n_clusters)
#             model.conv_final.weight.data.normal_(0, 0.01)
#             model.conv_final.bias.data.zero_()
#             model.conv_final = model.conv_final.double()
#             model.conv_final.to(device)
#             cluster_final_save = os.path.join(args.exp, 'cluster_final.pth.tar')
#             if os.path.isfile(cluster_final_save):
#                 category_layer_param = torch.load(cluster_final_save)
#                 model.conv_final.load_state_dict(category_layer_param)
#
#             pred_cluster = model.conv_final(feature_test)
#             pred_cluster = torch.argmax(F.softmax(pred_cluster, dim=1), dim=1)
#             pred_clusters.extend(pred_cluster.cpu().numpy())
#
#             '''
#             ####################################################
#             ####################################################
#             # pred_class
#             ###################################################
#             ####################################################'''
#             model.conv_final = None
#             model.conv_final = models.conv1x1(args.fd, args.n_classes)
#             model.conv_final.weight.data.normal_(0, 0.01)
#             model.conv_final.bias.data.zero_()
#             model.conv_final = model.conv_final.double()
#             model.conv_final.to(device)
#
#             # param load to conv_final
#             conv_final_save = os.path.join(args.exp, 'conv_final.pth.tar')
#             if os.path.isfile(conv_final_save):
#                 category_layer_param = torch.load(conv_final_save)
#                 model.conv_final.load_state_dict(category_layer_param)
#             else:
#                 print('\n\n########## NO available conv_final parameters. ##############\n\n')
#
#             # Evaluate test data
#             pred_class = model.conv_final(feature_test)
#
#             # Update loss count for test set
#             pred_mat = F.softmax(pred_class, dim=1)
#             pred_class = torch.argmax(pred_mat, dim=1)
#             predictions.extend(pred_class.cpu().numpy())
#     predictions = np.asarray(predictions)
#     pred_clusters = np.asarray(pred_clusters)
#     return predictions, pred_clusters
#
#
#
# def prediction_earlystop(dataloader_test, model, crit, epoch, device, args):
#     running_loss_test = AverageMeter()
#     labels = []
#     pred_clusters = []
#     predictions = []
#     predictions_mat = []
#
#     torch.cuda.empty_cache()
#     te_pred_pixel_count_save = torch.zeros(args.n_classes).double().to(device)
#     te_cluster_pixel_count_save = torch.zeros(args.n_clusters).double().to(device)
#     model.eval()
#     with torch.no_grad():
#         for i, (input_tensor, label) in enumerate(dataloader_test):
#             # Load test data and transfer from numpy to pytorch
#             inputs_test = input_tensor.double().to(device)
#             labels_test = label.long().to(device)
#             model.conv_final = None
#             feature_test = model(inputs_test)
#             '''
#             ####################################################
#             ####################################################
#             # pred_cluster
#             ###################################################
#             ####################################################'''
#             model.conv_final = models.conv1x1(args.fd, args.n_clusters)
#             model.conv_final.weight.data.normal_(0, 0.01)
#             model.conv_final.bias.data.zero_()
#             model.conv_final = model.conv_final.double()
#             model.conv_final.to(device)
#             cluster_final_save = os.path.join(args.exp_check, '%d_cluster_final.pth.tar' % epoch)
#             if os.path.isfile(cluster_final_save):
#                 category_layer_param = torch.load(cluster_final_save)
#                 model.conv_final.load_state_dict(category_layer_param)
#
#             pred_cluster = model.conv_final(feature_test)
#             pred_cluster = torch.argmax(F.softmax(pred_cluster, dim=1), dim=1)
#             te_cluster_pixel_count = torch.bincount(pred_cluster.flatten(), minlength=args.n_clusters)
#             te_cluster_pixel_count_save += te_cluster_pixel_count
#             pred_clusters.extend(pred_cluster.cpu().numpy())
#
#             '''
#             ####################################################
#             ####################################################
#             # pred_class
#             ###################################################
#             ####################################################'''
#             model.conv_final = None
#             model.conv_final = models.conv1x1(args.fd, args.n_classes)
#             model.conv_final.weight.data.normal_(0, 0.01)
#             model.conv_final.bias.data.zero_()
#             model.conv_final = model.conv_final.double()
#             model.conv_final.to(device)
#
#             # param load to conv_final
#             conv_final_save = os.path.join(args.exp, 'conv_final.pth.tar')
#             if os.path.isfile(conv_final_save):
#                 category_layer_param = torch.load(conv_final_save)
#                 model.conv_final.load_state_dict(category_layer_param)
#             else:
#                 print('\n\n########## NO available conv_final parameters. ##############\n\n')
#
#             # Evaluate test data
#             pred_class = model.conv_final(feature_test)
#             loss_test = crit(pred_class, labels_test)
#
#             # Update loss count for test set
#             running_loss_test.update(loss_test.item())
#             pred_mat = F.softmax(pred_class, dim=1)
#             pred_class = torch.argmax(pred_mat, dim=1)
#             te_pred_pixel_count = torch.bincount(pred_class.flatten(), minlength=args.n_classes)
#             te_pred_pixel_count_save += te_pred_pixel_count
#             predictions_mat.extend(pred_mat.cpu().numpy())
#             predictions.extend(pred_class.cpu().numpy())
#             labels.extend(labels_test.cpu().numpy())
#
#     labels = np.asarray(labels)
#     predictions = np.asarray(predictions)
#     predictions_mat = np.asarray(predictions_mat)
#     pred_clusters = np.asarray(pred_clusters)
#
#     keep_test_idx = np.where(labels > -1)
#     labels_vec = labels[keep_test_idx]
#     predictions_vec = predictions[keep_test_idx]
#     predictions_mat_sampled = predictions_mat[keep_test_idx[0], :, keep_test_idx[1], keep_test_idx[2]]
#     fpr, tpr, roc_auc, roc_auc_macro = roc_curve_macro(labels_vec, predictions_mat_sampled)
#     prob_mat, mat, f1_score, kappa = conf_mat(ylabel=labels_vec, ypred=predictions_vec, args=args)
#     acc_bg, acc_se, acc_ot = prob_mat.diagonal()
#
#     print('\n\n#############################################')
#     print('#############################################')
#     print('###############   TEST   ###################')
#     print(
#         'Epoch {0:3d} \t  Accuracy bg[0]: {1:.3f} \t Accuracy se[1]: {2:.3f} \t Accuracy ot[2]: {3:.3f}, Loss {4:.3f}'.format(
#             epoch, acc_bg, acc_se, acc_ot, running_loss_test.avg))
#     print('#############################################')
#     print('#############################################\n\n')
#     return running_loss_test.avg, prob_mat, mat, f1_score, kappa, predictions, pred_clusters, predictions_mat, labels, te_pred_pixel_count_save.cpu().numpy(), te_cluster_pixel_count_save.cpu().numpy(), fpr, tpr, roc_auc, roc_auc_macro
#
#
# def prediction_INTP(dataloader_INTP, dataset_test, test_idx, model, device, args):
#     labels = []
#     inputs_200khz = []
#     clusters_pred = []
#     predictions = []
#     predictions_softmax = []
#
#     torch.cuda.empty_cache()
#     model.train()
#     for i, (input_tensor, label) in enumerate(dataloader_INTP):
#         d_target, l_target = dataset_test[test_idx] # torch.float64, torch.int16
#         input_tensor = torch.cat([torch.DoubleTensor(np.expand_dims(d_target, 0)), input_tensor], axis=0)
#         label = torch.cat([torch.ShortTensor(np.expand_dims(l_target, 0)), label], axis=0)
#         inputs_test = input_tensor.double().to(device)
#         labels_test = label.long().to(device)
#         model.conv_final = None
#         feature_test = model(inputs_test)
#         '''
#         ####################################################
#         ####################################################
#         # pred_cluster
#         ###################################################
#         ####################################################'''
#         model.conv_final = models.conv1x1(args.fd, args.n_clusters)
#         model.conv_final.weight.data.normal_(0, 0.01)
#         model.conv_final.bias.data.zero_()
#         model.conv_final = model.conv_final.double()
#         model.conv_final.to(device)
#         cluster_final_save = os.path.join(args.exp, 'cluster_final.pth.tar')
#         if os.path.isfile(cluster_final_save):
#             category_layer_param = torch.load(cluster_final_save)
#             model.conv_final.load_state_dict(category_layer_param)
#
#         pred_cluster = model.conv_final(feature_test)
#         pred_cluster = torch.argmax(F.softmax(pred_cluster, dim=1), dim=1)
#         clusters_pred.append(pred_cluster[0].detach().cpu().numpy())
#         inputs_200khz.append(inputs_test[0][-1].cpu().detach().numpy())
#         labels.append(labels_test[0].cpu().detach().numpy())
#
#         '''
#         ####################################################
#         ####################################################
#         # pred_class
#         ###################################################
#         ####################################################'''
#         model.conv_final = None
#         model.conv_final = models.conv1x1(args.fd, args.n_classes)
#         model.conv_final.weight.data.normal_(0, 0.01)
#         model.conv_final.bias.data.zero_()
#         model.conv_final = model.conv_final.double()
#         model.conv_final.to(device)
#
#         # param load to conv_final
#         conv_final_save = os.path.join(args.exp, 'conv_final.pth.tar')
#         if os.path.isfile(conv_final_save):
#             category_layer_param = torch.load(conv_final_save)
#             model.conv_final.load_state_dict(category_layer_param)
#         else:
#             print('\n\n########## NO available conv_final parameters. ##############\n\n')
#
#         # Evaluate test data
#         pred_class_raw = model.conv_final(feature_test)
#
#         # Update loss count for test set
#         pred_softmax = F.softmax(pred_class_raw, dim=1)
#         pred_class = torch.argmax(pred_softmax, dim=1)
#         predictions_softmax.append(pred_softmax[0].detach().cpu().numpy())
#         predictions.append(pred_class[0].detach().cpu().numpy())
#
#     inputs_200khz = np.asarray(inputs_200khz[0])
#     labels = np.asarray(labels[0])
#     predictions = np.asarray(predictions)
#     predictions_softmax = np.asarray(predictions_softmax)
#     clusters_pred = np.asarray(clusters_pred)
#
#     return inputs_200khz, labels, predictions, predictions_softmax, clusters_pred
#
# def supervised_and_self_supervised_train(dataloader_train_labeled, model, inputs, pseudolabels, crit_self, opt_body, opt_final, epoch, device, args):
#     dataset_selfsupervised = Dataset_single(
#         inputs,
#         pseudolabels,
#         augmentation_function=None,
#         label_transform_function=None,
#         data_transform_function=None)
#
#     dataloader_train_pseudolabeled = DataLoader(dataset_selfsupervised,
#                                           batch_size=args.batch_size,
#                                           shuffle=True,
#                                           num_workers=args.num_workers,
#                                           worker_init_fn=np.random.seed,
#                                           drop_last=False,
#                                           pin_memory=True)
#
#     torch.cuda.empty_cache()
#
#     # measure elapsed time
#     running_self_loss = AverageMeter()
#     running_sup_loss_labeled = AverageMeter()
#     batch_time = AverageMeter()
#     self_output_save = []
#     self_pseudolabel_save = []
#     sup_output_save = []
#     label_save = []
#     tr_lab_pixel_count_save = np.zeros(args.n_classes)
#
#     model.train()
#     with torch.autograd.set_detect_anomaly(True):
#         for i, ((pinput_tensor, pseudo_label), (input_tensor, label)) in enumerate(zip(dataloader_train_pseudolabeled, dataloader_train_labeled)):
#             '''
#            ###############################################################################
#            SELF-SUPERVISED
#            ###############################################################################'''
#             end = time.time()
#             model.conv_final = None
#             model.conv_final = models.conv1x1(args.fd, args.n_clusters)
#             model.conv_final.weight.data.normal_(0, 0.01)
#             model.conv_final.bias.data.zero_()
#             model.conv_final = model.conv_final.double()
#             model.conv_final.to(device)
#
#             # param load to conv_final
#             cluster_final_save = os.path.join(args.exp, 'cluster_final.pth.tar')
#             if os.path.isfile(cluster_final_save):
#                 category_layer_param = torch.load(cluster_final_save)
#                 model.conv_final.load_state_dict(category_layer_param)
#             else:
#                 print(i, '########## NO available cluster_final parameters. ##############')
#
#             pinput_var = torch.autograd.Variable(pinput_tensor.to(device))
#             plabel_var = torch.autograd.Variable(pseudo_label.to(device, non_blocking=True))
#             poutput = model(pinput_var)
#             selfsupervised_loss = crit_self(poutput, plabel_var.long())
#
#             # compute gradient and do SGD step
#             opt_body.zero_grad()
#             opt_final.zero_grad()
#             selfsupervised_loss.backward()
#             opt_final.step()
#             opt_body.step()
#
#             poutput_softmax = torch.argmax(F.softmax(poutput, dim=1), dim=1)
#             running_self_loss.update(selfsupervised_loss.item())
#             self_output_save.extend(poutput_softmax.cpu().numpy())
#             self_pseudolabel_save.extend(pseudo_label.cpu().numpy())
#             batch_time.update(time.time() - end)
#
#             if (i % args.display_count) == 0:
#                 torch.cuda.empty_cache()
#                 print('PSEUDOLABELED: Epoch {0} \t'
#                       ' {1} / {2} \t'
#                       'self_loss:{running_self_loss.val:.3f}  \t'
#                       'self_loss_avg:{running_self_loss.avg:.3f} \t'
#                       ' Batch Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
#                       .format(epoch, i, len(dataloader_train_pseudolabeled),
#                               running_self_loss=running_self_loss,
#                               batch_time=batch_time))
#             torch.save(model.conv_final.state_dict(), os.path.join(args.exp, 'cluster_final.pth.tar'))
#             torch.save(model.conv_final.state_dict(), os.path.join(args.exp_check, '%d_cluster_final.pth.tar' % epoch))
#
#             '''
#             ###############################################################################
#             SUPERVISED
#             ###############################################################################'''
#             end = time.time()
#             model.conv_final = None
#             model.conv_final = models.conv1x1(args.fd, args.n_classes)
#             model.conv_final.weight.data.normal_(0, 0.01)
#             model.conv_final.bias.data.zero_()
#             model.conv_final = model.conv_final.double()
#             model.conv_final.to(device)
#
#             # param load to conv_final
#             conv_final_save = os.path.join(args.exp, 'conv_final.pth.tar')
#             if os.path.isfile(conv_final_save):
#                 category_layer_param = torch.load(conv_final_save)
#                 model.conv_final.load_state_dict(category_layer_param)
#             else:
#                 print(i, '########## NO available conv_final parameters. ##############')
#
#             input_var = torch.autograd.Variable(input_tensor.to(device))
#             label[label == -1] = 0  # seabed ignore
#             output = model(input_var)
#
#             with torch.no_grad():
#                 output_t = copy.copy(output)
#                 sup_output = torch.argmax(F.softmax(output_t, dim=1), dim=1)
#                 sup_output = sup_output.cpu().numpy()
#                 tr_lab_pixel_count = np.bincount(sup_output.ravel(), minlength=args.n_classes)
#                 tr_lab_pixel_count_save += tr_lab_pixel_count
#                 predloss_weight = np.reciprocal(tr_lab_pixel_count_save / np.mean(tr_lab_pixel_count_save))
#                 predloss_weight = predloss_weight/np.sum(predloss_weight)
#
#             crit_sup = nn.CrossEntropyLoss(weight=torch.Tensor(predloss_weight).double().to(device))
#             label_var = torch.autograd.Variable(label.to(device))
#             sup_loss = crit_sup(output, label_var.long())
#
#             opt_final.zero_grad()
#             opt_body.zero_grad()
#             sup_loss.backward()
#             opt_final.step()
#             opt_body.step()
#
#             running_sup_loss_labeled.update(sup_loss.item())
#             sup_output_save.extend(sup_output)
#             label_save.extend(label.cpu().numpy())
#             batch_time.update(time.time() - end)
#
#             if (i % args.display_count) == 0:
#                 print('LABELED: Epoch {0} \t'
#                       ' {1} / {2} \t'
#                       'sup_loss:{running_sup_loss_labeled.val:.3f}  \t'
#                       'sup_loss_avg:{running_sup_loss_labeled.avg:.3f} \t'
#                       ' Batch Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
#                       .format(epoch, i, len(dataloader_train_labeled),
#                               running_sup_loss_labeled=running_sup_loss_labeled,
#                               batch_time=batch_time))
#             torch.save(model.conv_final.state_dict(), os.path.join(args.exp, 'conv_final.pth.tar'))
#             torch.save(model.conv_final.state_dict(), os.path.join(args.exp_check, '%d_conv_final.pth.tar' % epoch))
#     model.conv_final = None
#     return running_sup_loss_labeled.avg, sup_output_save, label_save, tr_lab_pixel_count_save, running_self_loss.avg, self_output_save, self_pseudolabel_save
#

