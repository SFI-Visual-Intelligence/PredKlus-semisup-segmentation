import numpy as np
import os
import sklearn.metrics as skm
from itertools import cycle
from sklearn.preprocessing import label_binarize
from scipy import interp
import matplotlib.pyplot as plt
from scipy.special import softmax

def plot_conf(epoch, prob_mat, mat, f1_score, kappa, args):
    legend = ['BG', 'SE', 'OT']
    color_criteria = int(np.sum(mat) * 0.6)
    prob_color_criteria = 0.6
    fig, ax = plt.subplots()
    plt.rc('font', size=14)
    im = ax.imshow(mat, cmap='Greys')
    ax.set_xticks(np.arange(len(legend)))
    ax.set_yticks(np.arange(len(legend)))
    ax.set_xticklabels(legend)
    ax.set_yticklabels(legend)

    # Loop over data dimensions and create text annotations.
    for i in range(args.n_classes):
        for j in range(args.n_classes):
            if int(mat[i, j]) >= color_criteria:
                text = ax.text(j, i, int(mat[i, j]),
                               ha="center", va="center", color="white", fontsize=12, fontweight='bold')
            else:
                text = ax.text(j, i, int(mat[i, j]),
                               ha="center", va="center", color="black", fontsize=12, fontweight='bold')

    ax.set_title("OURS, Epoch %d, F1: %0.4f (%s)" % (epoch, f1_score, args.f1_avg))
    fig.tight_layout()
    plt.savefig(
        os.path.join(args.pred_test, '%d_confmat.jpg' % epoch))
    plt.close()

    # probablisitic conf.
    fig, ax = plt.subplots()
    im = ax.imshow(prob_mat, cmap='Greys', vmax=1, vmin=0)
    ax.set_xticks(np.arange(len(legend)))
    ax.set_yticks(np.arange(len(legend)))
    ax.set_xticklabels(legend)
    ax.set_yticklabels(legend)

    # Loop over data dimensions and create text annotations.
    for i in range(args.n_classes):
        for j in range(args.n_classes):
            if prob_mat[i, j] >= prob_color_criteria:
                text = ax.text(j, i, '%0.4f' % prob_mat[i, j],
                               ha="center", va="center", color="white", fontsize=18, fontweight='bold')
            else:
                text = ax.text(j, i, '%0.4f' % prob_mat[i, j],
                               ha="center", va="center", color="black", fontsize=18, fontweight='bold')

    ax.set_title("E %d, F1 %0.4f, kappa %0.4f" % (epoch, f1_score, kappa))
    fig.tight_layout()
    plt.savefig(os.path.join(args.pred_test,
                             '%d_prob_confmat.jpg' % epoch))
    plt.close()

def plot_conf_best(epoch, prob_mat, mat, f1_score, kappa, args):
    import matplotlib.pyplot as plt
    legend = ['BG', 'SE', 'OT']
    color_criteria = int(np.sum(mat) * 0.6)
    prob_color_criteria = 0.6
    fig, ax = plt.subplots()
    plt.rc('font', size=14)
    im = ax.imshow(mat, cmap='Greys')
    ax.set_xticks(np.arange(len(legend)))
    ax.set_yticks(np.arange(len(legend)))
    ax.set_xticklabels(legend)
    ax.set_yticklabels(legend)

    # Loop over data dimensions and create text annotations.
    for i in range(args.n_classes):
        for j in range(args.n_classes):
            if int(mat[i, j]) >= color_criteria:
                text = ax.text(j, i, int(mat[i, j]),
                               ha="center", va="center", color="white", fontsize=12, fontweight='bold')
            else:
                text = ax.text(j, i, int(mat[i, j]),
                               ha="center", va="center", color="black", fontsize=12, fontweight='bold')

    ax.set_title("E %d, F1 %0.4f, kappa %0.4f" % (epoch, f1_score, kappa))
    fig.tight_layout()
    plt.savefig(
        os.path.join(args.pred_test, 'best_%d_confmat.jpg' % epoch))
    plt.close()

    # probablisitic conf.
    fig, ax = plt.subplots()
    im = ax.imshow(prob_mat, cmap='Greys', vmax=1, vmin=0)
    ax.set_xticks(np.arange(len(legend)))
    ax.set_yticks(np.arange(len(legend)))
    ax.set_xticklabels(legend)
    ax.set_yticklabels(legend)

    # Loop over data dimensions and create text annotations.
    for i in range(args.n_classes):
        for j in range(args.n_classes):
            if prob_mat[i, j] >= prob_color_criteria:
                text = ax.text(j, i, '%0.4f' % prob_mat[i, j],
                               ha="center", va="center", color="white", fontsize=18, fontweight='bold')
            else:
                text = ax.text(j, i, '%0.4f' % prob_mat[i, j],
                               ha="center", va="center", color="black", fontsize=18, fontweight='bold')

    ax.set_title("OURS, Epoch %d, F1: %0.4f (%s)" % (epoch, f1_score, args.f1_avg))
    fig.tight_layout()
    plt.savefig(os.path.join(args.pred_test,
                             'best_%d_prob_confmat.jpg' % epoch))
    plt.close()


def IB_conf_mat(ylabel, ypred_four, args):
    conf_outs = {}
    mat = np.zeros([args.n_classes, args.n_classes])
    # gt: axis 0, pred: axis 1
    ylabel_vec = np.ravel(ylabel)
    ypred_four_transpose = np.swapaxes(ypred_four, 0, 1)
    for ch in range(args.total_channels):
        conf_outs[ch] = {}
        ypred_vec = np.ravel(np.argmax(softmax(ypred_four_transpose[ch], axis=1), axis=1))
        for (gt, pd) in zip(ylabel_vec, ypred_vec):
            mat[gt, pd] += 1
        f1_score = skm.f1_score(ylabel_vec, ypred_vec, average=args.f1_avg)
        kappa = skm.cohen_kappa_score(ylabel_vec, ypred_vec)
        prob_mat = np.zeros([args.n_classes, args.n_classes])
        # gt: axis 0, pred: axis 1
        for i in range(args.n_classes):
            prob_mat[i] = mat[i] / np.sum(mat[i])
        conf_outs[ch]['f1'] = f1_score
        conf_outs[ch]['kappa'] = kappa
        conf_outs[ch]['prob_mat'] = prob_mat
    return conf_outs

def IB_roc_curve_macro(ylabel, ypred_four, args):
    # Binarize the output
    ylabel_vec = np.ravel(ylabel)
    y_test = label_binarize(ylabel_vec, classes=[0, 1, 2])
    ypred_four_transpose = np.swapaxes(ypred_four, 0, 1)
    roc_outs = {}
    # Compute ROC curve and ROC area for each class and channel
    for ch in range(args.total_channels):
        roc_outs[ch] = {}
        y_score = np.reshape(softmax(ypred_four_transpose[ch], axis=1), (-1, 3))
        roc_outs[ch]['fpr'] = {}
        roc_outs[ch]['tpr'] = {}
        roc_outs[ch]['roc_auc'] = {}
        for i in range(args.n_classes):
            roc_outs[ch]['fpr'][i], roc_outs[ch]['tpr'][i], _ = skm.roc_curve(y_test[:, i], y_score[:, i])
            roc_outs[ch]['roc_auc'][i] = skm.auc(roc_outs[ch]['fpr'][i], roc_outs[ch]['tpr'][i])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([roc_outs[ch]['fpr'][i] for i in range(args.n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(args.n_classes):
            mean_tpr += interp(all_fpr, roc_outs[ch]['fpr'][i], roc_outs[ch]['tpr'][i])

        # Finally average it and compute AUC
        mean_tpr /= args.n_classes
        roc_outs[ch]['fpr']["macro"] = all_fpr
        roc_outs[ch]['tpr']["macro"] = mean_tpr
        roc_outs[ch]['roc_auc']["macro"] = skm.auc(roc_outs[ch]['fpr']["macro"], roc_outs[ch]['tpr']["macro"])

        macro_roc_auc_ovo = skm.roc_auc_score(y_test, y_score, average="macro", multi_class="ovo")
        weighted_roc_auc_ovo = skm.roc_auc_score(y_test, y_score, average="weighted", multi_class="ovo")
        macro_roc_auc_ovr = skm.roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")
        weighted_roc_auc_ovr = skm.roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")

        roc_outs[ch]['macro_roc_auc_ovo'] = macro_roc_auc_ovo
        roc_outs[ch]['weighted_roc_auc_ovo'] = weighted_roc_auc_ovo
        roc_outs[ch]['macro_roc_auc_ovr'] = macro_roc_auc_ovr
        roc_outs[ch]['weighted_roc_auc_ovr'] = weighted_roc_auc_ovr


        print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
              "(weighted by prevalence)"
              .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
        print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
              "(weighted by prevalence)"
              .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

        roc_auc_macro = {'macro_ovo': macro_roc_auc_ovo,
                         'weighted_ovo':weighted_roc_auc_ovo,
                         'macro_ovr': macro_roc_auc_ovr,
                         'weighted_ovr': weighted_roc_auc_ovr}
    return roc_outs


def roc_curve_macro(label_vec, y_score, n_classes=3):
    # Binarize the output
    y_test = label_binarize(label_vec, classes=[0, 1, 2])
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = skm.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = skm.auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = skm.auc(fpr["macro"], tpr["macro"])

    macro_roc_auc_ovo = skm.roc_auc_score(y_test, y_score, average="macro", multi_class="ovo")
    weighted_roc_auc_ovo = skm.roc_auc_score(y_test, y_score, multi_class="ovo", average="weighted")
    macro_roc_auc_ovr = skm.roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")
    weighted_roc_auc_ovr = skm.roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")
    print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
          "(weighted by prevalence)"
          .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
    print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
          "(weighted by prevalence)"
          .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

    roc_auc_macro = {'macro_ovo': macro_roc_auc_ovo,
                     'weighted_ovo':weighted_roc_auc_ovo,
                     'macro_ovr': macro_roc_auc_ovr,
                     'weighted_ovr': weighted_roc_auc_ovr}
    return fpr, tpr, roc_auc, roc_auc_macro

def plot_macro(fpr, tpr, roc_auc, epoch, args, n_classes=3):
    lw = 2
    legend = ['BG', 'SE', 'OT']
    # Plot all ROC curves
    plt.figure(figsize=(10, 10))

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='darkorange', linestyle=':', linewidth=4)

    colors = cycle(['deeppink', 'blue', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                 ''.format(legend[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('E %d, ROC-AUC curve' % epoch)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.pred_test,
                             '%d_roc_auc_curve.jpg' % epoch))
    plt.close()

def plot_macro_best(fpr, tpr, roc_auc, epoch, args, n_classes=3):
    lw = 2
    legend = ['BG', 'SE', 'OT']
    # Plot all ROC curves
    plt.figure(figsize=(10, 10))

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='darkorange', linestyle=':', linewidth=4)

    colors = cycle(['deeppink', 'blue', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                 ''.format(legend[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('E %d, ROC-AUC curve' % epoch)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.pred_test,
                             'best_%d_roc_auc_curve.jpg' % epoch))
    plt.close()

def conf_mat(ylabel, ypred, args):
    mat = np.zeros([args.n_classes, args.n_classes])
    # gt: axis 0, pred: axis 1
    for (gt, pd) in zip(ylabel, ypred):
        mat[gt, pd] += 1
    f1_score = skm.f1_score(ylabel, ypred, average=args.f1_avg)
    kappa = skm.cohen_kappa_score(ylabel, ypred)
    prob_mat = np.zeros([args.n_classes, args.n_classes])
    # gt: axis 0, pred: axis 1
    for i in range(args.n_classes):
        prob_mat[i] = mat[i] / np.sum(mat[i])
    return prob_mat, mat, f1_score, kappa
