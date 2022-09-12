# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['PIC', 'Kmeans', 'cluster_assign', 'arrange_clustering']


# def pil_loader(path):
#     """Loads an image.
#     Args:
#         path (string): path to image file
#     Returns:
#         Image
#     """
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')


class ReassignedDataset(data.Dataset):
    # ReassignedDataset(image_indexes, pseudolabels, dataset, None)
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))} # provide index per pseudoclass
        images = []
        for j, idx in enumerate(image_indexes):
            img = dataset[idx]
            # path = dataset[idx][0]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((img, pseudolabel, idx))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        img, pseudolabel, idx = self.imgs[index]
        # path, pseudolabel = self.imgs[index]
        # img = pil_loader(path)
        # if self.transform is not None:
        #     img = self.transform(img)
        return img, pseudolabel, idx

    def __len__(self):
        return len(self.imgs)


def preprocess_features(npdata, pca=None):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    nan_location = np.isnan(npdata)
    inf_location = np.isinf(npdata)
    if (not np.allclose(nan_location, 0)) or (not np.allclose(inf_location, 0)):
        print('before_Astype_Feature NaN or Inf found. Nan count: ', np.sum(nan_location), ' Inf count: ', np.sum(inf_location))
        print('######################  break  ##################################')
        return npdata

    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    nan_location = np.isnan(npdata)
    inf_location = np.isinf(npdata)
    if (not np.allclose(nan_location, 0)) or (not np.allclose(inf_location, 0)):
        print('after_Astype_Feature NaN or Inf found. Nan count: ', np.sum(nan_location), ' Inf count: ', np.sum(inf_location))
        print('######################  break  ##################################')
        return npdata

    if pca is not None:
        # Apply PCA-whitening with Faiss
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata

def cluster_assign(images_lists, dataset):
    # clustering.cluster_assign(deepcluster.images_lists, img_label_pair_train)
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)                  # [24 2 5 4] [6 9 87 54]
        pseudolabels.extend([cluster] * len(images))  # [0 0 0 0]  [1 1 1 1]

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    # t = transforms.Compose([transforms.RandomResizedCrop(224),
    #                         transforms.RandomHorizontalFlip(),
    #                         transforms.ToTensor(),
    #                         normalize])

    return ReassignedDataset(image_indexes, pseudolabels, dataset, None)
    # return ReassignedDataset(image_indexes, pseudolabels, dataset, t)

def run_kmeans(x, nmb_clusters, centroids, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape
    # print(n_data, d)
    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # print(centroids)
    if centroids is not None:
        faiss.copy_array_to_vector(centroids.ravel(), clus.centroids)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)
    clus.niter = 20
    # clus.min_points_per_centroid = 5
    clus.max_points_per_centroid = int(n_data / nmb_clusters * 1.5)
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    # flat_config = faiss.GpuIndexIVFFlatConfig()   # IVF
    flat_config.useFloat16 = False
    flat_config.device = 0
    # index = faiss.GpuIndexIVFFlat(res, d, nmb_clusters, faiss.METRIC_L2, flat_config)  # faiss.Metric_INNER_PRODUCT,
    index = faiss.GpuIndexFlatL2(res, d, flat_config)
    # index = faiss.GpuIndexIP(res, d, flat_config)  # Inner product between samples
    # perform the training
    clus.train(x, index)
    D, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    centroids = faiss.vector_to_array(clus.centroids).reshape(clus.k, clus.d)

    if verbose:
        print('k-means loss evolution: {0}'.format(losses))
    return [int(n[0]) for n in I], losses[-1], np.array([(d[0]) for d in D]), centroids, index
    # return [int(n[0]) for n in I], np.array([(d[0]) for d in D])

def assign_to_clusters(x, index):
    D, I = index.search(x, 1)
    return [int(n[0]) for n in I], np.array([(d[0]) for d in D])

def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    def __init__(self, k, pca):
        self.k = k
        self.pca = pca
        self.centroids = None
        self.index = None

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data, pca=self.pca)
        nan_location = np.isnan(xb)
        inf_location = np.isinf(xb)
        if (not np.allclose(nan_location, 0)) or (not np.allclose(inf_location, 0)):
            print('PCA: NaN or Inf found. Nan count: ', np.sum(nan_location), ' Inf count: ', np.sum(inf_location))
            loss = None
            return loss, xb
        else:
            # cluster the data
            I, loss, D, self.centroids, self.index = run_kmeans(xb, self.k, self.centroids, verbose)
            # I, D = run_kmeans(self.xb, self.k, verbose)
            self.images_dist_lists = [np.array(I), D]
            self.images_lists = [[] for i in range(self.k)]
            for i in range(len(data)):
                self.images_lists[I[i]].append(i)
            if verbose:
                print('k-means time: {0:.0f} s'.format(time.time() - end))
            return loss, xb

    def assign(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data, pca=self.pca)
        nan_location = np.isnan(xb)
        inf_location = np.isinf(xb)
        if (not np.allclose(nan_location, 0)) or (not np.allclose(inf_location, 0)):
            print('PCA: NaN or Inf found. Nan count: ', np.sum(nan_location), ' Inf count: ', np.sum(inf_location))
            loss = None
            return loss, xb
        else:
            # cluster the data
            I, D = assign_to_clusters(xb, self.index)

            self.images_dist_lists = [np.array(I), D]
            self.images_lists = [[] for i in range(self.k)]
            for i in range(len(data)):
                self.images_lists[I[i]].append(i)
            if verbose:
                print('k-means time: {0:.0f} s'.format(time.time() - end))
            return xb



# def make_adjacencyW(I, D, sigma):
#     """Create adjacency matrix with a Gaussian kernel.
#     Args:
#         I (numpy array): for each vertex the ids to its nnn linked vertices
#                   + first column of identity.
#         D (numpy array): for each data the l2 distances to its nnn linked vertices
#                   + first column of zeros.
#         sigma (float): Bandwidth of the Gaussian kernel.
#
#     Returns:
#         csr_matrix: affinity matrix of the graph.
#     """
#     V, k = I.shape
#     k = k - 1
#     indices = np.reshape(np.delete(I, 0, 1), (1, -1))
#     indptr = np.multiply(k, np.arange(V + 1))
#
#     def exp_ker(d):
#         return np.exp(-d / sigma**2)
#
#     exp_ker = np.vectorize(exp_ker)
#     res_D = exp_ker(D)
#     data = np.reshape(np.delete(res_D, 0, 1), (1, -1))
#     adj_matrix = csr_matrix((data[0], indices[0], indptr), shape=(V, V))
#     return adj_matrix
#
#
# def run_pic(I, D, sigma, alpha):
#     """Run PIC algorithm"""
#     a = make_adjacencyW(I, D, sigma)
#     graph = a + a.transpose()
#     cgraph = graph
#     nim = graph.shape[0]
#
#     W = graph
#     t0 = time.time()
#
#     v0 = np.ones(nim) / nim
#
#     # power iterations
#     v = v0.astype('float32')
#
#     t0 = time.time()
#     dt = 0
#     for i in range(200):
#         vnext = np.zeros(nim, dtype='float32')
#
#         vnext = vnext + W.transpose().dot(v)
#
#         vnext = alpha * vnext + (1 - alpha) / nim
#         # L1 normalize
#         vnext /= vnext.sum()
#         v = vnext
#
#         if i == 200 - 1:
#             clust = find_maxima_cluster(W, v)
#
#     return [int(i) for i in clust]
#
#
# def find_maxima_cluster(W, v):
#     n, m = W.shape
#     assert (n == m)
#     assign = np.zeros(n)
#     # for each node
#     pointers = list(range(n))
#     for i in range(n):
#         best_vi = 0
#         l0 = W.indptr[i]
#         l1 = W.indptr[i + 1]
#         for l in range(l0, l1):
#             j = W.indices[l]
#             vi = W.data[l] * (v[j] - v[i])
#             if vi > best_vi:
#                 best_vi = vi
#                 pointers[i] = j
#     n_clus = 0
#     cluster_ids = -1 * np.ones(n)
#     for i in range(n):
#         if pointers[i] == i:
#             cluster_ids[i] = n_clus
#             n_clus = n_clus + 1
#     for i in range(n):
#         # go from pointers to pointers starting from i until reached a local optim
#         current_node = i
#         while pointers[current_node] != current_node:
#             current_node = pointers[current_node]
#
#         assign[i] = cluster_ids[current_node]
#         assert (assign[i] >= 0)
#     return assign


# class PIC(object):
#     """Class to perform Power Iteration Clustering on a graph of nearest neighbors.
#         Args:
#             args: for consistency with k-means init
#             sigma (float): bandwidth of the Gaussian kernel (default 0.2)
#             nnn (int): number of nearest neighbors (default 5)
#             alpha (float): parameter in PIC (default 0.001)
#             distribute_singletons (bool): If True, reassign each singleton to
#                                       the cluster of its closest non
#                                       singleton nearest neighbors (up to nnn
#                                       nearest neighbors).
#         Attributes:
#             images_lists (list of list): for each cluster, the list of image indexes
#                                          belonging to this cluster
#     """
#
#     def __init__(self, args=None, sigma=0.2, nnn=5, alpha=0.001, distribute_singletons=True):
#         self.sigma = sigma
#         self.alpha = alpha
#         self.nnn = nnn
#         self.distribute_singletons = distribute_singletons
#
#     def cluster(self, data, verbose=False):
#         end = time.time()
#
#         # preprocess the data
#         xb = preprocess_features(data)
#
#         # construct nnn graph
#         I, D = make_graph(xb, self.nnn)
#
#         # run PIC
#         clust = run_pic(I, D, self.sigma, self.alpha)
#         images_lists = {}
#         for h in set(clust):
#             images_lists[h] = []
#         for data, c in enumerate(clust):
#             images_lists[c].append(data)
#
#         # allocate singletons to clusters of their closest NN not singleton
#         if self.distribute_singletons:
#             clust_NN = {}
#             for i in images_lists:
#                 # if singleton
#                 if len(images_lists[i]) == 1:
#                     s = images_lists[i][0]
#                     # for NN
#                     for n in I[s, 1:]:
#                         # if NN is not a singleton
#                         if not len(images_lists[clust[n]]) == 1:
#                             clust_NN[s] = n
#                             break
#             for s in clust_NN:
#                 del images_lists[clust[s]]
#                 clust[s] = clust[clust_NN[s]]
#                 images_lists[clust[s]].append(s)
#
#         self.images_lists = []
#         for c in images_lists:
#             self.images_lists.append(images_lists[c])
#
#         if verbose:
#             print('pic time: {0:.0f} s'.format(time.time() - end))
#         return 0
# def make_graph(xb, nnn):
#     """Builds a graph of nearest neighbors.
#     Args:
#         xb (np.array): data
#         nnn (int): number of nearest neighbors
#     Returns:
#         list: for each data the list of ids to its nnn nearest neighbors
#         list: for each data the list of distances to its nnn NN
#     """
#     N, dim = xb.shape
#
#     # we need only a StandardGpuResources per GPU
#     res = faiss.StandardGpuResources()
#
#     # L2
#     flat_config = faiss.GpuIndexFlatConfig()
#     flat_config.device = int(torch.cuda.device_count()) - 1
#     index = faiss.GpuIndexFlatL2(res, dim, flat_config)
#     index.add(xb)
#     D, I = index.search(xb, nnn + 1)
#     return I, D
