import os
# import random
import time
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
# import faiss
# import faiss.contrib.torch_utils

from image import imread, imwrite, imshow
from fold import fold2d, unfold2d
from nnlookup import nn_lookup2d, nn_lookup_soft2d, l2_dist
from utils import view_as_2d, view_2d_as
from Resizer import Resizer
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def get_pyramid(image, depth, ratio, verbose=False):
    device = image.device
    max_layer = depth
    pyramid = []
    ratio = _pair(ratio)
    curr = image
    pyramid.append(curr)
    for j in range(max_layer):
        if verbose:
            print(curr.shape)
            imshow(curr)
        shape = [1, image.shape[1], ceil(image.shape[2] * (ratio[0])**(j+1)), ceil(image.shape[3] * (ratio[1])**(j+1))]
        resizer = Resizer(curr.shape, ratio, shape).to(device=device)
        curr = resizer(curr)
        pyramid.append(curr)
    return pyramid


SOFTMIN_DTYPE = torch.float32


def extract_kvs(keys, values, patch_size, use_padding):
    if isinstance(keys, torch.Tensor):
        keys = [keys]
    if isinstance(values, torch.Tensor):
        values = [values]
    assert len(keys) == len(values)
    assert all(k.shape == v.shape for k, v in zip(keys, values))
    key = torch.cat([view_as_2d(unfold2d(k, patch_size, use_padding=use_padding))[0] for k in keys], dim=0).contiguous()
    value = torch.cat([view_as_2d(unfold2d(v, patch_size, use_padding=use_padding))[0] for v in values],
                      dim=0).contiguous()
    #     return key.to(dtype=NN_DTYPE), value.to(dtype=NN_DTYPE)
    return key, value


@torch.no_grad()
def torch_patch_weight_min_dist(query, key, dist_fn, alpha, batch_size):
    kbatch = (batch_size + query.shape[0] - 1) // query.shape[0]
    min_dists = []
    for j in range((key.shape[0] + kbatch - 1) // kbatch):
        idx = nn_lookup2d(key[j * kbatch:(j + 1) * kbatch], query, dist_fn=dist_fn)
        match_query = torch.index_select(query, 0, idx)
        min_dists.append((key[j * kbatch:(j + 1) * kbatch] - match_query).pow(2).mean(1))
    min_dist = torch.cat(min_dists, dim=0)
    weight = 1 / (alpha + min_dist)
    return weight


@torch.no_grad()
def torch_patch_nn(query, key, value, patch_size, dist_fn, bidirectional=False, alpha=1., temperature=None,
                   reduce='mean', use_padding=False, batch_size=2 ** 28):
    # reshape input
    # dtype = query.dtype  # XXX
    query = unfold2d(query, patch_size, use_padding=use_padding)
    query, size, ndim = view_as_2d(query)
    # query = query.to(dtype=NN_DTYPE)
    # print(query.shape, key.shape)
    # perform nn-search
    idxs = []
    qbatch = (batch_size + key.shape[0] - 1) // key.shape[0]
    kbatch = (batch_size + query.shape[0] - 1) // query.shape[0]
    if bidirectional:
        weight = torch_patch_weight_min_dist(query, key, dist_fn, alpha, batch_size).view(1, -1)
    else:
        weight = None
    for i in range((query.shape[0] + qbatch - 1) // qbatch):
        if temperature is None:
            idxs.append(nn_lookup2d(query[i * qbatch:(i + 1) * qbatch], key, weight=weight, dist_fn=dist_fn))
        else:
            idxs.append(nn_lookup_soft2d(query[i * qbatch:(i + 1) * qbatch], key, weight=weight, dist_fn=dist_fn,
                                         temperature=temperature, dtype=SOFTMIN_DTYPE))
    idx = torch.cat(idxs, dim=0)
    result = torch.index_select(value, 0, idx)

    # reshape output
    result = view_2d_as(result, size, ndim)
    # result = result.to(dtype=dtype)
    output = fold2d(result, reduce=reduce, use_padding=use_padding)
    return output


class TorchPatchNN(nn.Module):
    def __init__(self, patch_size, dist_fn, bidirectional=False, alpha=1., temperature=None, reduce='mean',
                 use_padding=False, batch_size=2 ** 28):
        super().__init__()
        self._patch_size = _pair(patch_size)
        self._dist_fn = dist_fn
        self._bidirectional = bidirectional
        self._alpha = alpha
        self._temperature = temperature
        self._reduce = reduce
        self._use_padding = use_padding
        self._batch_size = batch_size

    def forward(self, query, key, value):
        return torch_patch_nn(
            query=query,
            key=key,
            value=value,
            patch_size=self._patch_size,
            dist_fn=self._dist_fn,
            bidirectional=self._bidirectional,
            alpha=self._alpha,
            temperature=self._temperature,
            reduce=self._reduce,
            use_padding=self._use_padding,
            batch_size=self._batch_size)

    def extract_kvs(self, keys, values):
        key, value = extract_kvs(keys, values, self._patch_size, use_padding=self._use_padding)
        return key, value

    def __repr__(self):
        return '{}(patch_size={}, dist_fn={}, bidirectional={}, alpha={}, reduce="{}", use_padding={}, batch_size={})'.format(
            self.__class__.__name__,
            self._patch_size,
            self._dist_fn.__name__,
            self._bidirectional,
            self._alpha,
            self._reduce,
            self._use_padding,
            self._batch_size)


def new_image_generation(pnn, src_pyramid, dst_pyramid, mask_pyr, ratio=4 / 3, noise_std=0.75, noise_decay=None, num_iters=10,
                         top_level=9, verbose=False):
    to = {'device': src_pyramid[0].device, 'dtype': src_pyramid[0].dtype}
    start = time.time()
    new_im = dst_pyramid[top_level]
    new_im = new_im + noise_std * torch.randn_like(new_im)*mask_pyr[top_level]
    if noise_decay is None:
        noise_decay = 0.
    for l in range(top_level, -1, -1):
        start = time.time()
        resizer = Resizer(src_pyramid[l + 1].shape, ratio, src_pyramid[l].shape).to(**to)
        if l == top_level:
            curr = src_pyramid[l]
            prev = src_pyramid[l]
        else:
            curr = src_pyramid[l]
            prev = resizer(src_pyramid[l + 1])
        # key, value = extract_kvs(curr, prev, pnn._patch_size, use_padding=pnn._use_padding)
        key_index, value = pnn.extract_kvs(keys=prev, values=curr)  # TODO: change keys to prev! # create_index_l2(key)
        for k in range(num_iters if l != top_level else 1):
            # new_im = pnn(new_im, keys=prev, values=curr)
            start = time.time()
            new_im = pnn(new_im, key_index, value)
            # print(l, k, '%.2fms' % (1000 * (time.time() - start),))
        #         print(new_im.shape)
        #         imshow(new_im)
        if l > 0:
            resizer = Resizer(dst_pyramid[l].shape, ratio, dst_pyramid[l - 1].shape).to(**to)
            new_im = resizer(new_im)
            new_im = new_im + (noise_std * noise_decay ** (top_level + 1 - l)) * torch.randn_like(new_im)

    if verbose:
        print('Total time: %.2f[s]' % (time.time() - start,))
        imshow(new_im)

    return new_im