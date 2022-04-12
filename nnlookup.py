import torch
import torch.nn.functional as F

from utils import view_as_2d, view_2d_as


__all__ = ['nn_lookup', 'nn_lookup_soft', 'l2_dist', 'inner_prod_dist', 'nn_lookup2d', 'nn_lookup_soft2d']


def l2_dist(x, y):
    xx = x.pow(2).sum(1).view(-1, 1)
    yy = y.pow(2).sum(1).view(1, -1)
    xy = torch.einsum('id,jd->ij', x, y)
    dists = xx + yy - 2 * xy
    return dists


def inner_prod_dist(x, y):
    return torch.einsum('id,jd->ij', x, y)


def nn_lookup2d(queries, keys, dist_fn=l2_dist, weight=None):
    # queries (2D): n, d
    # keys (2D): m, d
    # output (2D): n, d
    dists = dist_fn(queries, keys)
    if weight is not None:
        dists = dists * weight
    idxs = dists.argmin(1)
    return idxs  # torch.index_select(keys, 1, idxs)


def nn_lookup_soft2d(queries, keys, dist_fn=l2_dist, weight=None, temperature=1., dtype=None):
    # queries (2D): n, d
    # keys (2D): m, d
    # output (2D): n, d
    dists = dist_fn(queries, keys)
    if weight is not None:
        dists = dists * weight
    probs = F.softmin(dists * temperature, dim=1, dtype=dtype)
    idxs = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return idxs  # torch.index_select(keys, 1, idxs)


def nn_lookup(queries, keys, *args, **kwargs):
    queries2d, size, ndim = view_as_2d(queries)
    keys2d, _, _ = view_as_2d(keys)
    output2d = nn_lookup2d(queries2d, keys2d, *args, **kwargs)
    output = view_2d_as(output2d, size, ndim)
    return output


def nn_lookup_soft(queries, keys, *args, **kwargs):
    queries2d, size, ndim = view_as_2d(queries)
    keys2d, _, _ = view_as_2d(keys)
    output2d = nn_lookup_soft2d(queries2d, keys2d, *args, **kwargs)
    output = view_2d_as(output2d, size, ndim)
    return output

