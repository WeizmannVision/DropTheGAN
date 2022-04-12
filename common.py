import numpy as np  #pylint: disable=unused-import

import torch  #pylint: disable=unused-import
import torch.nn as nn  #pylint: disable=unused-import
import torch.nn.functional as F  #pylint: disable=unused-import

__all__ = ['get_eps', 'to_numpy', 'allclose', 'nonzero', 'change_dtype', 'psnr', 'predict_topk', 'predict_topk_gt']


def get_eps(dtype):
    return torch.finfo(dtype).eps


def to_numpy(tensor, copy=True):
    tensor = tensor.detach()

    if copy:
        tensor = tensor.clone()

    return tensor.cpu().numpy()


def allclose(x, y):
    eps = get_eps(x.dtype)
    return torch.allclose(x, y, atol=eps)


def nonzero(x):
    eps = get_eps(x.dtype)
    return (x.abs() <= eps)


def predict_topk(model, x, k=1, keepdim=True):
    y = model(x)
    p, l = torch.topk(y, k)
    if k == 1 and not keepdim:
        l = l[..., 0]
        p = p[..., 0]    
    return l, p


def predict_topk_gt(model, x, y_gt, k=1, keepdim=True):
    y = model(x)
    p, l = torch.sort(y)

    r_gt = torch.nonzero(l == y_gt)
    p_gt = p[r_gt]
    p_gt = p_gt[..., 0]
    r_gt = r_gt[..., 0]

    if k == 1 and not keepdim:
        p = p[..., 0]
        l = l[..., 0]
    else:
        p = p[..., :k]
        l = l[..., :k]

    return l, p, r_gt, p_gt


def psnr(output, target):
    mse = F.mse_loss(output, target)
    return 10 * torch.log10(1 / (mse + get_eps(mse.dtype)))


def change_dtype(obj, dtype):
    if isinstance(obj, nn.Module):
        return _change_module_dtype(obj, dtype)
    elif isinstance(obj, torch.Tensor):
        return _change_tensor_dtype(obj, dtype)
    raise TypeError("unsupported type: %s" % (type(obj),))


_BATCH_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def _change_module_dtype(module, dtype):
    if dtype == torch.float16 and isinstance(module, _BATCH_NORM_TYPES):
        return _change_module_dtype(module, torch.float32)
    module = module.to(dtype=dtype)
    for submodule in module.children():
        _change_module_dtype(submodule, dtype)
    return module


def _change_tensor_dtype(tensor, dtype):
    return tensor.to(dtype=dtype)
