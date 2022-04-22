from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

__all__ = ['unfold2d', 'fold2d', 'view_as_column', 'view_as_image']

_Size6D = Tuple[int, int, int, int, int, int]


def unfold2d(image: torch.Tensor,
             kernel_size: Union[int, Tuple[int, int]],
             stride: Union[int, Tuple[int, int]] = 1,
             use_padding: bool = False) -> torch.Tensor:
    """Unfolds an image into patches.

    Args:
        image: A batch of images of shape (N, C ,H, W) to unfold.
        kernel_size: The size of each patch: (kH, kW).
        stride: The stride between patches: (sH, sW).
        use_padding: Whether to pad the image such that each pixels appears in
            exactly the same number of patches.

    Returns:
        An unfolded image of shape (N, C, kH, kW, H', W'). Not contiguous, but
        can be viewed as (N, C * kH * kW, H' * W').
    """
    if image.dim() != 4:
        raise ValueError('expects a 4D tensor as input')
    n, c, h, w = image.size()
    kh, kw = kernel_size = _pair(kernel_size)
    sh, sw = stride = _pair(stride)
    if use_padding:
        ph, pw = padding = (kh - sh, kw - sw)
    else:
        ph, pw = padding = (0, 0)
    oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
    output = F.unfold(image, kernel_size, stride=stride, padding=padding)
    output = output.view(n, c, kh, kw, oh, ow)
    return output


def fold2d(input, stride=1, use_padding=False, *, reduce='sum', std=1.7):
    # input dimensions (6D): n, c, kh, kw, h', w'
    # output dimensions (4D): n, c, h, w
    if input.dim() != 6:
        raise ValueError('expects a 6D tensor as input')
    n, c, kh, kw, h, w = input.shape
    sh, sw = stride = _pair(stride)
    if reduce == 'sum':
        output = _fold2d_sum(input, stride, use_padding)
    elif reduce == 'median':
        return _fold2d_median(input, stride, use_padding)
    elif reduce == 'mean':
        weights = _get_weights_fold2d_mean(input, kh, kw)
        output = _fold2d_sum(input, stride, use_padding)
        if use_padding:
            norm = weights[:, :, ::sh, ::sw, :, :].sum()
        else:
            weights = weights.expand(1, 1, kh, kw, h, w)
            norm = _fold2d_sum(weights, stride, use_padding)
            norm[norm == 0] = 1
        output = output / norm
    elif reduce == 'weighted_mean':
        weights = _get_weights_fold2d_weighted_mean(input, kh, kw, sh, sw, std)
        output = _fold2d_sum(input * weights, stride, use_padding)
        if use_padding and sh == 1 and sw == 1:
            norm = weights.sum()
        else:
            weights = weights.expand(1, 1, kh, kw, h, w)
            norm = _fold2d_sum(weights, stride, use_padding)
            norm[norm == 0] = 1
        output = output / norm
    else:
        raise ValueError(f'unknown reduction: {reduce}')
    return output


def _fold2d_sum(input, stride, use_padding):
    if input.dim() != 6:
        raise ValueError('expects a 6D tensor as input')
    n, c, kh, kw, h, w = input.shape
    sh, sw = stride = _pair(stride)
    if use_padding:
        ph, pw = padding = (kh - sh, kw - sh)
    else:
        ph, pw = padding = (0, 0)
    oh, ow = output_size = (sh * (h - 1) + kh - 2 * ph,
                            sw * (w - 1) + kw - 2 * pw)
    kernel_size = (kh, kw)
    input = input.reshape(n, c * kh * kw, h * w)
    output = F.fold(input,
                    output_size,
                    kernel_size,
                    stride=stride,
                    padding=padding)
    return output


def _fold2d_median(input, stride, use_padding):
    if input.dim() != 6:
        raise ValueError('expects a 6D tensor as input')
    n, c, kh, kw, h, w = input.shape
    sh, sw = stride = _pair(stride)
    dh, dw = kh // sh, kw // sw
    if kh % sh != 0:
        raise ValueError('kh should be divisible by sh')
    if kw % sw != 0:
        raise ValueError('kw should be divisible by sw')

    if use_padding:
        ph, pw = (kh - sh, kw - sw)
        oh, ow = (sh * (h - 1) + kh - 2 * ph, sw * (w - 1) + kw - 2 * pw)
        output = input.new_zeros(size=(dh * dw, n, c, oh, ow))
        for i in range(kh):
            for j in range(kw):
                ii, jj = i // sh, j // sw
                sii, sjj = (kh - i - 1) // sh, (kw - j - 1) // sw
                output[ii * dw + jj, :, :, i % sh::sh, j % sw::sw] = input[:,
                                                                           :, i, j, sii:h - ii, sjj:w - jj]  # yapf: disable
        output = torch.median(output, dim=0)[0]

    else:
        if not hasattr(torch, 'nanmedian'):
            raise RuntimeError(
                'fold2d_median with use_padding==False depends on torch.nanmedian()'
            )
        if sh != 1 or sw != 1:
            raise NotImplementedError(
                'fold2d_median with use_padding==False and stride!=1 is not implemented'
            )
        oh, ow = (sh * (h - 1) + kh, sw * (w - 1) + kw)
        output = input.new_full(size=(dh * dw, n, c, oh, ow),
                                fill_value=float('nan'))
        for i in range(kh):
            for j in range(kw):
                ii, jj = i // sh, j // sw
                output[ii * dw + jj, :, :, i:h + i:sh, j:w +
                       j:sw] = input[:, :, i, j, :, :]  # yapf: disable
        output = torch.nanmedian(output, dim=0)[0]

    return output


def view_as_column(
        unfloded_image: torch.Tensor) -> Tuple[torch.Tensor, _Size6D]:
    unfloded_image_size = n, c, kh, kw, h, w = unfloded_image.size()
    unfolded_column = unfloded_image.view(n, c * kh * kw, h * w)
    unfolded_column = unfolded_column.permute(0, 2, 1)
    return unfolded_column, unfloded_image_size


def view_as_image(unfolded_column: torch.Tensor,
                  unfloded_image_size: _Size6D) -> torch.Tensor:
    return unfolded_column.permute(0, 2, 1).view(unfloded_image_size)


def _get_weights_fold2d_mean(input, kh, kw):
    weights = input.new_ones(size=(kh, kw))
    return weights.view(1, 1, kh, kw, 1, 1)


def _get_weights_fold2d_weighted_mean(input, kh, kw, sh, sw, std):
    to = {'device': input.device, 'dtype': input.dtype}
    gh = sh * torch.linspace(-1, 1, kh, **to)
    gw = sw * torch.linspace(-1, 1, kw, **to)
    nh = torch.exp(-0.5 * (gh / std).pow(2))
    nw = torch.exp(-0.5 * (gw / std).pow(2))
    weights = torch.einsum('i,j->ij', nh, nw)
    return weights.view(1, 1, kh, kw, 1, 1)
