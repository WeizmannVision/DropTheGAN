import torch


__all__ = ['view_6d_as_3d', 'view_6d_as_2d', 'view_3d_as_6d', 'view_6d_as_2d', 'view_2d_as_6d', 'view_2d_as_3d', 'view_as_2d', 'view_2d_as']


def view_6d_as_3d(input):
    if input.dim() != 6:
        raise ValueError('expects a 6D tensor as input')
    size = n, c, kh, kw, h, w = input.size()
    output = input.reshape(n, c * kh * kw, h * w)
    return output, size


def view_3d_as_6d(input, size):
    if input.dim() != 3:
        raise ValueError('expects a 3D tensor as input')
    if len(size) != 6:
        raise ValueError('expects a 6D tuple as size')
    output = input.view(size)
    return output


def view_3d_as_2d(input):
    if input.dim() != 3:
        raise ValueError('expects a 3D tensor as input')
    size = n, d, l = input.size()
    output = input.permute(0, 2, 1).reshape(n * l, d)
    return output, size


def view_2d_as_3d(input, size):
    if input.dim() != 2:
        raise ValueError('expects a 2D tensor as input')
    if len(size) != 3:
        raise ValueError('expects a 3D tuple as size')
    n, d, l = size
    output = input.reshape(n, l, d).permute(0, 2, 1).contiguous()
    return output


def view_6d_as_2d(input):
    if input.dim() != 6:
        raise ValueError('expects a 6D tensor as input')
    output, size6d = view_6d_as_3d(input)
    output, size3d = view_3d_as_2d(output)
    return output, [size6d, size3d]


def view_2d_as_6d(input, sizes):
    if input.dim() != 2:
        raise ValueError('expects a 2D tensor as input')
    if len(sizes) != 2:
        raise ValueError('expects a 2D list as sizes')
    size6d, size3d = sizes
    output = view_2d_as_3d(input, size3d)
    output = view_3d_as_6d(output, size6d)
    return output


def view_as_2d(input):
    ndim = input.dim()
    if ndim == 2:
        output, size = input, None
    elif ndim == 3:
        output, size = view_3d_as_2d(input)
    elif ndim == 6:
        output, size = view_6d_as_2d(input)
    else:
        raise ValueError(f'unsupported ndim: {ndim}')
    return output, size, ndim


def view_2d_as(input, size, ndim):
    if ndim == 2:
        output = input
    elif ndim == 3:
        output = view_2d_as_3d(input, size)
    elif ndim == 6:
        output = view_2d_as_6d(input, size)
    else:
        raise ValueError(f'unsupported ndim: {ndim}')
    return output
