import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

import fold
import image as image_utils
import resize_right

_MAX_MEMORY_SIZE = 1 << 30


def gpnn(pyramid: List[torch.Tensor],
         initial_guess: torch.Tensor,
         downscale_ratio: float = 0.75,
         patch_size: int = 7,
         alpha: float = 10.0,
         output_pyramid_shape: Optional[List[torch.Size]] = None,
         mask_pyramid: Optional[List[torch.Tensor]] = None,
         num_iters_in_level: int = 10,
         num_iters_in_coarsest_level: int = 1,
         reduce: str = 'weighted_mean') -> torch.Tensor:
    if output_pyramid_shape is None:
        output_pyramid_shape = [image.shape for image in pyramid]
    if mask_pyramid is None:
        mask_pyramid = [None] * len(pyramid)
    generated = initial_guess
    coarsest_level = len(pyramid) - 1
    print(initial_guess.shape)
    image_utils.imshow(initial_guess)
    for level in range(coarsest_level, -1, -1):
        print(f'level: {level}')
        if level == coarsest_level:
            for i in range(num_iters_in_coarsest_level):
                print(f'corasest iteration: {i}')
                generated = pnn(generated,
                                key=pyramid[level],
                                value=pyramid[level],
                                mask=mask_pyramid[level],
                                patch_size=patch_size,
                                alpha=alpha,
                                reduce=reduce)
        else:
            blurred = resize_right.resize(pyramid[level + 1],
                                          1 / downscale_ratio,
                                          output_pyramid_shape[level])
            for i in range(num_iters_in_level):
                print(f'iteration: {i}')
                generated = pnn(generated,
                                key=blurred,
                                value=pyramid[level],
                                mask=mask_pyramid[level],
                                patch_size=patch_size,
                                alpha=alpha,
                                reduce=reduce)
        print(generated.shape)
        image_utils.imshow(generated)
        if level > 0:
            generated = resize_right.resize(generated, 1 / downscale_ratio,
                                            output_pyramid_shape[level - 1])
    return generated


def pnn(query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_size: int = 7,
        alpha: float = 10.0,
        reduce: str = 'weighted_mean') -> torch.Tensor:
    query_patches = fold.unfold2d(query, patch_size)
    query_patches_column, query_patches_size = fold.view_as_column(
        query_patches)
    key_patches_column, _ = fold.view_as_column(fold.unfold2d(key, patch_size))
    value_patches_column, _ = fold.view_as_column(
        fold.unfold2d(value, patch_size))
    if mask is not None:
        mask_patches_column, _ = fold.view_as_column(
            fold.unfold2d(mask, patch_size))
        unmasked_patches = mask_patches_column.sum(
            dim=2) > mask_patches_column.shape[2] - 0.5
        key_patches_column = key_patches_column[:, :, unmasked_patches]
        value_patches_column = value_patches_column[:, :, unmasked_patches]
    dist = compute_dist_matrix(query_patches_column, key_patches_column)
    dist /= dist.min(dim=0, keepdim=True)[0] + alpha
    indices = dist.argmin(dim=2)
    out_patches_column = F.embedding(indices, value_patches_column.squeeze(0))
    out_patches = fold.view_as_image(out_patches_column, query_patches_size)
    return fold.fold2d(out_patches, reduce=reduce)


def make_pyramid(image: torch.Tensor, num_levels: int,
                 downscale_ratio: float) -> List[torch.Tensor]:
    scale_factor = (1, ) * (image.ndim - 2) + (downscale_ratio,
                                               downscale_ratio)
    pyramid = [image]
    for level in range(1, num_levels + 1):
        output_shape = (*image.shape[:-2],
                        math.ceil(image.shape[-2] * downscale_ratio**level),
                        math.ceil(image.shape[-1] * downscale_ratio**level))
        pyramid.append(
            resize_right.resize(pyramid[-1], scale_factor, output_shape))
    return pyramid


def _find_tile_size(height: int, width: int, cell_size: int,
                    max_tile_size: int) -> Tuple[int, int]:
    best_tile_height = 1
    best_tile_width = 1
    best_tile_size = cell_size
    for tile_height in range(2, height + 1):
        if tile_height * cell_size > max_tile_size:
            break
        tile_width = min(width, max_tile_size // (tile_height * cell_size))
        tile_size = tile_height * tile_width * cell_size
        if tile_size == max_tile_size:
            return tile_height, tile_width
        elif tile_size < max_tile_size and tile_size > best_tile_size:
            best_tile_height, best_tile_width = tile_height, tile_width
            best_tile_size = tile_size
    return best_tile_height, best_tile_width


# def _compute_dist_matrix(queries: torch.Tensor,
#                          keys: torch.Tensor) -> torch.Tensor:
#     """Computes a matrix of MSE between each query and each key."""
#     return torch.cdist(queries, keys, p=2).pow(2) / queries.shape[-1]


def _compute_dist_matrix(queries: torch.Tensor,
                         keys: torch.Tensor) -> torch.Tensor:
    """Computes a matrix of MSE between each query and each key."""
    x2 = torch.einsum('bid,bid->bi',queries, queries).unsqueeze(2)
    y2 = torch.einsum('bjd,bjd->bj', keys, keys).unsqueeze(1)
    xy = torch.einsum('bid,bjd->bij', queries, keys)
    return (x2 + y2 - 2 * xy) / queries.shape[-1]


def compute_dist_matrix(
        queries: torch.Tensor,
        keys: torch.Tensor,
        max_memory_usage: int = _MAX_MEMORY_SIZE) -> torch.Tensor:
    """Computes distance matrix between queries and keys.
    
    The distance is MSE between patches.

    Args:
        queries: A tensor of queries of shape (B, Q, D).
        keys: A tensor of keys shape (B, K, D).
    
    Returns:
        A distance matrix shape (B, Q, K) of L2 distances between queries and keys.
    """
    if queries.shape[0] != keys.shape[0]:
        raise ValueError('qeuries and keys must have the same batch size.')
    if queries.shape[2] != keys.shape[2]:
        raise ValueError('qeuries and keys must have the same patch size.')
    size_in_bytes = torch.finfo(queries.dtype).bits // 8
    batch_size = queries.shape[0]
    num_queries = queries.shape[1]
    num_keys = keys.shape[1]
    tile_height, tile_width = _find_tile_size(num_queries, num_keys,
                                              size_in_bytes, max_memory_usage)
    num_rows = (num_queries + tile_height - 1) // tile_height
    num_cols = (num_keys + tile_width - 1) // tile_width
    dist = queries.new_empty(size=(batch_size, num_queries, num_keys))
    for row in range(num_rows):
        for col in range(num_cols):
            row_start, row_stop = row * tile_height, (row + 1) * tile_height
            col_start, col_stop = col * tile_width, (col + 1) * tile_width
            queries_tile = queries[:, row_start:row_stop]
            keys_tile = keys[:, col_start:col_stop]
            dist_tile = _compute_dist_matrix(queries_tile, keys_tile)
            dist[:, row_start:row_stop, col_start:col_stop] = dist_tile
    return dist
