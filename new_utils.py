import math
import os
from typing import List, Tuple

import PIL
import torch

import resize_right

_MAX_MEMORY_SIZE = 2 << 30


def to_numpy(tensor, clone=True):
    tensor = tensor.detach()
    tensor = tensor.clone() if clone else tensor
    return tensor.cpu().numpy()


def imread(path: str) -> torch.Tensor:
    image = torch.Tensor(PIL.Image.open(path), dtype=torch.float32)
    image = image.permute(2, 0, 1)
    return image / 255


def imwrite(path: str, image: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = image.permute(1, 2, 0)
    image = to_numpy((255 * image).clamp(0, 255).to(dtype=torch.uint8))
    return PIL.Image.fromarray(image).save(path)


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


def _compute_dist_matrix(queries: torch.Tensor,
                         keys: torch.Tensor) -> torch.Tensor:
    """Computes a matrix of MSE between each query and each key."""
    return torch.cdist(queries, keys, p=2).pow(2) / queries.shape[-1]


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
    size_in_bytes = torch.finfo(queries.dtype) // 8
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
