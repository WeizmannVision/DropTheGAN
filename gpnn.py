import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

import fold
import resize_right

_MAX_MEMORY_SIZE = 1 << 30
_INF = float('inf')


def gpnn(pyramid: Sequence[torch.Tensor],
         initial_guess: torch.Tensor,
         downscale_ratio: float = 0.75,
         patch_size: int = 7,
         alpha: float = _INF,
         output_pyramid_shape: Optional[Sequence[torch.Size]] = None,
         mask_pyramid: Optional[Sequence[torch.Tensor]] = None,
         num_iters_in_level: int = 10,
         num_iters_in_coarsest_level: int = 1,
         reduce: str = 'weighted_mean') -> torch.Tensor:
    if output_pyramid_shape is None:
        output_pyramid_shape = [image.shape for image in pyramid]
    if mask_pyramid is None:
        mask_pyramid = [None] * len(pyramid)
    generated = initial_guess
    coarsest_level = len(pyramid) - 1
    for level in range(coarsest_level, -1, -1):
        if level == coarsest_level:
            for i in range(num_iters_in_coarsest_level):
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
                                          pyramid[level].shape)
            for i in range(num_iters_in_level):
                generated = pnn(generated,
                                key=blurred,
                                value=pyramid[level],
                                mask=mask_pyramid[level],
                                patch_size=patch_size,
                                alpha=alpha,
                                reduce=reduce)
        if level > 0:
            generated = resize_right.resize(generated, 1 / downscale_ratio,
                                            output_pyramid_shape[level - 1])
    return generated


def pnn(query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_size: int = 7,
        alpha: float = _INF,
        reduce: str = 'weighted_mean') -> torch.Tensor:
    query = query.unsqueeze(0)
    key = key.unsqueeze(0)
    value = value.unsqueeze(0)
    if mask is not None:
        mask = mask.unsqueeze(0)
    query_patches = fold.unfold2d(query, patch_size)
    query_patches_column, query_patches_size = fold.view_as_column(
        query_patches)
    key_patches_column, _ = fold.view_as_column(fold.unfold2d(key, patch_size))
    value_patches_column, _ = fold.view_as_column(
        fold.unfold2d(value, patch_size))
    if mask is not None:
        mask = (mask > 0.5).to(query)
        mask_patches_column, _ = fold.view_as_column(
            fold.unfold2d(mask, patch_size))
        valid_patches_mask = mask_patches_column.sum(
            dim=2) > mask_patches_column.shape[2] - 0.5
        key_patches_column = key_patches_column.squeeze(0)[
            valid_patches_mask.squeeze(0)].unsqueeze(0)
        value_patches_column = value_patches_column.squeeze(0)[
            valid_patches_mask.squeeze(0)].unsqueeze(0)
    _, indices = find_normalized_nearest_neighbors(query_patches_column,
                                                   key_patches_column, alpha)
    out_patches_column = F.embedding(indices.squeeze(2),
                                     value_patches_column.squeeze(0))
    out_patches = fold.view_as_image(out_patches_column, query_patches_size)
    output = fold.fold2d(out_patches, reduce=reduce)
    return output.squeeze(0)


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
    # x2 = torch.einsum('bid,bid->bi',queries, queries).unsqueeze(2)
    # y2 = torch.einsum('bjd,bjd->bj', keys, keys).unsqueeze(1)
    # xy = torch.einsum('bid,bjd->bij', queries, keys)
    # return (x2 + y2 - 2 * xy) / queries.shape[-1]
    return torch.cdist(queries, keys, p=2).pow(2) / queries.shape[-1]


def _find_tile_height(height: int, width: int, cell_size: int,
                      max_tile_size: int) -> Tuple[int, int]:
    row_size = width * cell_size
    return min(height, (max_tile_size + row_size - 1) // row_size)


def _slice_weights(weights: Optional[torch.Tensor], start: int,
                   stop: int) -> Optional[torch.Tensor]:
    if weights is None:
        return None
    if weights.shape[1] == 1:
        return weights
    return weights[:, start:stop, :]


def _find_weighted_nearest_neighbors(
    queries: torch.Tensor,
    keys: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    dists = _compute_dist_matrix(queries, keys)
    if weights is not None:
        dists *= weights
    return dists.min(dim=2, keepdim=True)


def find_weighted_nearest_neighbors(
    queries: torch.Tensor,
    keys: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    max_memory_usage: int = _MAX_MEMORY_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    size_in_bytes = torch.finfo(queries.dtype).bits // 8
    batch_size = queries.shape[0]
    num_queries = queries.shape[1]
    num_keys = keys.shape[1]
    if weights is not None:
        max_memory_usage //= 2
    tile_height = _find_tile_height(num_queries, num_keys,
                                    batch_size * size_in_bytes,
                                    max_memory_usage)
    values = []
    indices = []
    for start in range(0, num_queries, tile_height):
        value, idx = _find_weighted_nearest_neighbors(
            queries[:, start:start + tile_height], keys,
            _slice_weights(weights, start, start + tile_height))
        values.append(value)
        indices.append(idx)
    return torch.cat(values, dim=1), torch.cat(indices, dim=1)

    # values = queries.new_zeros(size=(batch_size, num_queries, 1))
    # indices = torch.zeros(size=(batch_size, num_queries, 1), dtype=torch.int64, device=queries.device)
    # for start in range(0, num_queries, tile_height):
    #     value, idx = _find_weighted_nearest_neighbors(
    #         queries[:, start:start + tile_height], keys,
    #         _slice_weights(weights, start, start + tile_height))
    #     values[:, start:start + tile_height] = value
    #     indices[:, start:start + tile_height] = idx
    # return values, indices


def find_normalized_nearest_neighbors(
    queries: torch.Tensor,
    keys: torch.Tensor,
    alpha: float = _INF,
    max_memory_usage: int = _MAX_MEMORY_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha < _INF:
        # compute min distance where queries<-keys, and keys<-queries
        normalizer, _ = find_weighted_nearest_neighbors(
            keys, queries, None, max_memory_usage)
        normalizer += alpha
        normalizer = 1 / normalizer
        normalizer = normalizer.transpose(1, 2)  # "keys" <-> "queries"
    else:
        normalizer = None
    return find_weighted_nearest_neighbors(queries, keys, normalizer,
                                           max_memory_usage)
