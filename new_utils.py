import os

import numpy as np
import PIL
import torch


def to_numpy(tensor: torch.Tensor, clone=True) -> np.ndarray:
    tensor = tensor.detach()
    tensor = tensor.clone() if clone else tensor
    return tensor.cpu().numpy()


def imread(path: str) -> torch.Tensor:
    image = PIL.Image.open(path).convert(mode='RGB')
    image = torch.tensor(np.asarray(image), dtype=torch.float32)
    image = image.permute(2, 0, 1)
    return image / 255


def imwrite(path: str, image: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = image.permute(1, 2, 0)
    image = to_numpy((255 * image).clamp(0, 255).to(dtype=torch.uint8))
    return PIL.Image.fromarray(image).save(path)
