import fire
import torch

import applications
import new_utils as utils

_INF = float('inf')


def generation(input_path: str,
               output_path: str,
               alpha: float = _INF,
               patch_size: int = 7,
               noise_std: float = 0.75,
               pyramid_depth: int = 9,
               pyramid_downsacle_ratio: float = 0.75,
               device: str = 'cuda') -> None:
    """Generates diverse images based on a single image.
    
    Args:
        input_path: Path to the input image.
        output_path: Path to the output image.
        alpha: The alpha parameter, used to set the compeletness level.
            There is a tradeoff between completeness and coherence.
            Decreasing alpha would encourage completeness.
        noise_std: The standard deviation of the noise to the corasest level.
            Increasing it will diversify the outputs at the cost of fidelity.
        pyramid_levels: Number of pyramid levels to use.
        pyramid_scale: Downsampling scale between consecuitive pyramid levels.
            Should be less than 1.
        device: The device to use.
    """
    device = torch.device(device)
    image = utils.imread(input_path).to(device=device)
    return applications.image_generation(image,
                                         noise_std=noise_std,
                                         alpha=alpha,
                                         patch_size=patch_size,
                                         pyramid_depth=pyramid_depth,
                                         pyramid_downscale_ratio=pyramid_downsacle_ratio)


if __name__ == '__main__':
    fire.Fire()
