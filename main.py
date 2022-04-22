from typing import Optional, Tuple

import fire
import torch

import applications
import utils

_INF = float('inf')


def _get_device(requested_device: Optional[str] = None) -> torch.device:
    if requested_device is not None:
        return torch.device(requested_device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generation(input_path: str,
               output_path: str,
               noise_std: float = 0.75,
               alpha: float = _INF,
               patch_size: int = 7,
               downsacle_ratio: float = 0.75,
               num_levels: int = 9,
               device: Optional[str] = None) -> None:
    """Generates diverse images based on a single image.
    
    Args:
        input_path: Path to the input image.
        output_path: Path to the output image.
        noise_std: The standard deviation of the noise to the corasest level.
            Increasing it will diversify the outputs at the cost of fidelity.
        alpha: The alpha parameter, used to set the compeletness level.
            There is a tradeoff between completeness and coherence.
            Decreasing alpha would encourage completeness.
        patch_size: The size of the patch to use.
        downscale_ratio: Downsampling scale between consecuitive pyramid levels.
            Should be less than 1.
        num_levels: Number of pyramid levels to use.
        device: The device to use.
    """
    device = _get_device(device)
    input_image = utils.imread(input_path).to(device=device)
    output_image = applications.generation(input_image,
                                           noise_std=noise_std,
                                           alpha=alpha,
                                           patch_size=patch_size,
                                           num_levels=num_levels,
                                           downscale_ratio=downsacle_ratio)
    utils.imwrite(output_path, output_image)


def retargeting(input_path: str,
                output_path: str,
                retargeting_ratio: Tuple[float, float],
                alpha: float = 5e-3,
                patch_size: int = 7,
                downsacle_ratio: float = 0.8,
                num_levels: int = 8,
                device: Optional[str] = None) -> None:
    """Retargt an image to different sizes.
    Args:
        input_path: Path to the input image.
        output_path: Path to the output image.
        retargeting_ratio: The retargeting ratios along the height, width
            dimensions, respectively.
        alpha: The alpha parameter, used to set the compeletness level.
            There is a tradeoff between completeness and coherence.
            Decreasing alpha would encourage completeness.
        patch_size: The size of the patch to use.
        downscale_ratio: Downsampling scale between consecuitive pyramid levels.
            Should be less than 
        num_levels: Number of pyramid levels to use.
        device: The device to use.
    """
    device = _get_device(device)
    input_image = utils.imread(input_path).to(device=device)
    output_image = applications.retargeting(
        input_image,
        retargeting_ratio=retargeting_ratio,
        alpha=alpha,
        patch_size=patch_size,
        num_levels=num_levels,
        downscale_ratio=downsacle_ratio)
    utils.imwrite(output_path, output_image)


def editing(source_image_path: str,
            edited_image_path: str,
            output_path: str,
            alpha: float = _INF,
            patch_size: int = 7,
            downsacle_ratio: float = 0.75,
            num_levels: int = 5,
            device: Optional[str] = None) -> None:
    """ Edit an image, and semalessly blend the edit.

    Args:
        source_image_path: Path to the source input image.
        edited_image_path: Path to the edited input image.
        output_path: Path to the output image.
        alpha: The alpha parameter, used to set the compeletness level.
            There is a tradeoff between completeness and coherence.
            Decreasing alpha would encourage completeness.
        patch_size: The size of the patch to use.
        downscale_ratio: Downsampling scale between consecuitive pyramid levels.
            Should be less than 1.
        num_levels: Number of pyramid levels to use.
        device: The device to use.
    """
    device = _get_device(device)
    source_image = utils.imread(source_image_path).to(device=device)
    edited_image = utils.imread(edited_image_path).to(device=device)
    output_image = applications.editing(source_image,
                                        edited_image,
                                        alpha=alpha,
                                        patch_size=patch_size,
                                        num_levels=num_levels,
                                        downscale_ratio=downsacle_ratio)
    utils.imwrite(output_path, output_image)


def conditional_inpainting(masked_image_path: str,
                           mask_path: str,
                           output_path: str,
                           alpha: float = _INF,
                           patch_size: int = 7,
                           downscale_ratio: float = 0.75,
                           num_levels: int = 5,
                           device: Optional[str] = None) -> None:
    """ Given an image with an occluded region, 
        seamlessly fill the region with respect to a chosen color.
    Args:
        masked_image_path: Path to the conditionally masked image. Occluded
            regions should be filled with uniform color chosen by the user.
        mask_path: Path to the binary mask image. White pixels in the image (1.0) are
            considered valid (i.e., used to build the output image).
        output_path: Path to the output image.
        alpha: The alpha parameter, used to set the compeletness level.
            There is a tradeoff between completeness and coherence.
            Decreasing alpha would encourage completeness.
        patch_size: The size of the patch to use.
        num_levels: Number of pyramid levels to use.
        device: The device to use.
    """
    device = _get_device(device)
    masked_image = utils.imread(masked_image_path).to(device=device)
    mask = utils.mask_read(mask_path).to(device=device)
    output_image = applications.editing(masked_image,
                                        mask,
                                        alpha=alpha,
                                        patch_size=patch_size,
                                        num_levels=num_levels,
                                        downscale_ratio=downscale_ratio)
    utils.imwrite(output_path, output_image)


def structural_analogy(source_image_path: str,
                       structure_image_path: str,
                       output_path: str,
                       alpha: float = 5e-3,
                       patch_size: int = 7,
                       downsacle_ratio: float = 0.75,
                       num_levels: int = 5,
                       device: Optional[str] = None) -> None:
    """ Create an image that is aligned to the structure, 
        but shares patch distribution with the source.
    Args:
        source_image_path: Path to the source input image.
        structure_image_path: Path to the structure input image.
        output_path: Path to the output image.
        alpha: The alpha parameter, used to set the compeletness level.
            There is a tradeoff between completeness and coherence.
            Decreasing alpha would encourage completeness.
        patch_size: The size of the patch to use.
        downscale_ratio: Downsampling scale between consecuitive pyramid levels.
            Should be less than 1.
        num_levels: Number of pyramid levels to use.
        device: The device to use.
    """
    source_image = utils.imread(source_image_path).to(device=device)
    structure_image = utils.imread(structure_image_path).to(device=device)
    output_image = applications.structural_analogy(
        source_image,
        structure_image,
        alpha=alpha,
        patch_size=patch_size,
        num_levels=num_levels,
        downscale_ratio=downsacle_ratio)
    utils.imwrite(output_path, output_image)


if __name__ == '__main__':
    fire.Fire()
