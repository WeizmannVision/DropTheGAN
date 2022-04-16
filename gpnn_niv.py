def gpnn(pyr: Sequence[torch.Tensor], dst_pyr: Sequence[torch.Tensor], 
        query: torch.Tensor, noise_std: float = 0.75, alpha: float = 10.0,
        downscale_ratio: float = 0.75, patch_size: int = 7, reduce: str = 'weighted_mean',
        num_iters_top_level: int = 1, mask_pyr: torch.Tensor = None):
  start = time.time()
  query += torch.randn_like(query)*noise_std
  for l in range(len(pyr)-1,-1,-1): 
    num_iters = num_iters_top_level if l == top_level else 10
    for k in range(num_iters):
        if l == top_level:
            new_im = pnn(query, key=pyr[l] ,value=pyr[l], mask=mask_pyr[l], patch_size=patch_size, reduce=reduce)
        else:
            blurred = resize_right.resize(pyr[l+1], 1/downscale_ratio, dst_pyr[l].shape)
            query = pnn(query, key=blurred, value=pyr[l], mask=mask_pyr[l], patch_size=patch_size, reduce=reduce)
    if l > 0:
        query = resize_right.resize(pyr[l], 1/downscale_ratio, dst_pyr[l-1].shape)
  print('Total time: %.3f[s]' % (time.time() - start,))
  return query

def image_generation(input_path: str, noise_std: float = 0.75, alpha: float = 10.0,
                    patch_size: int = 7, reduce: str = 'weighted_mean', 
                    downscale_ratio: float = 0.75, num_levels: int = 9):
    pyr = make_pyramid(imread(input_path), num_levels, downscale_ratio)
    return gpnn(pyr, pyr, query=pyr[-1], noise_std=noise_std, alpha=alpha, 
        downscale_ratio=downscale_ratio, patch_size=patch_size, reduce=reduce)

def image_editing(input_path: str, input_path_edited: str, noise_std: float = 0.75, 
                    alpha: float = 10.0, device: str = 'cuda', 
                    patch_size: int = 7, reduce: str = 'weighted_mean', 
                    downscale_ratio: float = 0.75, num_levels: int = 9):
    pyr = make_pyramid(imread(input_path), num_levels, downscale_ratio)
    pyr_edited = make_pyramid(imread(input_path_edited), num_levels, downscale_ratio)
    return gpnn(pyr, pyr, query=pyr_edited[-1], noise_std=noise_std, alpha=alpha, 
        downscale_ratio=downscale_ratio, device=device, patch_size=patch_size,
        reduce=reduce)

def retargeting(input_path: str, noise_std: float = 0.0, 
                alpha: float = 1e-3, device: str = 'cuda', 
                patch_size: int = 7, reduce: str = 'weighted_mean', 
                downscale_ratio: float = 0.8, num_levels: int = 9,
                retargeting_ratio: tuple = (0.75, 0.75), gradual: bool = False,
                min_axis_size_coarsest: int = 21):
    pyr = make_pyramid(imread(input_path), num_levels, downscale_ratio)
    if gradual:
        step_size = [0.9, 0.9]
        if retargeting_ratio[0] >= 1:
            step_size[0] = 1.1
        if retargeting_ratio[1] >= 1:
            step_size[0] = 1.1
        num_steps = math.floor(max(math.log(retargeting_ratio[0])/math.log(step_size[0]), 
                                math.log(retargeting_ratio[1])/math.log(step_size[1])))
        step_size[0] = 10**(math.log10(ratio[0])/num_steps)
        step_size[1] = 10**(math.log10(ratio[1])/num_steps)
    else:
        num_steps = 1
        step_size = retargeting_ratio
    query = pyr[0].clone()
    for _ in num_steps:
        resized = resize_right.resize(query, step_size)
        top_level = math.floor(min(math.log(min_axis_size_coarsest/new_im.shape[-2])/math.log(downscale_ratio), 
                    math.log(min_axis_size_coarsest/new_im.shape[-1])/math.log(downscale_ratio)))
        retargeted_pyr = make_pyramid(resized, top_level, downscale_ratio)
        query = new_pyr[top_level].clone()
        query = gpnn(pyr, retargeted_pyr, query=query, noise_std=noise_std, alpha=alpha, 
            downscale_ratio=downscale_ratio, device=device, patch_size=patch_size,
            reduce=reduce)
    return query

def structural_analogy(input_path_source, input_path_structure, patch_size=7, noise_std: float = 0.0, 
                alpha: float = 1e-3, device: str = 'cuda', 
                patch_size: int = 7, reduce: str = 'weighted_mean', 
                downscale_ratio: float = 0.8, num_levels: int = 9,
                num_iters_top_level: int = 10):
    start = time.time()
    pyr_source = make_pyramid(imread(input_path_source), num_levels, downscale_ratio)
    pyr_structure = make_pyramid(imread(input_path_structure), num_levels, downscale_ratio)
    query = pyr_structure[-1].clone()
    return gpnn(pyr_source, pyr_structure, query=query, noise_std=noise_std, alpha=alpha, 
            downscale_ratio=downscale_ratio, device=device, patch_size=patch_size,
            reduce=reduce, num_iters_top_level=num_iters_top_level)

def conditional_inpainting(input_path, mask_path, patch_size=7, noise_std: float = 0.0, 
                alpha: float = 10.0, device: str = 'cuda', 
                patch_size: int = 7, reduce: str = 'weighted_mean', 
                downscale_ratio: float = 0.75, num_levels: int = 4,
                num_iters_top_level: int = 10):
    pyr = make_pyramid(imread(input_path), num_levels, downscale_ratio)
    pyr_mask = make_pyramid(imread(mask_path), num_levels, downscale_ratio)
    query = pyr[top_layer]

    return gpnn(pyr, pyr, query=query, mask=pyr_mask, noise_std=noise_std, alpha=alpha, 
            downscale_ratio=downscale_ratio, device=device, patch_size=patch_size,
            reduce=reduce, num_iters_top_level=num_iters_top_level)


def _calc_dist_l2(X, Y):
    Y = Y.transpose(0, 1)
    X2 = X.pow(2).sum(1, keepdim=True)
    Y2 = Y.pow(2).sum(0, keepdim=True)
    XY = X @ Y
    return X2 - (2 * XY) + Y2
