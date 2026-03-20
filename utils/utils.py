from typing import Union

import einops
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def multi_label(y_label, y_pred, func):
    class_num = y_label.shape[1]
    out = []
    for i in range(class_num):
        out.append(func(y_label[:, i], y_pred[:, i]))
    return out


def create_logits(x1, x2, logit_scale=1):
    # logit_scale=1
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()
    return logits_per_x1, logits_per_x2


def patchify(
        images: Union[np.ndarray, torch.Tensor],
        patch_size: tuple[int, int, int] = (16, 16, 16), img_size: tuple[int, int, int] = (128, 128, 128)
) -> Union[np.ndarray, torch.Tensor]:
    """
    Takes a batch of 2D or 3D images and outputs the patchify version of it.

    Args:
        images: A tensor of shape (N, C, H, W) or (N, C, D, H, W) for 3D images.
            N is the batch size, C is the number of channels, H, W, D are the height, width, and depth.
        patch_size: An integer indicating the size of the patches to be extracted.
                If an integer is provided, square/cubic patches are assumed for 2D/3D images.
        img_size: An integer indicating the size of the images. This is used to determine the number of patches to be extracted.

    Returns:
        A tensor of shape (N, L, D) where L is the number of patches and D is the flattened dimension of each patch.
    """
    assert images.dim() in (4, 5), "images must be either 4D or 5D tensors"

    # n_patches_per_axis = img_size // patch_size
    unfolded = einops.rearrange(
        images,
        "b c (gh ph) (gw pw) (gd pd) -> b (gh gw gd) (ph pw pd c)",
        gh=img_size[0] // patch_size[0],
        gw=img_size[1] // patch_size[1],
        gd=img_size[2] // patch_size[2],
        ph=patch_size[0],
        pw=patch_size[1],
        pd=patch_size[2],
    )

    return unfolded


def unpatchify(
        images: Union[np.ndarray, torch.Tensor],
        patch_size: tuple[int, int, int] = (16, 16, 16),
        img_size: tuple[int, int, int] = (128, 128, 128)
) -> Union[np.ndarray, torch.Tensor]:
    """
    Takes a batch of patchified images and outputs the original images.

    Args:
        images: A tensor of shape (N, L, D) where L is the number of patches and D is the flattened dimension of each patch.
        patch_size: An integer indicating the size of the patches to be extracted.
                If an integer is provided, square/cubic patches are assumed for 2D/3D images.
        img_size: An integer indicating the size of the images. This is used to determine the number of patches to be extracted.

    Returns:
        A tensor of shape (N, C, H, W) or (N, C, D, H, W) for 3D images.
        N is the batch size, C is the number of channels, H, W, D are the height, width, and depth.
    """
    assert images.dim() == 3, "images must be 3D tensors"
    images = einops.rearrange(
        images,
        "b (gh gw gd) (ph pw pd c) -> b c (gh ph) (gw pw) (gd pd)",
        gh=img_size[0] // patch_size[0],
        gw=img_size[1] // patch_size[1],
        gd=img_size[2] // patch_size[2],
        ph=patch_size[0],
        pw=patch_size[1],
        pd=patch_size[2],
    )
    return images

def norm_image(image):
    return (image - image.min()) / (image.max() - image.min())


def visualize_3d(reconstructions, inputs, mask):
    reconstructions = unpatchify(reconstructions)
    reconstructions = torch.einsum("nchwd->nhwdc", reconstructions).detach().cpu()[0]
    mask = mask.detach().unsqueeze(-1).repeat(1, 1, 16 * 16 * 16)
    mask = unpatchify(mask)
    mask = torch.einsum("nchwd->nhwdc", mask).detach().cpu()[0]
    inputs = torch.einsum("nchwd->nhwdc", inputs).detach().cpu()[0]

    im_masked = inputs * (1 - mask)
    im_paste = inputs * (1 - mask) + reconstructions * mask

    fig = plt.figure(figsize=(8, 2))
    plt.subplot(1, 4, 1)
    plt.imshow(norm_image(inputs[:, :, 64, -1].cpu().numpy()))
    plt.axis("off")
    plt.title(f"Original")

    plt.subplot(1, 4, 2)
    plt.imshow(norm_image(im_masked[:, :, 64, -1].cpu().numpy()))
    plt.axis("off")
    plt.title(f"Masked")

    plt.subplot(1, 4, 3)
    plt.imshow(norm_image(reconstructions[:, :, 64, -1].cpu().numpy()))
    plt.axis("off")
    plt.title(f"Recons.")

    plt.subplot(1, 4, 4)
    plt.imshow(norm_image(im_paste[:, :, 64, -1].cpu().numpy()))
    plt.axis("off")
    plt.title(f"Recons. + Original")

    return fig


visualize_multimodal_3d = visualize_3d
