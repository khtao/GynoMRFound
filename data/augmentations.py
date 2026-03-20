import numpy as np
import torch
from torch.nn.functional import interpolate
import random
from monai.transforms import (
    CenterSpatialCropd,
    RandSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,

    MapTransform,
    NormalizeIntensityd,
    ScaleIntensityd,
    RandFlipd,
    RandRotated,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandSimulateLowResolutiond,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    ResizeWithPadOrCropd,
)


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is for NCR
    label 2 is for ED
    label 4 is for ET
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                result = []
                result.append(np.logical_or(d[key] == 1, d[key] == 4))
                # merge labels 1, 2 and 4 to construct WT
                result.append(np.logical_or(np.logical_or(d[key] == 4, d[key] == 1), d[key] == 2))
                # label 4 is ET
                result.append(d[key] == 4)
                d[key] = np.stack(result, axis=0).astype(np.float32)
            else:
                if self.allow_missing_keys:
                    continue
        return d


class LoadTextFeature(MapTransform):
    def __init__(self, keys, path):
        super().__init__(keys)
        self.path = path if type(path) is list else [path]
        self.df = {}
        for pp in self.path:
            self.df.update(torch.load(pp))

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                my_key = d[key]
                feat = self.df[my_key]
                if len(feat) < 100:
                    feat = feat[0]
                    # n = random.randint(0, len(feat) - 1)
                    # feat = feat[n]
                feat = np.stack(feat, axis=0, dtype=np.float32)
                d[key] = feat
            else:
                if self.allow_missing_keys:
                    continue
        return d


def get_train_transforms(keys, need_load=False):
    transforms = [
        EnsureTyped(keys=keys),
        RandSpatialCropd(keys=keys, roi_size=[128, 128, 128]),
        ResizeWithPadOrCropd(keys=keys, spatial_size=[128, 128, 128]),

        RandSimulateLowResolutiond(keys=keys, prob=0.25),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys=keys, factors=0.1, prob=1.0),
        RandShiftIntensityd(keys=keys, offsets=0.1, prob=1.0),
    ]
    if need_load:
        transforms = [LoadImaged(keys=keys), ] + transforms

    return Compose(transforms).set_random_state(1999)


def get_train_transforms_v2(keys, need_load=False):
    transforms = [
        EnsureTyped(keys=keys),
        RandSpatialCropd(keys=keys, roi_size=[128, 128, 128]),
        ResizeWithPadOrCropd(keys=keys, spatial_size=[128, 128, 128]),
        # NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
    ]
    if need_load:
        transforms = [LoadImaged(keys=keys), ] + transforms

    return Compose(transforms).set_random_state(1999)


def get_val_transforms(keys, need_load=False):
    transforms = [
        EnsureTyped(keys=keys),
        CenterSpatialCropd(keys=keys, roi_size=[128, 128, 128]),
        ResizeWithPadOrCropd(keys=keys, spatial_size=[128, 128, 128]),
        NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        # ScaleIntensityd(keys=keys, channel_wise=True),
    ]
    if need_load:
        transforms = [LoadImaged(keys=keys), ] + transforms
    return Compose(transforms).set_random_state(1999)


class AugmentTensor:
    def __init__(self, trans: list[list]):
        self.trans = trans

    def __call__(self, data):
        for tt in self.trans:
            data = transforms_tensor(data, tt[0], tt[1])
        return data


def transforms_tensor(batch, name: str, p: float):
    name = name.lower()
    batch_new = []
    for image in batch:
        if 'flip' in name:
            kk = torch.rand(3) < p
            yy = []
            for i, k in enumerate(kk):
                if k:
                    yy.append(i + 1)
            image = torch.flip(image, dims=yy)
        elif 'intensity' in name:
            if torch.rand(1) < p:
                a = random.uniform(0.9, 1.1)
                image = image * a
            if torch.rand(1) < p:
                a = random.uniform(-0.1, 0.1)
                image = image + a
        elif 'nosie' in name:
            noise_std = 0.05
            noise_mean = 0
            if torch.rand(1) < p:
                gaussian_noise = torch.randn_like(image, device=image.device) * noise_std + noise_mean
                image = image + gaussian_noise
        elif 'resolution' in name:
            if torch.rand(1) < p:
                image = interpolate(image.unsqueeze(dim=0), scale_factor=0.25, mode="trilinear", align_corners=False)
                image = interpolate(image, scale_factor=4, mode="trilinear", align_corners=False).squeeze(dim=0)
        elif 'norm' in name:
            image = (image - image.mean()) / image.std()
        batch_new.append(image)

    return torch.stack(batch_new)


def get_train_seg_transforms():
    return Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            # ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(2.0, 2.0, 1.5),
                mode=("bilinear", "nearest"),
            ),
            RandRotated(keys=["image", "label"], range_x=0.5, range_y=0.5, range_z=0.5),
            CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128]),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=[128, 128, 128]),
            RandGaussianNoised(keys=["image"], prob=0.25),
            RandGaussianSmoothd(keys=["image"], prob=0.25),
            RandAdjustContrastd(keys=["image"], prob=0.25),
            RandSimulateLowResolutiond(keys=["image"], prob=0.25),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    ).set_random_state(1999)


def get_val_seg_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            # ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(2.0, 2.0, 1.5),
                mode=("bilinear", "nearest"),
            ),
            CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128]),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=[128, 128, 128]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    ).set_random_state(1999)


def get_train_pretrain_transforms_with_seg(keys):
    full = keys + ["seg"]
    mode = ["bilinear"] * len(keys) + ["nearest"]
    return Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=full),
            EnsureChannelFirstd(keys=keys),
            EnsureTyped(keys=full),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=full,
                pixdim=(2.0, 2.0, 1.0),
                mode=mode,
            ),
            CenterSpatialCropd(keys=full, roi_size=[128, 128, 128]),
            RandFlipd(keys=full, prob=0.5, spatial_axis=0),
            RandFlipd(keys=full, prob=0.5, spatial_axis=1),
            RandFlipd(keys=full, prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys=keys, factors=0.1, prob=1.0),
            RandShiftIntensityd(keys=keys, offsets=0.1, prob=1.0),
        ]
    ).set_random_state(1999)


def get_val_pretrain_transforms_with_seg(keys):
    full = keys + ["seg"]
    mode = ["bilinear"] * len(keys) + ["nearest"]
    return Compose(
        [
            LoadImaged(keys=full),
            EnsureChannelFirstd(keys=keys),
            EnsureTyped(keys=full),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=full,
                pixdim=(2.0, 2.0, 1.0),
                mode=mode,
            ),
            CenterSpatialCropd(keys=full, roi_size=[128, 128, 128]),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        ]
    ).set_random_state(1999)


if __name__ == '__main__':
    aug = AugmentTensor([['resolution', 0.25], ['flip', 0.25], ['intensity', 0.25], ])
    img = torch.rand(10, 1, 128, 128, 128)
    img = aug(img)
    print(img.shape)
