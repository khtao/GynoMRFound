import json
import os

from torch.utils.hipify.hipify_python import meta_data

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import numpy as np
import pandas as pd
from data.mri_dataset import list_file_tree
from tqdm import tqdm
from model.model_GynoMR import vit_base
from pretrained_models.Swin_UNETR.model import SwinUNETR
from pretrained_models.PRISM.model import PRISM
from pretrained_models.UNETR.model import UNETR
from pretrained_models.BrainSegFounder.model import BrainSegFounder
from pretrained_models.ResNet50.model import ResNet50
from pretrained_models.RP3D_Diag.model import RadNet
from scipy.ndimage.interpolation import zoom
from torch.utils.data import dataset
from monai.transforms import EnsureChannelFirstd, EnsureTyped, Spacingd, LoadImaged, CenterSpatialCropd, \
    ResizeWithPadOrCropd, NormalizeIntensityd, Compose
from torch.utils.data import DataLoader


def zoom_3d_image(image):
    return zoom(image, zoom=(1, 2, 2, 2), mode='nearest', order=3)


class SafeDataset(dataset.Dataset):
    def __init__(self, meta_data, transform):
        self.meta_data = meta_data
        self.transform = transform

    def __getitem__(self, index):
        meta = self.meta_data[index]
        file_path = meta['image']
        try:
            image = self.transform(meta)
            return image
        except:
            print('load error', file_path)
            return meta

    def __len__(self):
        return len(self.meta_data)


class CheckImage:
    def __init__(self, keys):
        if type(keys) == str:
            keys = [keys]
        self.keys = keys

    def __call__(self, d):
        for k in self.keys:
            if k in d.keys():
                img = d[k][0]
                img = img.numpy().astype(np.uint16)
                if img.max() == img.min() or img is None:
                    d['check'] = False
                elif len(img.shape) != 3:
                    d['check'] = False
                else:
                    d['check'] = True
        return d


def get_val_transforms(keys):
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        EnsureTyped(keys=keys),
        Spacingd(keys=keys, pixdim=[2.0, 2.0, 1.5]),
        CheckImage(keys=keys),
        CenterSpatialCropd(keys=keys, roi_size=[128, 128, 128]),
        ResizeWithPadOrCropd(keys=keys, spatial_size=[128, 128, 128]),
        NormalizeIntensityd(keys=keys),
    ]
    return Compose(transforms).set_random_state(3407)


def make_dataset(xlsx_path, dir_root, label_keys='label', image_keys='path'):
    meta_data = []
    if xlsx_path[-4:] == 'json':
        data = json.load(open(xlsx_path))
        for key, value in data.items():
            if type(value) is dict:
                files = value[image_keys]
                label = value[label_keys]
                for pp in files:
                    meta_data.append({'image': pp, 'label': label, 'path': pp})
    elif xlsx_path[-3:] == 'txt':
        data = open(xlsx_path).read().splitlines()
        for line in data:
            line_split = line.split(',')
            path = line_split[0]
            label = int(line_split[1])
            files = list_file_tree(os.path.join(dir_root, path), file_type="nii.gz")
            for pp in files:
                meta_data.append({'image': pp, 'label': label, 'path': pp})
    else:
        data = pd.read_excel(xlsx_path)
        for i in range(len(data[image_keys])):
            label = int(data[label_keys][i])
            path = data[image_keys][i]
            files = list_file_tree(os.path.join(dir_root, path), file_type="nii.gz")
            for pp in files:
                meta_data.append({'image': pp, 'label': label, 'path': pp})
    return meta_data


def loop_classification(model, val_dl, device, name):
    model.eval()
    with torch.no_grad():
        outputs = []
        paths = []
        targets = []
        for i, batch_data in enumerate(tqdm(val_dl, desc="Validation...")):
            if type(batch_data["image"]) != list:
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                if inputs.size(1) == 1 and batch_data['check'][0]:
                    out = model.forward_feature(inputs)
                    outputs.append(out.cpu())
                    paths += batch_data['path']
                    targets.append(labels)
                else:
                    print(batch_data['path'][0])

        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        data_dict = {'path': paths, 'feat': outputs, 'targets': targets}
        torch.save(data_dict, 'Classify_task/腮腺良性-恶性/' + name + '-dataset.pt')


def main(model_name):
    # meta_data = make_dataset(
    #     'Classify_task/子宫肌瘤_癌肉瘤/子宫肌瘤_癌肉瘤.xlsx',
    #     '/home/khtao/MRI_foundation_model_dataset',
    #     label_keys='label',
    #     image_keys='path')
    # meta_data = make_dataset(
    #     'Classify_task/癌肉瘤腺癌透明细胞癌/癌肉瘤腺癌透明细胞癌_withpath_withlabel.xlsx',
    #     '/home/khtao/MRI_foundation_model_dataset/zheng',
    #     label_keys='肿瘤类型',
    #     image_keys='path')
    # meta_data = make_dataset(
    #     'Classify_task/直肠癌LNM/直肠癌LNM.xlsx',
    #     '/home/khtao/MRI_foundation_model_dataset/dataset',
    #     label_keys='LNM',
    #     image_keys='path')
    # meta_data=[]
    # files = list_file_tree('/home/khtao/MRI_foundation_model_dataset/dataset/EC/zhongzhong1', file_type="nii.gz")
    # for pp in files:
    #     meta_data.append({'image': pp, 'label': 0, 'path': pp})
    meta_data = make_dataset(
        'Classify_task/腮腺良性-恶性/良性-恶性.txt',
        '/home/khtao/MRI_foundation_model_dataset/kang/腮腺分类数据集')
    train_dataset = SafeDataset(meta_data, get_val_transforms('image'))
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=24,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )
    if model_name == 'SwinUNETR':
        model = SwinUNETR()
        model_path = 'pretrained_models/Swin_UNETR/model_swinvit.pt'
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict['state_dict'], strict=False)
    elif model_name == 'RP3D_Diag':
        model = RadNet()
        model_path = 'pretrained_models/RP3D_Diag/pytorch_model.bin'
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    elif model_name == 'PRISM':
        model = PRISM()
        model_path = 'pretrained_models/PRISM/PRISM.pt'
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
    elif model_name == 'UNETR':
        model = UNETR()
        model_path = 'pretrained_models/UNETR/UNETR_model_best_acc.pth'
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    elif model_name == 'BrainSegFounder':
        model = BrainSegFounder()
        model_path = 'pretrained_models/BrainSegFounder/UK-Biobank/64-gpu-model_bestValRMSE.pt'
        state_dict = torch.load(model_path, weights_only=True)
        state_dict = {k.replace('module.swinViT', 'swinViT'): v for k, v in state_dict['state_dict'].items() if
                      'swinViT' in k}
        model.load_state_dict(state_dict)
    elif model_name == 'ResNet50':
        model = ResNet50()
        model_state_dict = torch.load(
            'pretrained_models/ResNet50/models--TencentMedicalNet--MedicalNet-Resnet50/'
            'snapshots/8d924de10880fc392b71cbf61804e880aeb74e53/resnet_50_23dataset.pth')['state_dict']
        model_state_dict = {key.replace("module.", "net."): value for key, value in model_state_dict.items()}
        model.load_state_dict(model_state_dict, strict=True)
    elif model_name == 'GynoMRFound':
        model = vit_base(
            patch_size=(16, 16, 16),
            img_size=(128, 128, 128),
            clip_dim=768,
            classification=True,
            dropout_rate=0,
            n_outputs=27
        )
        model_path = 'pretrained_models/GynoMRFound/pretrained_GynoMRFound_best.pth'
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    device = "cuda"
    model.to(device)
    loop_classification(
        model=model,
        val_dl=train_loader,
        device=device,
        name='all_datasets_' + model_name
    )


if __name__ == '__main__':
    models_list = [
        'GynoMRFound'
    ]
    for mm in models_list:
        main(mm)
