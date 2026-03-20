import os
from glob import glob
import torch

from torch.utils.data import dataset, DataLoader
import cv2
import lmdb
import numpy as np
import pandas as pd
import pickle
import random
from tqdm import tqdm
from monai.data import MetaTensor
from monai.transforms import LoadImage, Spacing, EnsureChannelFirst, EnsureType, Compose, EnsureTyped
from io import BytesIO
import tifffile
from augmentations import AugmentTensor


def list_file_tree(path, file_type="tif"):
    image_list = list()
    dir_list = os.listdir(path)
    if os.path.isdir(path):
        image_list += glob(os.path.join(path, "*" + file_type))
    for dir_name in dir_list:
        sub_path = os.path.join(path, dir_name)
        if os.path.isdir(sub_path):
            image_list += list_file_tree(sub_path, file_type)
    return image_list


class SimpleDataset(dataset.Dataset):
    def __init__(self, root_path, transform):
        self.files_list = []
        if type(root_path) is list:
            for pp in root_path:
                self.files_list += list_file_tree(pp, 'nii.gz')
        else:
            self.files_list = list_file_tree(root_path, 'nii.gz')
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.files_list[index]
        try:
            image = self.transform(file_path)
            return image[0], file_path
        except:
            print('load error', file_path)
            return file_path

    def __len__(self):
        return len(self.files_list)


def bounding_box_3d(arr, thre=0):
    # 获取所有非零元素的坐标
    coords = np.argwhere(arr > thre)

    if coords.size == 0:
        # 如果没有非零元素，返回 None 或抛出异常
        return None

    # 每个维度上的最小和最大索引
    d_min, h_min, w_min = coords.min(axis=0)
    d_max, h_max, w_max = coords.max(axis=0)

    return (d_min, d_max), (h_min, h_max), (w_min, w_max)


class MedLMDBDataset(dataset.Dataset):
    def __init__(self, lmdb_root, dir_root,
                 meta_path=None, return_key='text',
                 balance=False, contained_nan=False,
                 strict_center=True,
                 n_cpu=64, mode='val',
                 pixdim=(2.0, 2.0, 1.5),
                 image_size=(128, 128, 128)):

        self.dir_root = dir_root
        self.lmdb_root = lmdb_root
        self.meta_path = meta_path
        self.mode = mode
        self.balance = balance
        self.return_key = return_key
        self.contained_nan = contained_nan
        self.pixdim = pixdim
        self.image_size = image_size
        self.strict_center = strict_center
        self.n_cpu = n_cpu
        if os.path.exists(os.path.join(self.lmdb_root, 'data.mdb')):
            self.lmdb_env = lmdb.open(lmdb_root).begin()
        else:
            self.make_lmdb()
            self.lmdb_env = lmdb.open(self.lmdb_root).begin()
        self.lmdb_meta_data = self.get_metadata_lmdb()
        if self.meta_path is not None:
            self.excel_meta_data = self.get_metadata_excel()
        else:
            self.excel_meta_data = None
        self.meta_data = self.get_metadata()
        if self.balance and (self.meta_path is not None) and self.return_key == 'label' and self.contained_nan == False:
            self.meta_data = self.balance_data()
        self.print_info()

    def get_metadata(self):
        meta_data = []
        # 如果没有元数据文件，直接返回无标签结果
        if self.meta_path is None:
            for pp in self.lmdb_meta_data:
                center_path = os.path.dirname(os.path.dirname(pp))
                meta_data.append({'path': pp, 'label': None, 'center': center_path})
            return meta_data

        # === 预处理：构建“标签目录 -> label”的映射 ===
        # 假设 self.dataset_root 是 excel_meta_data 中相对路径的基准目录
        # 例如：self.dataset_root = '/data/dataset'
        label_dir_map = {}
        nan_num = 0
        for kk in self.excel_meta_data:
            rel_path = kk['path']
            label_val = kk['label']
            if str(label_val) not in ('', 'nan'):
                label_dir_map[rel_path] = label_val
            else:
                nan_num += 1

        # 将字典的键转为集合，便于快速判断
        label_dirs_set = set(label_dir_map.keys())
        assert len(label_dirs_set) == len(self.excel_meta_data) - nan_num
        # === 处理每个图像路径 ===
        for pp in self.lmdb_meta_data:
            pp_norm = os.path.normpath(pp)
            center_path = os.path.dirname(os.path.dirname(pp_norm))
            rel_path = pp.split('/')
            parent_dir_level3 = rel_path[-4] + '/' + rel_path[-3] + '/' + rel_path[-2]
            parent_dir_level2 = rel_path[-3] + '/' + rel_path[-2]
            # 判断该父目录是否在已知标签目录中
            if parent_dir_level3 in label_dirs_set:
                label = label_dir_map[parent_dir_level3]
                known = True
            elif parent_dir_level2 in label_dirs_set:
                label = label_dir_map[parent_dir_level2]
                known = True
            else:
                label = None
                known = False

            # 根据 strict_center 和 contained_nan 决定是否保留
            should_append = False
            if self.strict_center:
                if (self.contained_nan and known) or (label is not None):
                    should_append = True
            else:
                if self.contained_nan or (label is not None):
                    should_append = True

            if should_append:
                meta_data.append({
                    'path': pp,
                    'label': label,
                    'center': center_path
                })

        return meta_data

    def get_metadata_lmdb(self):
        metadata = self.lmdb_env.get(key="metadata".encode())
        metadata = pickle.loads(metadata)
        return metadata

    def get_metadata_excel(self):
        if type(self.meta_path) is str:
            meta_path = [self.meta_path]
        else:
            meta_path = self.meta_path
        meta_data = []
        for pp in meta_path:
            data = pd.read_excel(pp)
            for i in range(len(data['path'])):
                path = data['path'][i]
                label = data[self.return_key][i]
                if str(label) == '' or str(label) == 'nan':
                    label = ''
                elif self.return_key == 'class':
                    label = label.split('_')[1]
                meta_data.append({'path': path, 'label': label})
        return meta_data

    def statistic(self):
        assert self.return_key == 'label'
        allclasses = {}
        for kk in self.meta_data:
            label = str(kk['label'])
            allclasses[label] = 1 if label not in allclasses.keys() else allclasses[label] + 1
        return allclasses

    def balance_data(self):
        class_dict = {}
        assert self.return_key == 'label'
        for kk in self.meta_data:
            label = str(kk['label'])
            if label in class_dict.keys():
                class_dict[label].append(kk)
            else:
                class_dict[label] = [kk]
        num_class = []
        for key in class_dict.keys():
            random.shuffle(class_dict[key])
            num_class.append(len(class_dict[key]))
        num = max(num_class)
        meta_data = []
        for key in class_dict.keys():
            meta_data += (class_dict[key] * (num // len(class_dict[key]) + 1))[:num]
        return meta_data

    def print_info(self):
        info = {}
        for mm in self.meta_data:
            cc = mm['center']
            path = mm['path']
            patient_path = os.path.split(path)[0]
            if cc in info.keys():
                info[cc].append(patient_path)
            else:
                info[cc] = [patient_path]
        total_patient, total_images = 0, 0
        for cc, data in info.items():
            pat = len(set(data))
            img = len(data)
            total_patient += pat
            total_images += img
            print(cc, 'patient:', pat, 'images:', img)
        print('center:', len(info.keys()), 'total_patient:', total_patient, 'total_images:', total_images)
        if self.return_key == 'label':
            print(self.statistic())

    def __len__(self):
        return len(self.meta_data)

    def make_lmdb(self):
        env = lmdb.open(self.lmdb_root, map_size=1099511627776)
        tsf = Compose([
            LoadImage(),
            EnsureChannelFirst(),
            Spacing(self.pixdim, mode="bilinear", ),
        ])
        correct_meta = []
        data = SimpleDataset(root_path=self.dir_root, transform=tsf)
        dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=self.n_cpu)
        size_min = min(self.image_size) // 4
        total_len = len(dataloader)
        pbar = tqdm(total=total_len)
        for batch in dataloader:
            if len(batch) == 2:
                image, file_path = batch
                file_path = file_path[0]
                image = image[0]
                file_info = image.meta
                image = image.numpy().astype(np.uint16)
                old_shape = image.shape
                if image.max() == image.min() or image is None:
                    print('error image value max==min ')
                elif len(image.shape) != 3 or min(image.shape) < size_min:
                    print('2D or 4D image shape not supported', image.shape)
                else:
                    bbox = bounding_box_3d(image, image.max() * 0.01)
                    image = image[bbox[0][0]:bbox[0][1] + 1, bbox[1][0]:bbox[1][1] + 1, bbox[2][0]:bbox[2][1] + 1]
                    txn = env.begin(write=True)
                    byte_stream = BytesIO()
                    tifffile.imwrite(byte_stream, image, compression='JPEGXL')
                    encoded_bytes = byte_stream.getvalue()
                    txn.put(key=(file_path + '-image').encode(), value=encoded_bytes)
                    txn.put(key=file_path.encode(), value=pickle.dumps(file_info))
                    correct_meta.append(file_path)
                    txn.commit()
                pbar.set_description(f'old_shape:{old_shape}, shape:{image.shape}')
            pbar.update(1)
        txn = env.begin(write=True)
        metadata = pickle.dumps(correct_meta)
        txn.put(key="metadata".encode(), value=metadata)
        txn.commit()

    def to_target_size(self, image):
        x, y, z = image.shape
        tx, ty, tz = self.image_size
        center = [x // 2, y // 2, z // 2]
        image_type = image.dtype
        if x >= tx:
            if self.mode == 'train':
                p = random.randint(0, x - tx)
            else:
                p = max(min(center[0] - tx // 2, x - tx), 0)
            image = image[p:p + tx, :, :]
        else:
            diff_x = (tx - x) // 2
            pad_left = np.zeros((diff_x, y, z), dtype=image_type)
            pad_right = np.zeros((tx - x - diff_x, y, z), dtype=image_type)
            image = np.concatenate([pad_left, image, pad_right], axis=0)

        if y >= ty:
            if self.mode == 'train':
                p = random.randint(0, y - ty)
            else:
                p = max(min(center[1] - ty // 2, y - ty), 0)
            image = image[:, p:p + ty, :]
        else:
            diff_y = (ty - y) // 2
            pad_left = np.zeros((tx, diff_y, z), dtype=image_type)
            pad_right = np.zeros((tx, ty - y - diff_y, z), dtype=image_type)
            image = np.concatenate([pad_left, image, pad_right], axis=1)

        if z >= tz:
            if self.mode == 'train':
                p = random.randint(0, z - tz)
            else:
                p = max(min(center[2] - tz // 2, z - tz), 0)
            image = image[:, :, p:p + tz]
        else:
            diff_z = (tz - z) // 2
            pad_left = np.zeros((tx, ty, diff_z), dtype=image_type)
            pad_right = np.zeros((tx, ty, tz - z - diff_z), dtype=image_type)
            image = np.concatenate([pad_left, image, pad_right], axis=2)
        return image

    def __getitem__(self, idx):
        metadata = self.meta_data[idx]
        file_path = metadata['path']
        buff = self.lmdb_env.get(key=(file_path + '-image').encode())
        image = tifffile.imread(BytesIO(buff))
        if len(image.shape) == 5 and image.shape[-1] < 3:
            print(file_path)
            image = image[..., 0]
        image = self.to_target_size(image)
        if image.std() == 0:
            image = (image.astype(np.float32) - image.mean())
        else:
            image = (image.astype(np.float32) - image.mean()) / image.std()

        data = {'image': image.astype(np.float32)[np.newaxis, ...], 'label': str(metadata['label'])}
        if metadata['label'] is None:
            data['label_mask'] = np.array(0)
        else:
            data['label_mask'] = np.array(1)
        return data


def random_split_xlsx(path, ratio=0.8):
    df = pd.read_excel(path)
    total = len(df)
    num_train = int(total * ratio)
    ind = list(np.arange(total))
    random.shuffle(ind)
    train_ind = ind[:num_train]
    val_ind = ind[num_train:]
    train_set = df.iloc[train_ind]
    val_set = df.iloc[val_ind]
    train_set.to_excel(path[:-5] + '_train.xlsx', index=False)
    val_set.to_excel(path[:-5] + '_val.xlsx', index=False)
    print('finished', path)


if __name__ == '__main__':
    kk = MedLMDBDataset('/home/khtao/cache/temp',
                        '/home/khtao/mri_datasets/kang/shenshan',
                        ['report_with_multilabel/中山肿瘤医院.xlsx',
                         'report_with_multilabel/深汕妇科_ori.xlsx'],
                        return_key='class',
                        contained_nan=False
                        )
    print(kk[0])
