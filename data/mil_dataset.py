import json
import os.path
import shutil

import numpy as np
from torch.utils.data import dataset
import torch
import pandas as pd


class MILDataset(dataset.Dataset):
    def __init__(self, path, return_key, meta_path, max_feat=20, mode='val', balance=None, task_type='classification'):
        self.feature_data = torch.load(path, weights_only=False, map_location='cpu')
        self.meta_path = meta_path
        self.return_key = return_key
        self.max_feat = max_feat
        self.task_type = task_type
        self.mode = mode
        self.excel_meta_data = self.get_metadata_excel()
        self.meta_data = self.get_metadata()
        self.max_label = max([a['label'] for a in self.meta_data])
        # print(self.statistic())
        if balance is not None:
            self.meta_data = self.balance_data(balance)
            # print(self.statistic())

    def get_metadata(self):
        meta_data_dict = {}
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
        for i in range(len(self.feature_data['path'])):
            pp = self.feature_data['path'][i]
            rel_path = pp.split('/')
            parent_dir_level3 = rel_path[-4] + '/' + rel_path[-3] + '/' + rel_path[-2]
            parent_dir_level2 = rel_path[-3] + '/' + rel_path[-2]
            parent_dir_level0 = rel_path[-3]
            # 判断该父目录是否在已知标签目录中
            if parent_dir_level3 in label_dirs_set:
                label = label_dir_map[parent_dir_level3]
                known_key = parent_dir_level3
            elif parent_dir_level2 in label_dirs_set:
                label = label_dir_map[parent_dir_level2]
                known_key = parent_dir_level2
            elif parent_dir_level0 in label_dirs_set:
                label = label_dir_map[parent_dir_level0]
                known_key = parent_dir_level0
            else:
                label = None
                known_key = None

            # 根据 strict_center 和 contained_nan 决定是否保留
            if label is not None and known_key is not None:
                if known_key not in meta_data_dict.keys():
                    meta_data_dict[known_key] = {'feat': [self.feature_data['feat'][i]], 'label': label}
                else:
                    meta_data_dict[known_key]['feat'].append(self.feature_data['feat'][i])
        meta_data = []
        feat_nums = []
        for k, v in meta_data_dict.items():
            v['path'] = k
            feat_nums.append(len(v['feat']))
            meta_data.append(v)
        # print('max', max(feat_nums), 'min', min(feat_nums),
        #       'mean', np.mean(feat_nums), 'median', np.median(feat_nums),
        #       'top', np.percentile(feat_nums, [50, 75, 85, 95]), )
        return meta_data

    def get_metadata_excel(self):
        if type(self.meta_path) is str:
            meta_path = [self.meta_path]
        else:
            meta_path = self.meta_path
        meta_data = []
        for pp in meta_path:
            if pp[-4:] in ['xlsx', '.xls']:
                data = pd.read_excel(pp)
                if self.return_key in data.keys():
                    for i in range(len(data['path'])):
                        path = data['path'][i]
                        label = data[self.return_key][i]
                        if str(label) != '' and str(label) != 'nan':
                            if self.return_key == 'class':
                                label = label.split('_')[1]
                            else:
                                try:
                                    label = int(label)
                                except:
                                    label = None
                            meta_data.append({'path': path, 'label': label})
            elif pp[-4:] in ['.txt']:
                data = open(pp).read().splitlines()
                for line in data:
                    line_split = line.split(' ')
                    path = line_split[0]
                    label = int(line_split[1])
                    meta_data.append({'path': path, 'label': label})
            else:
                raise NotImplementedError

        return meta_data


    def statistic(self):
        allclasses = {}
        for kk in self.meta_data:
            label = str(kk['label'])
            allclasses[label] = 1 if label not in allclasses.keys() else allclasses[label] + 1
        max_num = max(v for k, v in allclasses.items())
        self.balance_dict = {k: v / max_num for k, v in allclasses.items()}
        return allclasses

    def balance_data(self, balance):
        class_dict = {}
        for kk in self.meta_data:
            label = str(kk['label'])
            if label in class_dict.keys():
                class_dict[label].append(kk)
            else:
                class_dict[label] = [kk]
        num_class = []
        for key in class_dict.keys():
            np.random.shuffle(class_dict[key])
            num_class.append(len(class_dict[key]))
        num = int(max(num_class) * balance)
        meta_data = []
        for key in class_dict.keys():
            if len(class_dict[key]) < max(num_class):
                meta_data += (class_dict[key] * (num // len(class_dict[key]) + 1))[:num]
            else:
                meta_data += class_dict[key]
        return meta_data

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, item):
        meta_data = self.meta_data[item]
        meta_feat = meta_data['feat']
        if self.mode == 'train':
            np.random.shuffle(meta_feat)
        meta_feat = torch.stack(meta_feat, dim=0)
        if meta_feat.shape[0] > self.max_feat:
            meta_feat = meta_feat[:self.max_feat]
        else:
            meta_feat = torch.concatenate([meta_feat, torch.zeros(self.max_feat - meta_feat.size(0),
                                                                  meta_feat.size(1))], dim=0)
        label = meta_data['label']
        return meta_feat, label, meta_data['path']


class MILFoldDataset(dataset.Dataset):
    def __init__(self, path_list, return_key, meta_path, fold_num=0, fold_path=None, max_feat=20, mode='val',
                 balance=None, task_type='classification'):
        self.path_list = path_list
        self.return_key = return_key
        self.meta_path = meta_path
        self.task_type = task_type
        if type(self.path_list) is str:
            self.path_list = [self.path_list]
        for i in range(len(self.path_list)):
            if i == 0:
                self.data = torch.load(self.path_list[i], weights_only=False, map_location='cpu')
            else:
                temp = torch.load(self.path_list[i], weights_only=False, map_location='cpu')
                for k in temp.keys():
                    self.data[k] += temp[k]

        self.max_feat = max_feat
        self.mode = mode
        self.fold_num = fold_num
        self.fold_path = fold_path
        self.excel_meta_data = self.get_metadata_excel()
        self.meta_data = self.get_metadata()
        all_labels = list(set([a['label'] for a in self.meta_data]))
        if self.task_type == 'classification':
            self.max_label = max(all_labels)
        elif self.task_type == 'regression':
            self.map_factor = (int(max(all_labels) / 10) + 1) * 10
        if self.task_type == 'classification':
            # print(self.statistic())
            if balance is not None:
                self.meta_data = self.balance_data(balance)
                # print(self.statistic())

    def get_metadata(self):
        meta_data_dict = {}
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
        # assert len(label_dirs_set) == len(self.excel_meta_data) - nan_num
        # === 处理每个图像路径 ===
        for i in range(len(self.data['path'])):
            pp = self.data['path'][i]
            rel_path = pp.split('/')
            parent_dir_level3 = rel_path[-4] + '/' + rel_path[-3] + '/' + rel_path[-2]
            parent_dir_level2 = rel_path[-3] + '/' + rel_path[-2]
            # 判断该父目录是否在已知标签目录中
            if parent_dir_level3 in label_dirs_set:
                label = label_dir_map[parent_dir_level3]
                known_key = parent_dir_level3
            elif parent_dir_level2 in label_dirs_set:
                label = label_dir_map[parent_dir_level2]
                known_key = parent_dir_level2
            else:
                label = None
                known_key = None

            # 根据 strict_center 和 contained_nan 决定是否保留
            if label is not None and known_key is not None:
                if known_key not in meta_data_dict.keys():
                    meta_data_dict[known_key] = {'feat': [self.data['feat'][i]], 'label': label}
                else:
                    meta_data_dict[known_key]['feat'].append(self.data['feat'][i])
        data_list = []
        if self.fold_path is None:
            self.fold_path = self.path_list[0][:-3] + '_fold5.pt'
        if os.path.exists(self.fold_path):
            fold_data = torch.load(self.fold_path, weights_only=False, map_location='cpu')
        else:
            all_list = list(meta_data_dict.keys())
            np.random.seed(None)
            np.random.shuffle(all_list)
            every_num = len(all_list) // 5
            fold_data = [all_list[i * every_num:(i + 1) * every_num] for i in range(5)]
            torch.save(fold_data, self.fold_path)
        if self.mode == 'train':
            for i in range(len(fold_data)):
                if i != self.fold_num:
                    data_list += fold_data[i]
        else:
            data_list = fold_data[self.fold_num]
        meta_data = []
        feat_nums = []
        for k, v in meta_data_dict.items():
            feat_nums.append(len(v['feat']))
            if k in data_list:
                v['path'] = k
                meta_data.append(v)
        # print('max', max(feat_nums), 'min', min(feat_nums),
        #       'mean', np.mean(feat_nums), 'median', np.median(feat_nums),
        #       'top', np.percentile(feat_nums, [50, 75, 85, 95]), )
        return meta_data

    def get_metadata_excel(self):
        if type(self.meta_path) is str:
            meta_path = [self.meta_path]
        else:
            meta_path = self.meta_path
        meta_data = []
        for pp in meta_path:
            data = pd.read_excel(pp)
            if self.return_key in data.keys():
                for i in range(len(data['path'])):
                    path = data['path'][i]
                    label = data[self.return_key][i]
                    if str(label) != '' and str(label) != 'nan':
                        if self.return_key in ['class', 'structure_class']:
                            label = label.split('_')[1]
                        else:
                            try:
                                label = int(label)
                            except:
                                label = None
                        meta_data.append({'path': path, 'label': label})

        return meta_data

    def statistic(self):
        allclasses = {}
        for kk in self.meta_data:
            label = kk['label']
            label = str(label)
            allclasses[label] = 1 if label not in allclasses.keys() else allclasses[label] + 1

        return allclasses

    def balance_data(self, balance):
        class_dict = {}
        for kk in self.meta_data:
            label = str(kk['label'])
            if label in class_dict.keys():
                class_dict[label].append(kk)
            else:
                class_dict[label] = [kk]
        num_class = []
        for key in class_dict.keys():
            np.random.shuffle(class_dict[key])
            num_class.append(len(class_dict[key]))
        num = int(max(num_class) * balance)
        meta_data = []
        for key in class_dict.keys():
            if len(class_dict[key]) < max(num_class):
                meta_data += (class_dict[key] * (num // len(class_dict[key]) + 1))[:num]
            else:
                meta_data += class_dict[key]
        return meta_data

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, item):
        meta_data = self.meta_data[item]
        meta_feat = meta_data['feat']
        if self.mode == 'train':
            np.random.shuffle(meta_feat)
        meta_feat = torch.stack(meta_feat, dim=0)
        if meta_feat.shape[0] > self.max_feat:
            meta_feat = meta_feat[:self.max_feat]
        else:
            meta_feat = torch.concatenate([meta_feat, torch.zeros(self.max_feat - meta_feat.size(0),
                                                                  meta_feat.size(1))], dim=0)

        label = meta_data['label']
        if self.task_type == 'classification':
            pass
        elif self.task_type == 'regression':
            label = label / self.map_factor
        return meta_feat, label, meta_data['path']


if __name__ == '__main__':
    print('hello')
    import SimpleITK as sitk

    data = open('/home/khtao/MRI_foundation_model_dataset/Public_Abdomen/atlas-train-dataset-1.0.1/train/dataset.json')
    data = json.load(data)
    files = []
    for dd in data['training']:
        files.append(dd['image'] + ',' + dd['label'])

    save_root = '/home/khtao/MRI_foundation_model_dataset/Public_Abdomen/atlas-train-dataset-1.0.1/train'
    total = len(files)
    # for line in data:
    #     if cc[jj] in line:
    #         pp = line.split(', ')[0].split('/')[-2]
    #         if pp not in data_dict.keys():
    #             data_dict[pp] = []
    #         data_dict[pp].append(line)
    #         total += 1
    # keys_list = list(data_dict.keys())
    np.random.shuffle(files)
    every_fold = int(total / 5)
    fs_list = []
    for i in range(5):
        train_fs = open(save_root + f'/cross_fold{i}.txt', 'w')
        fs_list.append(train_fs)
    num = 0
    for pp in files:
        fs = fs_list[min(num // every_fold, 4)]
        fs.write(pp + '\n')
        num += 1
    # fs.close()
    # fs = open(root_path + '/val_list.txt', 'w')
    # for i in range(train_num, total):
    #     ori_dir = data_list[i][0]
    #     mask_dir = data_list[i][1]
    #     fs.write(ori_dir + ', ' + mask_dir + '\n')
    # fs.close()
    # my_dataset = MILFoldDataset(path_list='datasets/train-风险分层-vit3d-all-datasets-update-dataset.pt',
    #                             return_key='Ki67（小于等于50%：弱阳性，大于50%：强阳性）', mode='train',
    #                             meta_path=[
    #                                 '/mnt/Dataset/Foundation_Model/下游任务/可用下游任务表格/子宫内膜癌/风险分层-深汕妇科_ori.xlsx',
    #                                 '/mnt/Dataset/Foundation_Model/下游任务/可用下游任务表格/子宫内膜癌/风险分层-所见及结论_中山二院.xlsx',
    #                                 '/mnt/Dataset/Foundation_Model/下游任务/可用下游任务表格/子宫内膜癌/风险分层-中山肿瘤医院.xlsx',
    #                                 '/mnt/Dataset/Foundation_Model/下游任务/可用下游任务表格/子宫内膜癌/风险分层-dongguan.xlsx',
    #                                 '/mnt/Dataset/Foundation_Model/下游任务/可用下游任务表格/子宫内膜癌/风险分层-foushan.xlsx',
    #                                 '/mnt/Dataset/Foundation_Model/下游任务/可用下游任务表格/子宫内膜癌/风险分层-shantou.xlsx'])
    # print(my_dataset[0])
