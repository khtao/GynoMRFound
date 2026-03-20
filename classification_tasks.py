import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
from rrt.rrt import RRTMIL
from data.mil_dataset import MILDataset, MILFoldDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, balanced_accuracy_score
from utils.logger import Visualizer
import numpy as np
import random
import time


def create_file(filename):
    filename += time.strftime('_%m%d_%H%M%S.pt')
    if os.path.exists(filename):
        i = 1
        while os.path.exists(filename + '_' + str(i)):
            i += 1
        filename = filename + '_' + str(i)
    return filename


def test(model, val_loader, device, loss_fn):
    model.eval()
    metric_dict = {}
    preds = []
    targets = []
    path_list = []
    val_loss = 0
    for step, (x, y, pp) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)
        path_list += pp
        with torch.no_grad():
            output = model(x)
        preds.append(output)
        targets.append(y)
        loss = loss_fn(output, y)
        val_loss += loss.item()
    val_loss /= len(val_loader)
    metric_dict['loss'] = float(val_loss)
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    preds = preds.softmax(dim=-1)
    result_dict = {
        'preds': preds.cpu().numpy(),
        'targets': targets.cpu().numpy(),
        'path_list': path_list,
    }
    preds_label = preds.argmax(dim=-1)
    if targets.max() > 1:
        preds_scores = preds
    else:
        preds_scores = preds[:, 1]
    targets = targets.detach().cpu().numpy()
    preds_label = preds_label.detach().cpu().numpy()
    preds_scores = preds_scores.detach().cpu().numpy()
    if targets.max() > 1:
        micro_auc = roc_auc_score(targets, preds_scores, multi_class='ovr', average='micro')
        macro_auc = roc_auc_score(targets, preds_scores, multi_class='ovo', average='macro')
        weighted_auc = roc_auc_score(targets, preds_scores, multi_class='ovo', average='weighted')
        auc = max([micro_auc, macro_auc, weighted_auc])
        metric_dict['micro_auc'] = float(micro_auc)
        metric_dict['macro_auc'] = float(macro_auc)
        metric_dict['weighted_auc'] = float(weighted_auc)
        metric_dict['auc'] = float(auc)

    else:
        auc = roc_auc_score(targets, preds_scores)
        metric_dict['auc'] = float(auc)
    print(confusion_matrix(targets, preds_label))
    accuracy = accuracy_score(targets, preds_label)
    balanced_accuracy = balanced_accuracy_score(targets, preds_label)
    metric_dict['accuracy'] = float(accuracy)
    metric_dict['balanced_accuracy'] = float(balanced_accuracy)
    return metric_dict, result_dict


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(return_key, train_path, test_path, lr, balance, fold_path, feat_path):
    epochs = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = return_key.split('（')[0].split('：')[0]

    visualizer = Visualizer(f'子宫内膜癌-{key}', vis_root='子宫内膜癌-最终结果3')
    print('feat_path=', feat_path)
    test_dataset = MILDataset(feat_path, return_key=return_key, meta_path=test_path, max_feat=20, mode='val')
    val_dataset = MILFoldDataset(feat_path, return_key=return_key, meta_path=train_path,
                                 fold_path=fold_path, fold_num=0,
                                 max_feat=20, mode='val')
    train_dataset = MILFoldDataset(feat_path, return_key=return_key, meta_path=train_path, max_feat=20,
                                   fold_path=fold_path, fold_num=0,
                                   mode='train', balance=balance)
    setup_seed(3407)
    task_type = 'classification'
    if task_type == 'regression':
        task_class_num = 1
    else:
        task_class_num = test_dataset.max_label + 1
    temp = val_dataset[0][0]
    model = RRTMIL(input_dim=int(temp.size(1)), mlp_dim=2048, trans_dim=128, n_classes=task_class_num,
                   trans_dropout=0.1,
                   dropout=0.8).to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    model = model.to(device)
    print('task_key', key, 'lr', lr, 'balance', balance)
    print('train data', train_path, )
    print('test data', test_path, )
    print('fold_path', fold_path, )
    if task_type == 'regression':
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    best_metric = {}
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        visualizer.plot('lr', optimizer.param_groups[0]["lr"])
        for step, (x, y, _) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            if task_type == 'regression':
                y = y.unsqueeze(-1).float()
            output = model(x)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_metric, val_result = test(model, val_loader, device, loss_fn)
        test_metric, test_result = test(model, test_loader, device, loss_fn)
        mean_metric = {'mean_' + k: (val_metric[k] + test_metric[k]) / 2 for k in val_metric.keys()}
        val_metric = {'val_' + k: v for k, v in val_metric.items()}
        test_metric = {'test_' + k: v for k, v in test_metric.items()}

        visualizer.plot_many(val_metric)
        visualizer.plot_many(test_metric)
        visualizer.plot_many(mean_metric)
        print(val_metric)
        print(test_metric)
        print(mean_metric)
        for k, v in mean_metric.items():
            if 'loss' not in k:
                if k in best_metric.keys():
                    if v > best_metric[k]:
                        best_metric[k] = v
                        visualizer.save_model(model.state_dict(), f'best_{k}.pt')
                        print(f'save best {k} model: {v:0.4f}')
                        best_epoch = epoch
                        if 'auc' in k:
                            visualizer.save_model(test_result, f'best_test_result.pt')
                            visualizer.save_model(val_result, f'best_val_result.pt')

                else:
                    best_metric[k] = v
                    visualizer.save_model(model.state_dict(), f'best_{k}.pt')
                    print(f'save best {k} model: {v:0.4f}')
                    best_epoch = epoch
                    if 'auc' in k:
                        visualizer.save_model(test_result, f'best_test_result.pt')
                        visualizer.save_model(val_result, f'best_val_result.pt')
        if epoch - best_epoch > 10:
            break
        lr_scheduler.step()


if __name__ == '__main__':
    keys = ['宫旁浸润', '淋巴结转移（0：无；1：有）', '卵巢', '侵犯宫颈', '绝经（0-否，1-是）', 'P53', 'KI67']
    param_dict = {
        '侵犯宫颈': [8.5e-05, 0.15],
        'KI67': [2e-05, 0.8],
        '卵巢': [1e-05, 0.8],
        '宫旁浸润': [9e-05, 0.9],
        '淋巴结转移（0：无；1：有）': [6e-05, 1.0],
        '绝经（0-否，1-是）': [8e-05, None],
        'P53': [5e-05, 1.0],
        'P16': [8e-05, 0.2],
        '肌层深度：0：无；1：小于1/2；2：大于1/2；3：全层）': [2e-5, None],
        'label': [1e-5, None],
        '输卵管': [3e-05, 0.2],
        'ER': [1e-5, None],
        '脉管癌栓或神经束': [0.0001, 0.3],
        'PR': [6e-05, None],
        '病理分期': [2e-05, None],
        '分化程度（0：高分化；1：中分化；2低分化）': [2e-5, None],
    }
    train_splits = ['Classify_task/可用下游任务表格/子宫内膜癌/风险分层-中山肿瘤医院.xlsx',
                    'Classify_task/可用下游任务表格/子宫内膜癌/风险分层-dongguan.xlsx',
                    'Classify_task/可用下游任务表格/子宫内膜癌/风险分层-深汕妇科_ori.xlsx',
                    'Classify_task/可用下游任务表格/子宫内膜癌/风险分层-所见及结论_中山二院.xlsx',
                    ]
    test_split = [
        'Classify_task/可用下游任务表格/子宫内膜癌/风险分层-foushan.xlsx',
    ]
    best_fold = 'Classify_task/best_fold/子宫内膜癌_最佳分布_1217_214028.pt'
    test_methods = ['datasets/all_datasets_SRPMAEBert-dataset.pt']
    for ff in test_methods:
        for kk, (tt, bb) in param_dict.items():
            train(kk, train_splits, test_split, tt, bb, best_fold, ff)
