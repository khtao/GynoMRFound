import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.nn as nn
from rrt.rrt import RRTMIL
from data.mil_dataset import MILFoldDataset
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
    path_ids = []
    val_loss = 0
    num_class_list = [6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2]
    res_dict = {'num_class_list': num_class_list}
    for step, (x, y, pp) in enumerate(val_loader):
        x = x.to(device)
        label = torch.tensor([[int(b) for b in a] for a in y]).long().to(device)
        with torch.no_grad():
            output = model(x)
            preds.append(output)
            targets.append(label)
            path_ids += pp
            loss_list = []
            n = 0
            for t, num in enumerate(num_class_list):
                loss_list.append(loss_fn(output[:, n:n + num], label[:, t]))
                n += num

            loss = torch.mean(torch.stack(loss_list))
            val_loss += loss.item()
    val_loss /= len(val_loader)
    metric_dict['loss'] = float(val_loss)
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    res_dict['outputs'] = preds.cpu().numpy()
    res_dict['targets'] = targets.cpu().numpy()
    res_dict['path_ids'] = path_ids
    n = 0
    acc_list = []
    auc_list = []
    for t, num in enumerate(num_class_list):
        pp = preds[:, n:n + num]
        ll = targets[:, t]
        n += num
        class_num = list(set(ll.cpu().numpy()))
        class_num.sort()
        cls2_dict = {}
        for i in range(len(class_num)):
            cls2_dict[class_num[i]] = i

        pp = pp[:, class_num].softmax(dim=-1)

        pp_label = pp.argmax(dim=-1)
        pp = pp.cpu().numpy()
        ll = ll.detach().cpu().numpy()
        ll = [cls2_dict[a] for a in ll]
        if max(ll) > 1:
            auc = roc_auc_score(ll, pp, average='macro', multi_class='ovo')
        elif max(ll) == 1:
            auc = roc_auc_score(ll, pp[:, 1])
        else:
            auc = 1.0
        auc_list.append(float(auc))
        pp_label = pp_label.detach().cpu().numpy()
        print(confusion_matrix(ll, pp_label))
        accuracy = accuracy_score(ll, pp_label)
        acc_list.append(float(accuracy))

    print('acc_list', acc_list)
    print('auc_list', auc_list)
    metric_dict['accuracy'] = np.mean(acc_list)
    metric_dict['auc'] = np.mean(auc_list)
    return metric_dict, res_dict


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(return_key, train_path, lr, balance, fold_path, feat_path):
    epochs = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = return_key.split('（')[0].split('：')[0]

    visualizer = Visualizer(f'report-{key}', vis_root='report-最终结果2')
    print('feat_path=', feat_path)
    val_dataset = MILFoldDataset(feat_path, return_key=return_key, meta_path=train_path,
                                 fold_path=fold_path, fold_num=0, task_type='report',
                                 max_feat=30, mode='val')
    train_dataset = MILFoldDataset(feat_path, return_key=return_key, meta_path=train_path, max_feat=30,
                                   fold_path=fold_path, fold_num=0, task_type='report',
                                   mode='train', balance=balance)
    setup_seed(3407)
    task_class_num = 79
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

    model = model.to(device)
    print('task_key', key, 'lr', lr, 'balance', balance)
    print('train data', train_path, )
    print('fold_path', fold_path, )
    loss_fn = nn.CrossEntropyLoss()
    loss_fn_bce = nn.BCEWithLogitsLoss()
    model = model.to(device)
    best_metric = {}
    best_epoch = 0
    num_class_list = [6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2]
    for epoch in range(epochs):
        model.train()
        visualizer.plot('lr', optimizer.param_groups[0]["lr"])
        for step, (x, y, _) in enumerate(train_loader):
            x = x.to(device)
            label = torch.tensor([[int(b) for b in a] for a in y]).long().to(device)
            output = model(x)
            loss_list = []
            n = 0
            for t, num in enumerate(num_class_list):
                binary_label = (label[:, t] < 1).float()  # 0->1;other->0
                loss_list.append(loss_fn(output[:, n:n + num], label[:, t])
                                 + loss_fn_bce(output[:, n], binary_label))
                n += num
            loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_metric, _ = test(model, val_loader, device, loss_fn)
        val_metric = {'val_' + k: v for k, v in val_metric.items()}

        visualizer.plot_many(val_metric)
        print(val_metric)
        for k, v in val_metric.items():
            if 'loss' not in k:
                if k in best_metric.keys():
                    if v > best_metric[k]:
                        best_metric[k] = v
                        visualizer.save_model(model.state_dict(), f'best_{k}.pt')
                        best_epoch = epoch
                        print(f'save best {k} model: {v:0.4f}')
                else:
                    best_metric[k] = v
                    visualizer.save_model(model.state_dict(), f'best_{k}.pt')
                    best_epoch = epoch
                    print(f'save best {k} model: {v:0.4f}')
        if epoch - best_epoch > 50:
            break
        lr_scheduler.step()





if __name__ == '__main__':
    # output_final_result(return_key='structure_class', root_path='report-最终结果2')
    # # #
    # write2excel(root_path='report-最终结果2')
    train_splits = ['all_report_dataset/new_report_dataset.xlsx']
    test_methods = [
        'all_report_dataset/all_datasets_BrainSegFounderNew-dataset.pt',
        'all_report_dataset/all_datasets_PRISMNew-dataset.pt',
        'all_report_dataset/all_datasets_ResNet50New-dataset.pt',
        'all_report_dataset/all_datasets_SRPMAEv2New-dataset.pt',
        'all_report_dataset/all_datasets_SwinUNETRNew-dataset.pt',
        'all_report_dataset/all_datasets_UNETRNew-dataset.pt',
    ]
    best_fold = f'all_report_dataset/fold_new_19.pt'

    for mm in test_methods:
        train('structure_class', train_splits, 1e-5, None, best_fold, mm)
