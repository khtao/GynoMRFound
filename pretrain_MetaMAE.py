import argparse
import os

from glob import glob
import torch
import torch.optim as optim
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from collections import defaultdict
from model.model_MetaMAE import get_meta_mae_model
import numpy as np
from tqdm import tqdm
from utils import visualize_3d
from data.dataset_config import get_val_class_dataset, get_train_class_dataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils import multi_label
from data.augmentations import AugmentTensor
from utils.logger import Visualizer


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


def loop_pretrain_clip(
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        epochs,
        device,
        visualizer,
        save_path,
        data_parallel=False
):
    metrics = defaultdict(list)
    loss_func = torch.nn.BCEWithLogitsLoss()
    best_f1 = 0
    if data_parallel:
        model = DataParallel(model).to(device)
    for epoch in range(epochs):
        visualizer.plot('lr', optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        model.train()
        epoch_loss = 0.0
        total_len = len(train_loader)
        pbar = tqdm(total=total_len)
        aug = AugmentTensor([['resolution', 0.25], ['flip', 0.25], ['nosie', 0.25],
                             ['intensity', 0.25], ])
        for i, batch_data in enumerate(train_loader):
            inputs = batch_data['image'].to(device)
            labels = []
            for lab in batch_data["label"]:
                if lab != 'None':
                    labels.append([int(a) for a in lab])
            labels = torch.from_numpy(np.array(labels)).float().to(device)
            label_mask = batch_data["label_mask"].to(device)
            with torch.no_grad():
                inputs = aug(inputs).detach()
            optimizer.zero_grad()
            outputs, pred, mask, target = model(inputs)
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            outputs = outputs[label_mask == 1, ...]
            loss_cls = loss_func(outputs, labels)
            loss += loss_cls
            if i == 0:
                fig = visualize_3d(pred, inputs, mask)
                visualizer.img('train_image', fig)
            loss.backward()
            optimizer.step()
            pbar.set_description("epoch %d:loss=%0.5f" % (epoch, float(loss)))
            pbar.update(1)
            epoch_loss += loss.item()
        lr_scheduler.step()

        epoch_loss /= total_len
        print(f"Training loss: {epoch_loss:.4f}, Training learning rate: {lr_scheduler.get_last_lr()[0]:.8f}")
        visualizer.plot('train_loss', epoch_loss)

        if epoch == 0 or epoch == epochs - 1 or epoch % 1 == 0:
            val_loss = 0.0
            model.eval()
            preds = []
            targets = []
            with torch.no_grad():
                for i, batch_data in enumerate(tqdm(val_loader, desc="Validation...")):
                    inputs = batch_data['image'].to(device)
                    labels = []
                    for lab in batch_data["label"]:
                        if lab != 'None':
                            labels.append([int(a) for a in lab])
                    labels = torch.from_numpy(np.array(labels)).float().to(device)
                    label_mask = batch_data["label_mask"].to(device)
                    outputs, pred, mask, target = model(inputs)
                    outputs = outputs[label_mask == 1, ...]
                    loss = (pred - target) ** 2
                    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
                    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
                    preds.append(outputs)
                    targets.append(labels)
                    val_loss += loss.item()
                    if i == 0:
                        fig = visualize_3d(pred, inputs, mask)
                        visualizer.img('val_image', fig)

                val_loss /= len(val_loader)
                visualizer.plot('val_loss', val_loss)
                print(f"Validation loss: {val_loss:.4f}")
                preds = torch.cat(preds, dim=0).sigmoid()
                targets = torch.cat(targets, dim=0)
                preds_label = (preds >= 0.5).long()
                preds = preds.cpu().numpy()
                targets = targets.cpu().numpy()
                preds_label = preds_label.cpu().numpy()
                auc = multi_label(targets, preds, roc_auc_score)
                acc = multi_label(targets, preds_label, accuracy_score)
                f1 = multi_label(targets, preds_label, f1_score)
                good_auc = []
                good_f1 = []
                for a, f in zip(auc, f1):
                    if not np.isnan(a):
                        good_auc.append(a)
                        good_f1.append(f)
                mean_auc = np.mean(good_auc)
                mean_acc = np.mean(acc)
                mean_f1 = np.mean(good_f1)
                visualizer.plot('mean_acc', mean_acc)
                visualizer.plot('mean_f1', mean_f1)
                visualizer.plot('mean_auc', mean_auc)
                print('AUC:', mean_auc, auc)
                print('ACC:', mean_acc, acc)
                print('F1:', mean_f1, f1)
                if best_f1 < mean_f1:
                    best_f1 = mean_f1
                    if data_parallel:
                        torch.save(model.module.state_dict(), save_path[:-4] + '_best.pth')
                    else:
                        torch.save(model.state_dict(), save_path[:-4] + '_best.pth')
                if data_parallel:
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="base", help="model name to train")
    parser.add_argument("--save_path", type=str,
                        default="pretrained_models/pretrained_SRPMAEDirect_base.pth",
                        help="Path to save the model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", type=int, default=28, help="Batch size for training")
    parser.add_argument("--n_cpus", type=int, default=8, help="Number of cpus to use for data loading")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for training")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the model")
    parser.add_argument("--img_size", type=int, default=128, help="Image size for the model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--masking_ratio", type=float, default=0.75, help="Masking ratio for the model")
    parser.add_argument("--project", type=str, default="SRPMAE", help="Project for wandb logging")
    parser.add_argument("--seed", type=int, default=1999, help="Seed for reproducibility")
    args = parser.parse_args()
    visualizer = Visualizer(args.project)
    visualizer.print_args(args)
    set_determinism(seed=args.seed)

    train_dataset = get_train_class_dataset()
    val_dataset = get_val_class_dataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpus,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_cpus,
        pin_memory=True,
        shuffle=False,
    )
    model = get_meta_mae_model(patch_size=(args.patch_size, args.patch_size, args.patch_size),
                               img_size=(args.img_size, args.img_size, args.img_size),
                               masking_ratio=args.masking_ratio,
                               name=args.model_name,
                               num_classes=27,
                               dropout_rate=args.dropout_rate,
                               )
    model.load_state_dict(torch.load(args.save_path[:-4] + '_best.pth'))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    device = "cuda"
    model = model.to(device)
    optimizer = optim.NAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    _, _ = loop_pretrain_clip(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=device,
        visualizer=visualizer,
        save_path=args.save_path,
        data_parallel=True
    )
