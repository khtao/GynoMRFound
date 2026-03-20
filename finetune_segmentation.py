import argparse
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from monai.data import Dataset
from monai.networks.nets import UNETR, UNet, BasicUNetPlusPlus, SwinUNETR
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from loss import DC_and_CE_loss, MemoryEfficientSoftDiceLoss
from data.augmentations import get_train_seg_transforms, get_val_seg_transforms
from utils.logger import Visualizer
from monai.metrics import DiceHelper
from collections import defaultdict
from tqdm import tqdm


def loop_segmentation(
        epochs, model, loss_fn, opt, scheduler, train_dl, valid_dl, device, path_to_save, visualizer
):
    dice_func = DiceHelper(include_background=True, reduction="mean_batch", softmax=True, sigmoid=False)
    metrics = defaultdict(list)
    best_dice = 0
    best_epoch = 0
    for epoch in range(epochs):
        visualizer.plot('lr', opt.param_groups[0]["lr"])
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        model.train()
        epoch_loss = 0.0

        for i, batch_data in enumerate(tqdm(train_dl, desc="Training...")):
            inputs, mask = (batch_data["image"].to(device), batch_data["label"].to(device).long())
            opt.zero_grad()
            outputs = model(inputs)
            if type(outputs) is list:
                outputs = outputs[0]
            loss = loss_fn(outputs, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            opt.step()
            epoch_loss += float(loss.item())
            if i == 0:
                max_layer = mask[0, 0].sum(dim=[1, 2]).argmax()
                max_mask = mask[0, 0, :, :, max_layer] / (outputs.size(1) - 1)
                img = inputs[0, 0, :, :, max_layer]
                img = (img - img.min()) / (img.max() - img.min())
                max_pred = 1 - outputs.softmax(dim=1)[0, 0, :, :, max_layer]
                visualizer.img_many({'train_image': img, 'train_mask': max_mask, 'train_out': max_pred})

        scheduler.step()
        epoch_loss /= len(train_dl)
        print(f"Training loss: {epoch_loss:.4f}, Training learning rate: {scheduler.get_last_lr()[0]:.8f}")
        visualizer.plot('train_loss', epoch_loss)

        model.eval()
        with torch.no_grad():
            val_dice, val_nans = None, None
            val_loss = 0.0
            for i, batch_data in enumerate(tqdm(valid_dl, desc="Validation...")):
                inputs, mask = (batch_data["image"].to(device), batch_data["label"].to(device).long())
                outputs = model(inputs)
                if type(outputs) is list:
                    outputs = outputs[0]
                loss = loss_fn(outputs, mask)
                val_loss += float(loss.item())
                try:
                    outputs = outputs.as_tensor()
                except:
                    pass
                # put everything on cpu
                outputs = outputs.cpu()
                mask = mask.cpu()
                if i == 0:
                    max_layer = mask[0, 0].sum(dim=[1, 2]).argmax()
                    max_mask = mask[0, 0, :, :, max_layer] / (outputs.size(1) - 1)
                    img = inputs[0, 0, :, :, max_layer]
                    img = (img - img.min()) / (img.max() - img.min())
                    max_pred = 1 - outputs.softmax(dim=1)[0, 0, :, :, max_layer]
                    visualizer.img_many({'val_image': img, 'val_mask': max_mask, 'val_out': max_pred})
                socre, not_nans = dice_func(y_pred=outputs, y=mask)
                if val_dice is None:
                    val_dice = socre * not_nans
                else:
                    val_dice += socre * not_nans
                if val_nans is None:
                    val_nans = not_nans
                else:
                    val_nans += not_nans
            dice = (val_dice / val_nans).mean()
            print(val_dice / val_nans, val_nans)
            visualizer.plot('val_dice', dice)
            val_loss /= len(valid_dl)
            visualizer.plot('val_loss', val_loss)
            if path_to_save is not None:
                if best_dice < dice:
                    visualizer.save_model(model.state_dict(), path_to_save)
                    best_dice = dice
                    print(f"Best Dice: {best_dice:.4f}")
                    best_epoch = epoch

            print(
                f"Validation loss: {val_loss:.4f}\n"
                f"Dice: {dice:.4f}\n "
            )
            if epoch - best_epoch >= 50:
                print('early stopping is working')
                break

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/home/khtao/MRI_foundation_model_dataset/kang/endometrial_cancer_dataset",
                        help="Path to the data directory")
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--n_cpus", type=int, default=16, help="Number of cpus to use for data loading")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for training")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the model")
    parser.add_argument("--img_size", type=int, default=128, help="Image size for the model")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size for the model")
    parser.add_argument("--mlp_dim", type=int, default=3072, help="Number of layers for the model")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads for the model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--project", type=str, default="子宫内膜癌分割", help="Project for wandb logging")
    parser.add_argument("--seed", type=int, default=1999, help="Seed for reproducibility")
    # parser.add_argument("--fold", type=int, default=0, help="fold number for 5 fold cross validation")
    parser.add_argument("--model", type=str, default='UNETR')
    args = parser.parse_args()
    visualizer = Visualizer(args.project + f'-{args.model}', vis_root='分割-Result')
    visualizer.print_args(args)
    set_determinism(seed=args.seed)
    val_data = open(os.path.join(args.data_path, f'val_all_repair2d.csv')).read().splitlines()
    train_data = open(os.path.join(args.data_path, f'train_all_repair2d.csv')).read().splitlines()
    train_files = [
        {
            "image": os.path.join(args.data_path, patient.split(", ")[0]),
            "label": os.path.join(args.data_path, patient.split(", ")[1]),
        }
        for patient in train_data
    ]
    val_files = [
        {
            "image": os.path.join(args.data_path, patient.split(", ")[0]),
            "label": os.path.join(args.data_path, patient.split(", ")[1]),
        }
        for patient in val_data
    ]
    train_dataset = Dataset(data=train_files, transform=get_train_seg_transforms())
    val_dataset = Dataset(data=val_files, transform=get_val_seg_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpus,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpus,
        pin_memory=True,
        persistent_workers=True,
    )
    if args.model == 'UNet':
        print("Using model - UNet")
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )

    else:
        model = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(args.img_size, args.img_size, args.img_size),
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
            proj_type="conv",
            qkv_bias=True,
        )

    if args.model == 'GynoMRFound':
        print(f"Using {args.model} pretrained model")
        state_dict = torch.load(f"pretrained_models/GynoMRFound/pretrained_GynoMRFound_best.pth",
                                weights_only=True)
        modify_state_dict = {}
        for k, v in state_dict.items():
            if "decoder" in k:
                continue
            if "cls_token" in k:
                continue
            if "mask_token" in k:
                continue
            if "clip_mlp" in k:
                continue
            if "output_layer" in k:
                continue
            if "Tpar" in k:
                continue
            if k == 'position_embeddings':
                continue
            k = k.replace('tokenizer.', '')
            modify_state_dict[k] = v
        model.vit.load_state_dict(modify_state_dict, strict=True)
        print('Load SRPMAE pretrained model to scratch UNETR')
    elif args.model == 'UNETR':
        print(f"Using {args.model} pretrained model")
        model_path = 'pretrained_models/UNETR/UNETR_model_best_acc.pth'
        state_dict = torch.load(model_path, weights_only=True)
        modify_state_dict = {}
        for k, v in state_dict.items():
            if "out.conv.conv" in k:
                continue
            if "patch_embedding" in k:
                continue
            modify_state_dict[k] = v
        model.load_state_dict(modify_state_dict, strict=False)
        print('Load UNETR pretrained model to scratch UNETR')
    elif args.model == 'UNETRScratch':
        print('train scratch UNETR')

    print(f"Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    device = "cuda"

    model = model.to(device)
    loss_fn = DC_and_CE_loss({'batch_dice': False,
                              'smooth': 1e-5, 'do_bg': False}, {},
                             weight_ce=1,
                             weight_dice=1,
                             ignore_label=None,
                             dice_class=MemoryEfficientSoftDiceLoss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print(lr_scheduler.get_last_lr())
    path_to_save = f"{args.project}_{args.model}.pth"
    _, _ = loop_segmentation(
        epochs=args.epochs,
        model=model,
        loss_fn=loss_fn,
        opt=optimizer,
        scheduler=lr_scheduler,
        train_dl=train_loader,
        valid_dl=val_loader,
        device=device,
        path_to_save=path_to_save,
        visualizer=visualizer,
    )
