import argparse
import os

import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from glob import glob
import torch
import torch.optim as optim
from torch.nn import DataParallel
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from collections import defaultdict
from model.model_MAE3DCLIP import get_mae_model
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from data.dataset_config import get_val_text_dataset, get_train_text_dataset
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
        save_path,
        data_parallel=False,
):
    metrics = defaultdict(list)
    aug = AugmentTensor([['resolution', 0.25], ['flip', 0.25], ['nosie', 0.25],
                         ['intensity', 0.25], ])
    pretrained_model = 'models--hfl--chinese-macbert-base/snapshots/a986e004d2a7f2a1c2f5a3edef4e20604a974ed1'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    bert = BertModel.from_pretrained(pretrained_model).to(device)
    for p in bert.parameters():
        p.requires_grad = False
    bert.eval()
    if data_parallel:
        model = DataParallel(model).to(device)
    for epoch in range(epochs):
        metrics["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        model.train()
        epoch_loss = 0.0
        total_len = len(train_loader)
        pbar = tqdm(total=total_len)
        for i, batch_data in enumerate(train_loader):
            inputs = batch_data['image'].to(device)
            text = []
            for lab in batch_data["label"]:
                if str(lab) != 'None':
                    text.append(lab)
            label_mask = batch_data["label_mask"].to(device)
            with torch.no_grad():
                inputs = aug(inputs).detach()
            inputs_text = tokenizer(text,
                                    padding='max_length',
                                    max_length=256,
                                    truncation=True,
                                    return_tensors="pt")
            input_ids = inputs_text['input_ids'].to(device)
            masks = inputs_text['attention_mask'].to(device)
            with torch.no_grad():
                outputs = bert(input_ids=input_ids, attention_mask=masks)
                latent_report = outputs.pooler_output
            optimizer.zero_grad()
            loss, pred, mask, latent_global = model(inputs)
            latent_global = latent_global[label_mask == 1, ...]
            if data_parallel:
                loss = torch.mean(loss)
                loss_clip = model.module.clip_loss(latent_global, latent_report)
                loss = loss * 0.9 + loss_clip * 0.1
            else:
                loss_clip = model.clip_loss(latent_global, latent_report)
                loss = loss * 0.9 + loss_clip * 0.1
            loss.backward()
            optimizer.step()
            pbar.set_description("epoch %d:loss=%0.5f" % (epoch, float(loss)))
            pbar.update(1)
            epoch_loss += loss.item()
        lr_scheduler.step()

        epoch_loss /= total_len
        print(f"Training loss: {epoch_loss:.4f}, Training learning rate: {lr_scheduler.get_last_lr()[0]:.8f}")
        vis.plot('train_loss', float(epoch_loss))

        # enter val loop on first and last epochs and every 10 epochs
        if epoch == 0 or epoch == epochs - 1 or epoch % 1 == 0:
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for i, batch_data in enumerate(tqdm(val_loader, desc="Validation...")):
                    inputs = batch_data['image'].to(device)
                    loss, pred, mask, latent_global = model(inputs)
                    if data_parallel:
                        loss = torch.mean(loss)
                    val_loss += loss.item()
                val_loss /= len(val_loader)
                print(f"Validation loss: {val_loss:.4f}")
                vis.plot('validation_loss', float(val_loss))
                if data_parallel:
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="base", help="model name to train")
    parser.add_argument("--save_path", type=str, default="pretrained_models/pretrained_maeclip_base_bert_200epoch.pth",
                        help="Path to save the model")
    parser.add_argument("--pretrained_path", type=str,
                        default='pretrained_models/pretrained_maeclip_base_bert_128.pth',
                        help="Path to save the model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--n_cpus", type=int, default=24, help="Number of cpus to use for data loading")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for training")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the model")
    parser.add_argument("--img_size", type=int, default=128, help="Image size for the model")
    parser.add_argument("--decoder_hidden_size", type=int, default=384, help="Decoder hidden size for the model")
    parser.add_argument("--decoder_num_layers", type=int, default=3, help="Number of layers for the decoder")
    parser.add_argument("--decoder_num_heads", type=int, default=12, help="Number of heads for the decoder")
    parser.add_argument(
        "--qkv_bias", type=bool, default=True, help="Whether to use bias in qkv projection for the model"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--masking_ratio", type=float, default=0.75, help="Masking ratio for the model")
    parser.add_argument("--norm_pix_loss", type=bool, default=False, help="Whether to normalize the loss or not")
    parser.add_argument("--entity", type=str, default='khtao', help="Entity for wandb logging")
    parser.add_argument("--project", type=str, default="maeclip", help="Project for wandb logging")
    parser.add_argument("--seed", type=int, default=1999, help="Seed for reproducibility")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs for the model")
    parser.add_argument("--scaling_factor", type=int, default=20, help="Scaling factor for the model")
    args = parser.parse_args()
    set_determinism(seed=args.seed)
    pixdim = (2.0, 2.0, 1.5)
    image_size = (128, 128, 128)
    vis = Visualizer(args.project)
    print(args)
    train_dataset = get_train_text_dataset()
    val_dataset = get_val_text_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpus,
        pin_memory=True,
        # persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_cpus,
        pin_memory=True,
        shuffle=True,
        # persistent_workers=True,
    )
    # image_size = train_dataset.image_size
    model = get_mae_model(patch_size=(args.patch_size, args.patch_size, args.patch_size),
                          img_size=train_dataset.image_size,
                          masking_ratio=args.masking_ratio,
                          clip_dim=768,
                          name=args.model_name,
                          )
    if args.pretrained_path is not None:
        state_dict = torch.load(args.pretrained_path, weights_only=True)
        try:
            model.load_state_dict(state_dict)
        except:
            state_dict = {k: v for k, v in state_dict.items() if "patch_embedding" not in k}
            state_dict = {k: v for k, v in state_dict.items() if "position_embeddings" not in k}
            state_dict = {k: v for k, v in state_dict.items() if "decoder_pred" not in k}
            model.load_state_dict(state_dict, strict=False)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    device = "cuda:0"
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loop_pretrain_clip(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=device,
        # vis=vis,
        save_path=args.save_path,
        data_parallel=True
    )
