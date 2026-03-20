import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn as nn
from monai.networks.blocks import TransformerBlock
from monai.networks.layers import trunc_normal_
from monai.utils import ensure_tuple_rep
from monai.networks.blocks import PatchEmbeddingBlock
from utils import patchify, create_logits
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding


class MRITokenizer(nn.Module):
    def __init__(
            self,
            patch_size: Tuple[int, ...],
            img_size: Tuple[int, ...],
            hidden_size: int,
            in_channels: int = 1,
            num_heads: int = 12,
            proj_type: str = "conv",
            pos_embed_type: str = "sincos",
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        if proj_type == "conv":
            w = self.patch_embedding.patch_embeddings.weight.data
            nn.init.xavier_uniform_(w.view(w.size(0), -1))

        if pos_embed_type == "sincos":
            # disable the gradient of the position embeddings (not done in the used monai version)
            self.patch_embedding.position_embeddings.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(x)


class MAECLIP3D(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 img_size: Tuple[int, ...] = (128, 128, 128),
                 patch_size: Tuple[int, ...] = (16, 16, 16),
                 masking_ratio: float = 0.75,
                 hidden_size: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_dim: int = 1536,
                 clip_dim: int = 5120,
                 decoder_embed_dim=384,
                 decoder_mlp_dim: int = 768,
                 decoder_num_heads=12,
                 dropout_rate: float = 0.0,
                 qkv_bias: bool = True,
                 T=0.07,
                 norm_pix_loss=False):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = img_size
        self.num_patches = np.prod([img_size[i] // patch_size[i] for i in range(len(img_size))])
        patch_dim = np.prod(list(patch_size))
        self.tokenizer = MRITokenizer(patch_size=patch_size, img_size=img_size, hidden_size=hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                qkv_bias=qkv_bias,
                save_attn=False,
            )
            for i in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.masking_ratio = masking_ratio
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        trunc_normal_(self.cls_token, std=0.02)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(hidden_size, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        grid_size = []
        for in_size, pa_size in zip(img_size, patch_size):
            grid_size.append(in_size // pa_size)
        self.position_embeddings = build_sincos_position_embedding(grid_size, decoder_embed_dim, spatial_dims=3)

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                decoder_embed_dim,
                mlp_dim=decoder_mlp_dim,
                num_heads=decoder_num_heads,
                dropout_rate=dropout_rate,
                qkv_bias=qkv_bias,
                save_attn=False,
            )
            for _ in range(num_layers)])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------
        self.clip_mlp = nn.Linear(hidden_size, clip_dim, bias=True)
        self.T = 1 / T
        self.Tpar = nn.Parameter(torch.ones([]) * self.T)
        self.norm_pix_loss = norm_pix_loss
        self.CE = torch.nn.CrossEntropyLoss()

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.masking_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_img_encoder_nomask(self, x):
        x = self.tokenizer(x)
        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_encoder(self, x):
        # embed patches
        x = self.tokenizer(x)
        x, mask, ids_restore = self.random_masking(x)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # add pos embed
        x_ = x_ + self.position_embeddings
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_clip_loss(self, latent, latent_report):
        latent_img = self.clip_mlp(latent)
        latent_img = latent_img[:, 1:, :]
        latent_global = latent_img.mean(dim=1)
        logits1, logits2 = create_logits(latent_global, latent_report, self.Tpar)
        gt = torch.arange(logits1.shape[0], dtype=torch.long).to(logits1.device)
        loss_c = self.CE(logits1, gt)
        loss_c += self.CE(logits2, gt)
        return loss_c / 2.0

    def clip_loss(self, latent_global, latent_report):
        logits1, logits2 = create_logits(latent_global, latent_report, self.Tpar)
        gt = torch.arange(logits1.shape[0], dtype=torch.long).to(logits1.device)
        loss_c = self.CE(logits1, gt)
        loss_c += self.CE(logits2, gt)
        return loss_c / 2.0

    def forward_latent(self, latent):
        latent = self.clip_mlp(latent)
        latent = latent[:, 1:, :]
        latent_global = latent.mean(dim=1)
        return latent_global

    def forward_loss(self, target, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs):
        target = patchify(imgs, self.patch_size, self.image_size)
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(target, pred, mask)
        latent_global = self.forward_latent(latent)
        return loss, pred, mask, latent_global


class MAECLIP3DVIT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 img_size: Tuple[int, ...] = (128, 128, 128),
                 patch_size: Tuple[int, ...] = (16, 16, 16),
                 hidden_size: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_dim: int = 1536,
                 clip_dim: int = 5120,
                 dropout_rate: float = 0.0,
                 qkv_bias: bool = True,
                 classification: bool = True,
                 n_outputs: Optional[int] = 2,
                 pool=None,
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = img_size
        self.num_patches = np.prod([img_size[i] // patch_size[i] for i in range(len(img_size))])
        self.tokenizer = MRITokenizer(patch_size=patch_size, img_size=img_size, hidden_size=hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate / 5,
                qkv_bias=qkv_bias,
                save_attn=False,
            )
            for i in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.pool = 'mean'
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # trunc_normal_(self.cls_token, std=0.02)
        self.clip_mlp = nn.Linear(hidden_size, clip_dim, bias=True)
        self.pool = pool
        if classification:
            if n_outputs is None:
                raise ValueError("if classification mode, provide a not None n_outputs")
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.output_layer = nn.Linear(hidden_size, n_outputs)
            trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        input_tokens = self.tokenizer(x)
        # append cls token
        if hasattr(self, "cls_token"):
            cls_tokens = self.cls_token.expand(input_tokens.shape[0], -1, -1)
            input_tokens = torch.cat([cls_tokens, input_tokens], dim=1)

        if not hasattr(self, "cls_token"):
            hidden_states = []

        ## Transformer forward pass
        for blk in self.blocks:
            input_tokens = blk(input_tokens)
            if not hasattr(self, "cls_token"):
                if input_tokens.size(1) != 512:
                    hidden_states.append(
                        torch.stack(
                            [
                                input_tokens[:, (i * 512): ((i + 1) * 512), :]
                                for i in range(input_tokens.size(1) // 512)
                            ],
                            dim=1,
                        ).mean(1)
                    )
                else:
                    hidden_states.append(input_tokens)

        input_tokens = self.norm(input_tokens)

        if hasattr(self, "cls_token"):
            if self.pool == 'mean':
                input_tokens = input_tokens[:, 1:, :]
                input_tokens = input_tokens.mean(dim=1)
            else:
                input_tokens = input_tokens[:, 0]
            return self.output_layer(input_tokens)
        else:
            if input_tokens.size(1) != 512:
                return torch.stack(
                    [input_tokens[:, i * 512: (i + 1) * 512, :] for i in range(input_tokens.size(1) // 512)],
                    dim=1,
                ).mean(1), hidden_states
            else:
                return input_tokens, hidden_states

    def forward_latent(self, latent):
        latent = self.clip_mlp(latent)
        latent = latent[:, 1:, :]
        latent_global = latent.mean(dim=1)
        return latent_global


def mae_tiny(patch_size, img_size, masking_ratio, clip_dim, qkv_bias=True, norm_pix_loss=False):
    model = MAECLIP3D(
        patch_size=patch_size,
        img_size=img_size,
        hidden_size=768,
        masking_ratio=masking_ratio,
        num_layers=12,
        num_heads=12,
        mlp_dim=1536,
        qkv_bias=qkv_bias,
        norm_pix_loss=norm_pix_loss,
        clip_dim=clip_dim,
        decoder_embed_dim=384,
        decoder_mlp_dim=768,
        decoder_num_heads=12,
    )
    return model


def mae_base(patch_size, img_size, masking_ratio, clip_dim, qkv_bias=True, norm_pix_loss=False):
    model = MAECLIP3D(
        patch_size=patch_size,
        img_size=img_size,
        hidden_size=768,
        masking_ratio=masking_ratio,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        qkv_bias=qkv_bias,
        norm_pix_loss=norm_pix_loss,
        clip_dim=clip_dim,
        decoder_embed_dim=1536,
        decoder_mlp_dim=768,
        decoder_num_heads=12,
    )
    return model


def mae_large(patch_size, img_size, masking_ratio, clip_dim, qkv_bias=True, norm_pix_loss=False):
    model = MAECLIP3D(
        patch_size=patch_size,
        img_size=img_size,
        hidden_size=1024,
        masking_ratio=masking_ratio,
        num_layers=24,
        num_heads=16,
        mlp_dim=4096,
        qkv_bias=qkv_bias,
        norm_pix_loss=norm_pix_loss,
        clip_dim=clip_dim,
        decoder_embed_dim=2048,
        decoder_mlp_dim=1024,
        decoder_num_heads=12,
    )
    return model


def mae_huge(patch_size, img_size, masking_ratio, clip_dim, qkv_bias=True, norm_pix_loss=False):
    model = MAECLIP3D(
        patch_size=patch_size,
        img_size=img_size,
        hidden_size=1280,
        masking_ratio=masking_ratio,
        num_layers=32,
        num_heads=16,
        mlp_dim=5120,
        qkv_bias=qkv_bias,
        norm_pix_loss=norm_pix_loss,
        clip_dim=clip_dim,
        decoder_embed_dim=2560,
        decoder_mlp_dim=1280,
        decoder_num_heads=32,
    )
    return model


def get_mae_model(patch_size, img_size, masking_ratio, clip_dim, name='tiny'):
    if 'tiny' in name:
        model = mae_tiny(patch_size, img_size, masking_ratio, clip_dim)
    elif 'base' in name:
        model = mae_base(patch_size, img_size, masking_ratio, clip_dim)
    elif 'large' in name:
        model = mae_large(patch_size, img_size, masking_ratio, clip_dim)
    elif 'huge' in name:
        model = mae_huge(patch_size, img_size, masking_ratio, clip_dim)
    return model


def vit_tiny(n_outputs, pretrained=True):
    model = MAECLIP3DVIT(
        patch_size=(16, 16, 16),
        img_size=(128, 128, 128),
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=1536,
        qkv_bias=True,
        clip_dim=768,
        classification=True,
        n_outputs=n_outputs,
    )
    if pretrained:
        state_dict = torch.load("pretrained_models/pretrained_maeclip_bert_v1.pth", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    return model


def vit_base(patch_size, img_size, clip_dim, dropout_rate=0.0, classification=True, n_outputs=2, qkv_bias=True, ):
    model = MAECLIP3DVIT(
        patch_size=patch_size,
        img_size=img_size,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        qkv_bias=qkv_bias,
        clip_dim=clip_dim,
        classification=classification,
        n_outputs=n_outputs,
        dropout_rate=dropout_rate,
    )
    return model


def vit_large(patch_size, img_size, clip_dim, classification=True, n_outputs=2, qkv_bias=True, ):
    model = MAECLIP3DVIT(
        patch_size=patch_size,
        img_size=img_size,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        mlp_dim=4096,
        qkv_bias=qkv_bias,
        clip_dim=clip_dim,
        classification=classification,
        n_outputs=n_outputs,
    )
    return model


def vit_huge(patch_size, img_size, clip_dim, classification=True, n_outputs=2, qkv_bias=True, ):
    model = MAECLIP3DVIT(
        patch_size=patch_size,
        img_size=img_size,
        hidden_size=1280,
        num_layers=32,
        num_heads=16,
        mlp_dim=5120,
        qkv_bias=qkv_bias,
        clip_dim=clip_dim,
        classification=classification,
        n_outputs=n_outputs,
    )
    return model


# def get_mae_model(patch_size, img_size, clip_dim, name='tiny'):
#     if 'tiny' in name:
#         model = vit_tiny(patch_size, img_size, clip_dim)
#     elif 'base' in name:
#         model = vit_base(patch_size, img_size, clip_dim)
#     elif 'large' in name:
#         model = vit_large(patch_size, img_size, clip_dim)
#     elif 'huge' in name:
#         model = vit_huge(patch_size, img_size, clip_dim)
#     return model

if __name__ == '__main__':
    kk = torch.rand(6, 1, 128, 128, 128)
    yy = torch.rand(6, 5120)
    model = vit_tiny(2)
    print(model(kk)[0])
