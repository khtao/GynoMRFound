import torch
from collections.abc import Sequence
from monai.networks.nets import MaskedAutoEncoderViT
from utils import patchify


class MetaMAE(MaskedAutoEncoderViT):
    def __init__(self,
                 in_channels: int,
                 img_size: Sequence[int] | int,
                 patch_size: Sequence[int] | int,
                 num_classes: int,
                 hidden_size: int = 768,
                 mlp_dim: int = 512,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 masking_ratio: float = 0.75,
                 decoder_hidden_size: int = 384,
                 decoder_mlp_dim: int = 512,
                 decoder_num_layers: int = 4,
                 decoder_num_heads: int = 12,
                 proj_type: str = "conv",
                 pos_embed_type: str = "sincos",
                 decoder_pos_embed_type: str = "sincos",
                 dropout_rate: float = 0.0,
                 spatial_dims: int = 3,
                 qkv_bias: bool = False,
                 save_attn: bool = False,
                 ):
        super().__init__(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            masking_ratio=masking_ratio,
            decoder_hidden_size=decoder_hidden_size,
            decoder_mlp_dim=decoder_mlp_dim,
            decoder_num_heads=decoder_num_heads,
            decoder_num_layers=decoder_num_layers,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            decoder_pos_embed_type=decoder_pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn
        )
        self.meta_layer = torch.nn.Linear(hidden_size, num_classes)

        self.meta_layer = torch.nn.Linear(hidden_size, num_classes)
        if pos_embed_type == "sincos":
            # disable the gradient of the position embeddings (not done in the used monai version)
            self.patch_embedding.position_embeddings.requires_grad = False
        if proj_type == "conv":
            w = self.patch_embedding.patch_embeddings.weight.data
            torch.nn.init.xavier_uniform_(w.view(w.size(0), -1))

    def forward_meta(self, x):
        x = (x[:, 1:, :]).mean(dim=1)
        x = self.meta_layer(x)
        return x

    def forward(self, x, masking_ratio: float | None = None):
        target = patchify(x, self.patch_size, self.img_size)

        x = self.patch_embedding(x)
        x, selected_indices, mask = self._masking(x, masking_ratio=masking_ratio)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.blocks(x)
        meta_out = self.forward_meta(x)
        # decoder
        x = self.decoder_embed(x)

        x_ = self.mask_tokens.repeat(x.shape[0], mask.shape[1], 1)
        x_[torch.arange(x.shape[0]).unsqueeze(-1), selected_indices] = x[:, 1:, :]  # no cls token
        x_ = x_ + self.decoder_pos_embedding
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = self.decoder_blocks(x)
        x = self.decoder_pred(x)

        x = x[:, 1:, :]
        return meta_out, x, mask, target


def meta_mae_tiny(num_classes, patch_size, img_size, masking_ratio, dropout_rate=0.1):
    model = MetaMAE(
        in_channels=1,
        patch_size=patch_size,
        img_size=img_size,
        num_classes=num_classes,
        hidden_size=768,
        masking_ratio=masking_ratio,
        num_layers=12,
        num_heads=12,
        mlp_dim=1536,
        qkv_bias=True,
        decoder_mlp_dim=768,
        decoder_num_heads=12,
        decoder_hidden_size=384,
        decoder_num_layers=12,
        dropout_rate=dropout_rate,
    )
    return model


def meta_mae_base(num_classes, patch_size, img_size, masking_ratio, dropout_rate=0.1):
    model = MetaMAE(
        in_channels=1,
        patch_size=patch_size,
        img_size=img_size,
        num_classes=num_classes,
        hidden_size=768,
        masking_ratio=masking_ratio,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        qkv_bias=True,
        decoder_mlp_dim=768,
        decoder_num_heads=12,
        decoder_hidden_size=768,
        decoder_num_layers=12,
        dropout_rate=dropout_rate,
    )

    return model


def meta_mae_large(num_classes, patch_size, img_size, masking_ratio, dropout_rate=0.1):
    model = MetaMAE(
        in_channels=1,
        patch_size=patch_size,
        img_size=img_size,
        num_classes=num_classes,
        hidden_size=1024,
        masking_ratio=masking_ratio,
        num_layers=24,
        num_heads=16,
        mlp_dim=4096,
        qkv_bias=True,
        decoder_mlp_dim=1024,
        decoder_num_heads=12,
        decoder_hidden_size=2048,
        decoder_num_layers=12,
        dropout_rate=dropout_rate,
    )
    return model


def get_meta_mae_model(num_classes, patch_size, img_size, masking_ratio, name='tiny', dropout_rate=0.1):
    if 'tiny' in name:
        model = meta_mae_tiny(num_classes, patch_size, img_size, masking_ratio, dropout_rate)
    elif 'base' in name:
        model = meta_mae_base(num_classes, patch_size, img_size, masking_ratio, dropout_rate)
    elif 'large' in name:
        model = meta_mae_large(num_classes, patch_size, img_size, masking_ratio, dropout_rate)
    else:
        model = None
    return model


if __name__ == '__main__':
    kk = torch.rand(6, 1, 128, 128, 128)
    yy = torch.rand(6, 5120)
