# Author: Jonathan Prexl
# Adapted from:
# https://github.com/IcarusWizard/MAE/

import torch
import timm
import numpy as np
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from .maskingfunction import take_indexes, PatchShuffle, DummyShuffle


class SensorParameterEncoding_Responsefunction(torch.nn.Module):
    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            *[
                nn.Linear(2301, emb_dim * 2),
                nn.ReLU(),
                nn.Linear(emb_dim * 2, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
            ]
        )

    def forward(self, x):
        return self.network(x)


class SensorParameterEncoding_GSD(torch.nn.Module):
    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    *[
                        nn.Linear(1, emb_dim // 2),
                        nn.ReLU(),
                        nn.Linear(emb_dim // 2, emb_dim),
                        nn.ReLU(),
                        nn.Linear(emb_dim, 2 * emb_dim),
                        nn.ReLU(),
                        nn.Linear(2 * emb_dim, emb_dim),
                        nn.ReLU(),
                        nn.Linear(emb_dim, emb_dim),
                    ]
                )
            ]
        )

    def forward(self, x):
        return self.network(x)


class Encoder(torch.nn.Module):
    """
    Encoder class for processing images into patch tokens.

    Args:
        image_size (int): Size of the input image (assumed to be square).
        patch_size (int): Size of each patch (assumed to be square).
        emb_dim (int): Dimension of the embedding space.
        num_layer (int): Number of transformer layers.
        num_head (int): Number of attention heads in the transformer.
        num_channels (int): Number of channels in the input image.
        channels_seperate_tokens (bool): Whether to treat channels as separate tokens.
        positional_embedding_3D (bool): Whether to use 3D positional embeddings.
        sensor_parameter_embedding_active (bool): Whether to use sensor parameter embeddings.
        maskingfunction (callable): Function to apply masking on all imput patches.
    """

    def __init__(
        self,
        image_size,
        patch_size,
        emb_dim,
        num_layer,
        num_head,
        num_channels,
        channels_seperate_tokens,
        positional_embedding_3D,
        sensor_parameter_embedding_active,
        maskingfunction,
    ) -> None:
        super().__init__()

        self.positional_embedding_3D = positional_embedding_3D
        self.channels_seperate_tokens = channels_seperate_tokens
        self.num_patches = (
            (image_size // patch_size) ** 2
        )  # patches along the x and y direction (not including the channel dimension)
        self.sqrt_num_patches = int(np.sqrt(self.num_patches))
        self.sensor_parameter_embedding_active = sensor_parameter_embedding_active
        self.num_channels = num_channels
        self.img_size = image_size
        self.patch_size = patch_size

        # Tests for the input parameters
        if self.positional_embedding_3D == True:
            assert sensor_parameter_embedding_active == False, "Does not make sense"
        if self.channels_seperate_tokens == False:
            assert sensor_parameter_embedding_active == False, "Does not make sense"
        assert image_size % patch_size == 0

        # Setup Network
        if channels_seperate_tokens:
            self.linEmmbedingLayer = nn.Linear(patch_size**2, emb_dim)

            self.patchify = Rearrange(
                "b c (h1 h) (w1 w) -> b c (h1 w1) (h w)",
                h1=self.sqrt_num_patches,
                w1=self.sqrt_num_patches,
            )

            if positional_embedding_3D == False:
                self.pos_embedding = torch.nn.Parameter(
                    torch.zeros(1, 1, self.num_patches, emb_dim)
                )
            else:
                self.pos_embedding = torch.nn.Parameter(
                    torch.zeros(1, self.num_channels, self.num_patches, emb_dim)
                )

        else:
            raise NotImplementedError("Not properly tested yet")
            self.linEmmbedingLayer = nn.Linear(num_channels * patch_size**2, emb_dim)
            self.patchify = Rearrange(
                "b c (h h1) (w w1) -> b (h1 w1) (h w c)",
                h1=self.sqrt_num_patches,
                w1=self.sqrt_num_patches,
            )
            self.pos_embedding = torch.nn.Parameter(
                torch.zeros(1, self.num_patches, emb_dim)
            )

        # adaption for sensor parameter encoding
        if self.sensor_parameter_embedding_active:
            self.linEmmbedingLayerResponsefunction = (
                SensorParameterEncoding_Responsefunction(emb_dim)
            )
            self.linEmmbedingLayerGSD = SensorParameterEncoding_GSD(emb_dim)

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.shuffle = maskingfunction

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, img, rf=None, gsd=None):
        assert self.img_size == img.shape[2]

        # ---------------------------------------------------------------------------
        # patchify image and make tokens though linear embeddings

        patches = self.patchify(img)

        patches = self.linEmmbedingLayer(patches)

        # ---------------------------------------------------------------------------
        # apply positional encoding and sensor parameter encoding

        patches = patches + self.pos_embedding

        if self.sensor_parameter_embedding_active:
            patches = (
                patches + self.linEmmbedingLayerResponsefunction(rf)[:, :, None, :]
            )
            patches = (
                patches + self.linEmmbedingLayerGSD(gsd[:, :, None])[:, :, None, :]
            )

        if self.channels_seperate_tokens:
            patches = rearrange(patches, "b c t e -> b (c t) e")

        patches = rearrange(patches, "b T e -> T b e")

        # ---------------------------------------------------------------------------
        # masking of patches

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        # ---------------------------------------------------------------------------
        # encoding step

        patches = rearrange(patches, "T b e -> b T e")

        features = self.layer_norm(self.transformer(patches))

        features = rearrange(features, "b T e -> T b e")

        return features, backward_indexes


class Decoder(torch.nn.Module):
    """
    Decoder module for transforming encoded features back into image space.

    Args:
        image_size (int): The size of the input image.
        patch_size (int): The size of each patch.
        emb_dim (int): The dimension of the embedding.
        num_layer (int): The number of transformer layers.
        num_head (int): The number of attention heads in the transformer.
        num_channels (int): The number of channels in the input image.
        channels_seperate_tokens (bool): Whether to use separate tokens for each channel.
        positional_embedding_3D (bool): Whether to use 3D positional embeddings (depending on the ).
        sensor_parameter_embedding_active (bool): Whether to use sensor parameter encoding.
    """

    def __init__(
        self,
        image_size,
        patch_size,
        emb_dim,
        num_layer,
        num_head,
        num_channels,
        channels_seperate_tokens,
        positional_embedding_3D,
        sensor_parameter_embedding_active,
    ) -> None:
        super().__init__()

        self.positional_embedding_3D = positional_embedding_3D
        if self.positional_embedding_3D == True:
            assert sensor_parameter_embedding_active == False, "Does not make sense"

        assert not (sensor_parameter_embedding_active and positional_embedding_3D)
        assert any([sensor_parameter_embedding_active, positional_embedding_3D])

        self.sensor_parameter_embedding_active = sensor_parameter_embedding_active
        self.num_patches = (
            image_size // patch_size
        ) ** 2  # patches along the x and y direction EX channels
        self.sqrt_num_patches = int(np.sqrt(self.num_patches))
        self.num_channels = num_channels
        self.patch_size = patch_size

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))

        if channels_seperate_tokens:
            if positional_embedding_3D == False:
                self.pos_embedding = torch.nn.Parameter(
                    torch.zeros(1, 1, self.num_patches, emb_dim)
                )
            else:
                self.pos_embedding = torch.nn.Parameter(
                    torch.zeros(1, self.num_channels * self.num_patches, emb_dim)
                )

            self.head = torch.nn.Linear(emb_dim, patch_size**2)
            self.patch2img = Rearrange(
                "(c h w) b (p1 p2) -> b c (h p1) (w p2)",
                p1=self.patch_size,
                p2=self.patch_size,
                h=self.sqrt_num_patches,
                w=self.sqrt_num_patches,
                c=num_channels,
            )
        else:
            self.pos_embedding = torch.nn.Parameter(
                torch.zeros(1, self.num_patches, emb_dim)
            )
            self.head = torch.nn.Linear(emb_dim, num_channels * patch_size**2)
            self.patch2img = Rearrange(
                "(h w) b (c p1 p2) -> b c (h p1) (w p2)",
                p1=self.patch_size,
                p2=self.patch_size,
                h=self.sqrt_num_patches,
                w=self.sqrt_num_patches,
                c=num_channels,
            )

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        # adaption for hyper parameter infusion
        if self.sensor_parameter_embedding_active:
            self.linEmmbedingLayerResponsefunction = (
                SensorParameterEncoding_Responsefunction(emb_dim)
            )
            self.linEmmbedingLayerGSD = SensorParameterEncoding_GSD(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, features, rf, gsd, backward_indexes):
        T = features.shape[0]

        features = torch.cat(
            [
                features,
                self.mask_token.expand(
                    backward_indexes.shape[0] - features.shape[0], features.shape[1], -1
                ),
            ],
            dim=0,
        )

        # ---------------------------------------------------------------------------
        # unshuffle features

        features = take_indexes(features, backward_indexes)

        # ---------------------------------------------------------------------------
        # positional and parameter encoding

        if self.sensor_parameter_embedding_active:
            features = rearrange(features, "(c t) b e -> b c t e", c=self.num_channels)

            # add spectral and gsd embedding plus pos embedding
            features = features + self.pos_embedding
            features = (
                features + self.linEmmbedingLayerResponsefunction(rf)[:, :, None, :]
            )
            features = (
                features + self.linEmmbedingLayerGSD(gsd[:, :, None])[:, :, None, :]
            )

            features = rearrange(features, "b c t e -> b (c t) e")

        else:
            features = rearrange(features, "T b e -> b T e")
            features = features + self.pos_embedding

        # ---------------------------------------------------------------------------
        # decoding step

        features = self.transformer(features)

        # ---------------------------------------------------------------------------
        # deconvolve into image space

        features = rearrange(features, "b T e -> T b e")

        patches = self.head(features)

        mask = torch.zeros_like(patches)

        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes)

        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask


class SenPaMAE(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img, rf=None, gsd=None):
        features, backward_indexes = self.encoder(img, rf, gsd)
        predicted_img, mask = self.decoder(features, rf, gsd, backward_indexes)

        return predicted_img, mask, len(features)


def vit_base_patch16(
    image_size=144,
    patch_size=16,
    emb_dim=768,
    num_layer=12,
    num_head=12,
    num_channels=4,
    channels_seperate_tokens=True,
    positional_embedding_3D=False,
    sensor_parameter_embedding_active=True,
):
    model = Encoder(
        image_size=image_size,
        patch_size=patch_size,
        emb_dim=emb_dim,
        num_layer=num_layer,
        num_head=num_head,
        num_channels=num_channels,
        channels_seperate_tokens=channels_seperate_tokens,
        positional_embedding_3D=positional_embedding_3D,
        sensor_parameter_embedding_active=sensor_parameter_embedding_active,
        maskingfunction=DummyShuffle(),
    )
    return model


# def vit_base_patch16_doubleSPE(**kwargs):
#     model = Encoder(
#         image_size=144,
#         patch_size=16,
#         emb_dim=768,
#         num_layer=12,
#         num_head=12,
#         num_channels=4,
#         channels_seperate_tokens=True,
#         positional_embedding_3D=False,
#         sensor_parameter_embedding_active=True,
#         maskingfunction=DummyShuffle(masking_ratio=0, maskingStrategy="random"),
#         **kwargs
#     )
#     return model
if __name__ == "__main__":
    from torchinfo import summary
    import hydra
    import omegaconf

    config = "/home/user/src/configs/basemae.yaml"
    cfg = omegaconf.OmegaConf.load(config)

    cfg.model.encoder.maskingfunction.masking_ratio = 0
    cfg.dataset.numChannels = 4
    model = hydra.utils.instantiate(cfg.model)

    batchsize = 16
    out = model(
        torch.zeros(batchsize, cfg.dataset.numChannels, 144, 144),
        torch.zeros(batchsize, cfg.dataset.numChannels, 2301),
        torch.zeros(batchsize, cfg.dataset.numChannels),
    )

    print(
        model.encoder.positional_embedding_3D,
        model.encoder.sensor_parameter_embedding_active,
    )
    print(
        model.decoder.positional_embedding_3D,
        model.decoder.sensor_parameter_embedding_active,
    )
    print(
        "pos embedding shape\n",
        model.encoder.pos_embedding.shape,
        "\n",
        model.decoder.pos_embedding.shape,
    )
