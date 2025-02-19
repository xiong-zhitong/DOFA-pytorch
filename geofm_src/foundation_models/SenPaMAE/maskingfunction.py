# Author: Jonathan Prexl
# Adapted from:
# https://github.com/IcarusWizard/MAE/

import torch
import torch.nn as nn
import numpy as np
from einops import repeat


def take_indexes(sequences, indexes):
    return torch.gather(
        sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1])
    )


class PatchShuffle(torch.nn.Module):
    def __init__(
        self, image_size, num_channels, patch_size, masking_ratio, maskingStategy
    ) -> None:
        super().__init__()

        assert image_size % patch_size == 0, "img_size%patch_size must be zero"

        self.num_patches = (
            image_size // patch_size
        ) ** 2  # patches along the x and y direction EX channels
        self.num_channels = num_channels
        self.num_tokens = self.num_patches * self.num_channels
        self.masking_ratio = masking_ratio
        self.maskingStategy = maskingStategy

        assert self.maskingStategy in ["random"]

        self.numPatchesToMask = int(self.num_patches / 100 * self.masking_ratio)

    def random_indices(self):
        random_channels_list = torch.randperm(self.num_channels).numpy()

        if self.maskingStategy == "random":
            potentialTokens = np.arange(self.num_tokens)
            potentialTokens = torch.from_numpy(potentialTokens)
            rper = torch.randperm(potentialTokens.size(0))
            potentialTokens = potentialTokens[rper]
            potentialTokens = potentialTokens.numpy()
            token_pos = potentialTokens[: self.numPatchesToMask].tolist()

        else:
            raise ValueError()

        # should not have duplicates
        assert len(token_pos) == len(set(token_pos))

        # save how many we have masked to cut the indices later
        self.ammount_masked_tokens = len(token_pos)

        rest = np.delete(np.arange(self.num_tokens), token_pos)
        forward_indexes = np.concatenate([rest, token_pos])
        backward_indexes = np.argsort(forward_indexes)

        return forward_indexes, backward_indexes

    def forward(self, patches: torch.Tensor):
        iT, iB, iC = patches.shape
        indexes = [self.random_indices() for _ in range(iB)]
        forward_indexes = torch.as_tensor(
            np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)
        backward_indexes = torch.as_tensor(
            np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)
        patches = take_indexes(patches, forward_indexes)
        patches = patches[: -self.ammount_masked_tokens]
        return patches, forward_indexes, backward_indexes


class DummyShuffle(torch.nn.Module):
    def __init__(
        self,
        image_size=None,
        num_channels=None,
        patch_size=None,
        masking_ratio=None,
        maskingStategy=None,
    ) -> None:
        super().__init__()

    def forward(self, patches: torch.Tensor):
        return patches, torch.zeros(1), torch.zeros(1)


if __name__ == "__main__":
    PS = PatchShuffle(128, 10, 16, 75, "random")
    patches = torch.zeros((640, 2, 400))
    patches, forward_indexes, backward_indexes = PS(patches)
    print(len(patches))
