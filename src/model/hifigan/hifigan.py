from typing import List

import torch
from torch import Tensor, nn

from src.model.hifigan.discriminator import MPDiscriminator, MSDiscriminator
from src.model.hifigan.generator import Generator


class HiFiGAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        transposed_kernels: List,
        mrf_kernels: List,
        dilations: List,
        relu_constant: float,
    ):
        """
        Args:
            input_dim (int): number of mels in melspectrogram (n_mels).
            hidden_dim (int): hidden dimension in network.
            transposed_kernels (List): 1D list of kernel sizes.
            mrf_kernels (List): 1D list of mrf kernel sizes.
            dilations (List): 3D list of dilations.
            relu_constant (float): constant for LeakyReLU.
        """
        super(HiFiGAN, self).__init__()
        self.generator = Generator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            transposed_kernels=transposed_kernels,
            mrf_kernels=mrf_kernels,
            dilations=dilations,
            relu_constant=relu_constant,
        )

        self.mpd = MPDiscriminator(relu_constant=relu_constant)
        self.msd = MSDiscriminator(relu_constant=relu_constant)

    def forward(self, x: Tensor, **batch) -> dict[str, Tensor]:
        """
        Args:
            x (Tensor): melspectrogram (B, F, T).
        Return:
            output (dict[Tensor]): generated wav (B, 1, T').
        """

        return self.generator(x)

    def generate(self, x: Tensor, **batch) -> dict[str, Tensor]:
        """
        Args:
            x (Tensor): melspectrogram (B, F, T).
        Return:
            output (dict[Tensor]): generated wav (B, 1, T').
        """

        return self.forward(x)

    def discriminate(
        self, audio: Tensor, generated_wav: Tensor, **batch
    ) -> dict[str, Tensor]:
        """
        Args:
            audio (Tensor): ground truth wav (B, 1, T).
            generated_wav (Tensor): generated wav (B, 1, T).
        Return:
            output (dict[Tensor]): discriminator results.
        """
        generated_wav = generated_wav.detach()

        if audio.shape[2] != generated_wav.shape[2]:
            audio, generated_wav = self.pad_(audio, generated_wav)

        mpd_gt_res, mpd_gt_feat = self.mpd(audio)
        mpd_gen_res, mpd_gen_feat = self.mpd(generated_wav)

        msd_gt_res, msd_gt_feat = self.msd(audio)
        msd_gen_res, msd_gen_feat = self.msd(generated_wav)

        return {
            "gt_res": mpd_gt_res + msd_gt_res,
            "gt_feat": mpd_gt_feat + msd_gt_feat,
            "gen_res": mpd_gen_res + msd_gen_res,
            "gen_feat": mpd_gen_feat + msd_gen_feat,
        }

    def pad_(self, audio1: Tensor, audio2: Tensor) -> tuple[Tensor, Tensor]:
        """
        Pad melspectrograms to be same size.
        Args:
            audio1 (Tensor): first melspectrogram.
            audio2 (Tensor): second melspectrogram.
        Returns:
            audio1 (Tensor): padded first melspectrogram.
            audio2 (Tensor): padded second melspectrogram.
        """
        B, C, _ = audio1.shape

        if audio1.shape[2] < audio2.shape[2]:
            padding_size = audio2.shape[2] - audio1.shape[2]
            padding = torch.zeros((B, C, padding_size), device=audio1.device)

            audio1 = torch.cat([audio1, padding], dim=2)
        if audio1.shape[2] > audio2.shape[2]:
            padding_size = audio1.shape[2] - audio2.shape[2]
            padding = torch.zeros((B, C, padding_size), device=audio2.device)

            audio2 = torch.cat([audio2, padding], dim=2)

        return audio1, audio2
