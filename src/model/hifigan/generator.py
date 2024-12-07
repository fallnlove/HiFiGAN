from typing import List

import torch
from torch import Tensor, nn

from src.model.hifigan.mrf_block import MRFBlock
from src.model.melspec_generator import MelSpectrogram


class Generator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        transposed_kernels: List,
        mrf_kernels: List,
        dilations: List,
        relu_constant: float,
        *args,
        **kwargs,
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
        super(Generator, self).__init__()
        self.preprocess = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=7,
            dilation=1,
            padding="same",
        )

        body = []
        for i in range(len(transposed_kernels)):
            body.append(
                nn.Sequential(
                    nn.LeakyReLU(relu_constant),
                    nn.ConvTranspose1d(
                        in_channels=hidden_dim // (2 ** (i)),
                        out_channels=hidden_dim // (2 ** (i + 1)),
                        kernel_size=transposed_kernels[i],
                        stride=transposed_kernels[i] // 2,
                        padding=transposed_kernels[i] // 4,
                    ),
                    MRFBlock(
                        hidden_dim // (2 ** (i + 1)),
                        mrf_kernels,
                        dilations,
                        relu_constant,
                    ),
                )
            )

        self.body = nn.Sequential(*body)

        self.out_processing = nn.Sequential(
            nn.LeakyReLU(relu_constant),
            nn.Conv1d(
                in_channels=hidden_dim // (2 ** len(transposed_kernels)),
                out_channels=1,
                kernel_size=1,
                padding="same",
            ),
            nn.Tanh(),
        )

        self.melspec = MelSpectrogram(*args, **kwargs)
        self.transposed_kernels = transposed_kernels

    def forward(self, gt_melspectrogram: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): melspectrogram (B, F, T).
        Return:
            output (dict[Tensor]): generated wav (B, 1, T').
        """

        output = self.preprocess(gt_melspectrogram)
        output = self.body(output)
        output = self.out_processing(output)

        gen_melspectrogram = self.melspec(output.squeeze(1))
        if gt_melspectrogram.shape[2] != gen_melspectrogram.shape[2]:
            gt_melspectrogram, gen_melspectrogram = self.pad_(
                gt_melspectrogram, gen_melspectrogram
            )

        return {
            "generated_wav": output,
            "gt_melspectrogram": gt_melspectrogram,
            "gen_melspectrogram": gen_melspectrogram,
        }

    def pad_(self, mel1: Tensor, mel2: Tensor) -> tuple[Tensor, Tensor]:
        """
        Pad melspectrograms to be the same size.
        Args:
            mel1 (Tensor): first melspectrogram.
            mel2 (Tensor): second melspectrogram.
        Returns:
            mel1 (Tensor): padded first melspectrogram.
            mel2 (Tensor): padded second melspectrogram.
        """
        B, C, _ = mel1.shape

        if mel1.shape[2] < mel2.shape[2]:
            padding_size = mel2.shape[2] - mel1.shape[2]
            padding = torch.zeros((B, C, padding_size), device=mel1.device).fill_(
                self.melspec.pad_value
            )

            mel1 = torch.cat([mel1, padding], dim=2)
        if mel1.shape[2] > mel2.shape[2]:
            padding_size = mel1.shape[2] - mel2.shape[2]
            padding = torch.zeros((B, C, padding_size), device=mel2.device).fill_(
                self.melspec.pad_value
            )

            mel2 = torch.cat([mel2, padding], dim=2)

        return mel1, mel2

    def length_(self, lengths: Tensor) -> Tensor:
        """
        Return wav lengths after HiFiGAN processing.
        Args:
            lengths (Tensor): input lengths.
        Returns:
            output_lengths (Tensor): output lengths.
        """

        output_lengths = lengths

        for kernel_size in self.transposed_kernels:
            output_lengths = (
                (output_lengths - 1) * (kernel_size // 2)
                - 2 * (kernel_size // 4)
                + kernel_size
            )

        return output_lengths
