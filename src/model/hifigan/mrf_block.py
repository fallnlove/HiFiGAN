from typing import List

import torch
from torch import Tensor, nn


class ResBlock(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int, dilations: List, relu_constant: float
    ):
        """
        Args:
            channels (int): number input and output channels.
            kernel_size (int): kernel_size.
            dilations (ArrayLike): 1D array of kernels.
            relu_constant (float): constant for LeakyReLU.
        """
        super(ResBlock, self).__init__()

        convs = []

        for dilation in dilations:
            layer = []
            for dil in dilation:
                layer.append(nn.LeakyReLU(relu_constant))
                layer.append(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        dilation=dil,
                        padding="same",
                    ),
                )
            convs.append(nn.Sequential(*layer))

        self.convs = nn.ModuleList(convs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor (B, F, T).
        Return:
            output (Tensor): output tensor (B, F', T).
        """

        output = x

        for layer in self.convs:
            output = output + layer(output)

        return output


class MRFBlock(nn.Module):
    def __init__(
        self, channels: int, kernels: List, dilations: List, relu_constant: float
    ):
        """
        Args:
            channels (int): number input and output channels.
            kernels (ArrayLike): 1D array of kernel sizes.
            dilations (ArrayLike): 2D array of kernels.
            relu_constant (float): constant for LeakyReLU.
        """
        super(MRFBlock, self).__init__()

        self.blocks = nn.ModuleList(
            [
                ResBlock(channels, kernel_size, dilation, relu_constant)
                for kernel_size, dilation in zip(kernels, dilations)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor (B, F, T).
        Return:
            output (Tensor): output tensor (B, F', T).
        """

        output = 0

        for layer in self.blocks:
            output = output + layer(x)

        return output / len(self.blocks)
