import torch
from matplotlib.pylab import ArrayLike
from torch import Tensor, nn

from src.model.hifigan.mrf_block import MRFBlock
from src.model.melspec_generator import MelSpectrogram


class Generator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        transposed_kernels: ArrayLike,
        mrf_kernels: ArrayLike,
        dilations: ArrayLike,
        *args,
        **kwargs,
    ):
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
                    nn.PReLU(),
                    nn.ConvTranspose1d(
                        in_channels=hidden_dim // (2 ** (i)),
                        out_channels=hidden_dim // (2 ** (i + 1)),
                        kernel_size=transposed_kernels[i],
                        stride=transposed_kernels[i] // 2,
                        padding=transposed_kernels[i] // 4,
                    ),
                    MRFBlock(hidden_dim // (2 ** (i + 1)), mrf_kernels, dilations[i]),
                )
            )

        self.body = nn.Sequential(*body)

        self.out_processing = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(
                in_channels=hidden_dim // (2 ** len(transposed_kernels)),
                out_channels=1,
                kernel_size=1,
                padding="same",
            ),
            nn.Tanh(),
        )

        self.melspec = MelSpectrogram(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): melspectrogram (B, F, T).
        Return:
            output (dict[Tensor]): generated wav (B, 1, T').
        """

        output = self.preprocess(x)
        output = self.body(output)
        output = self.out_processing(output)

        return {
            "generated_wav": output,
            "gt_melspectrogram": x,
            "gen_melspectrogram": self.melspec(output.squeeze(1)),
        }
