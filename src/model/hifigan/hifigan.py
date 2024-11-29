import torch
from matplotlib.pylab import ArrayLike
from torch import Tensor, nn

from src.model.hifigan.discriminator import MPDiscriminator, MSDiscriminator
from src.model.hifigan.generator import Generator


class HiFiGAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        transposed_kernels: ArrayLike,
        mrf_kernels: ArrayLike,
        dilations: ArrayLike,
    ):
        super(HiFiGAN, self).__init__()
        self.generator = Generator(
            input_dim,
            hidden_dim,
            transposed_kernels,
            mrf_kernels,
            dilations,
        )

        self.mpd = MPDiscriminator()
        self.msd = MSDiscriminator()

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
