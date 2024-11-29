import torch
from torch import Tensor, nn

from src.model.melspec_generator import MelSpectrogram


class GeneratorLoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self, l_fm: float, l_mel: float, *args, **kwargs):
        super(GeneratorLoss, self).__init__()
        self.l_fm = l_fm
        self.l_mel = l_mel
        self.l1 = nn.L1Loss()
        self.melspec = MelSpectrogram(*args, **kwargs)

    def forward(
        self,
        gen_res: Tensor,
        gt_feat: Tensor,
        gen_feat: Tensor,
        gt_melspectrogram: Tensor,
        gen_melspectrogram: Tensor,
        **batch
    ):
        """
        Generator loss

        Args:
            gen_res (Tensor): discriminator output for generated.
            gt_feat (Tensor): discriminator features of ground truth.
            gen_feat (Tensor): discriminator features of generated.
            gt_melspectrogram (Tensor): melspectrogram of ground truth wav.
            gen_melspectrogram (Tensor): melspectrogram of generated wav.
        Returns:
            loss (dict): dict containing calculated loss generator loss.
        """

        loss_mel = self.l1(gt_melspectrogram, gen_melspectrogram)

        loss_adv = 0
        for gen in gen_res:
            loss_adv += torch.mean((gen - 1) ** 2)

        loss_fm = 0

        for gt, gen in zip(gt_feat, gen_feat):
            loss_fm += torch.mean(torch.abs(gt - gen))

        loss_gen = loss_adv + self.l_fm * loss_fm + self.l_mel * loss_mel

        return {"loss_gen": loss_gen}
