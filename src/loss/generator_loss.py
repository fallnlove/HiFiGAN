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
        gt_feat: Tensor,
        gen_res: Tensor,
        gen_feat: Tensor,
        generated_wav: Tensor,
        gt_melspectrogram: Tensor,
        **batch
    ):
        """
        Generator loss

        Args:
            gt_res (Tensor): discriminator output for ground truth.
            gen_res (Tensor): discriminator output for generated.
        Returns:
            loss (dict): dict containing calculated loss discriminator loss.
        """
        gen_melspectrogram = self.melspec(generated_wav.squeeze(1))

        loss_mel = self.l1(gt_melspectrogram, gen_melspectrogram)

        loss_adv = 0
        for gen in gen_res:
            loss_adv += torch.mean((gen - 1) ** 2)

        loss_fm = 0

        for gt, gen in zip(gt_feat, gen_feat):
            loss_fm += torch.mean(torch.abs(gt - gen))

        loss_gen = loss_adv + self.l_fm * loss_fm + self.l_mel * loss_mel

        return {"loss_gen": loss_gen, "gen_melspectrogram": gen_melspectrogram}
