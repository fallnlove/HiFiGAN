import torch
from torch import Tensor, nn


class DiscriminatorLoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, gt_res: torch.Tensor, gen_res: torch.Tensor, **batch):
        """
        Discriminator loss

        Args:
            gt_res (Tensor): discriminator output for ground truth.
            gen_res (Tensor): discriminator output for generated.
        Returns:
            loss (dict): dict containing calculated loss discriminator loss.
        """
        loss_disc = 0
        for gt, gen in zip(gt_res, gen_res):
            loss_disc += torch.mean((gt - 1) ** 2) + torch.mean(gen**2)

        return {"loss_disc": loss_disc}
