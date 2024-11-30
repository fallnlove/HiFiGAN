import torch
from torch import Tensor, nn


class MPDiscriminator(nn.Module):
    def __init__(self, periods: list = [2, 3, 5, 7, 11]):
        """
        Args:
            periods (list[int]): period to reshape wav.
        """
        super(MPDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(period) for period in periods]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): wav (B, 1, T).
        Return:
            predictions (list[Tensor]): discriminators predictions (B,).
            features (list(Tensor)): discriminators features (B, 1, T/p, p).
        """
        predictions = []
        features = []

        for discriminator in self.discriminators:
            pred, feat = discriminator(x)
            predictions.append(pred)
            features += feat

        return predictions, features


class MSDiscriminator(nn.Module):
    def __init__(self):
        super(MSDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList([ScaleDiscriminator() for _ in range(3)])

        self.upsampling = nn.ModuleList(
            [
                nn.Identity(),
                nn.AvgPool1d(4, 2, 2),
                nn.AvgPool1d(4, 2, 2),
            ]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): wav (B, 1, T).
        Return:
            predictions (list[Tensor]): discriminators predictions (B,).
            features (list(Tensor)): discriminators features (B, 1, T').
        """
        predictions = []
        features = []

        for discriminator, upsampling in zip(self.discriminators, self.upsampling):
            x = upsampling(x)
            pred, feat = discriminator(x)
            predictions.append(pred)
            features += feat

        return predictions, features


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super(ScaleDiscriminator, self).__init__()

        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(1, 16, 15, 1, "same"), nn.LeakyReLU()),
                nn.Sequential(nn.Conv1d(16, 64, 41, 4, 20, groups=4), nn.LeakyReLU()),
                nn.Sequential(nn.Conv1d(64, 256, 41, 4, 20, groups=16), nn.LeakyReLU()),
                nn.Sequential(
                    nn.Conv1d(256, 1024, 41, 4, 20, groups=64), nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.Conv1d(1024, 1024, 41, 4, 20, groups=256), nn.LeakyReLU()
                ),
                nn.Sequential(nn.Conv1d(1024, 1024, 5, 1, "same"), nn.LeakyReLU()),
                nn.Conv1d(1024, 1, 3, 1, "same"),
            ]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): wav (B, 1, T).
        Return:
            prediction (Tensor): discriminator prediction (B,).
            features (list(Tensor)): discriminator features (B, 1, T').
        """

        B, _, _ = x.shape

        output = x
        features = []

        for layer in self.layers:
            output = layer(output)
            features.append(output)

        return output.view(B, -1), features


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int):
        """
        Args:
            period (int): period to reshape wav.
        """
        super(PeriodDiscriminator, self).__init__()

        self.period = period

        layers = []
        last_channels = 1

        for i in range(4):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=last_channels,
                        out_channels=int(2 ** (6 + i)),
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    ),
                    nn.LeakyReLU(),
                )
            )
            last_channels = int(2 ** (6 + i))

        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=last_channels,
                    out_channels=1024,
                    kernel_size=(5, 1),
                    padding="same",
                ),
                nn.LeakyReLU(),
            )
        )
        layers.append(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1,
                kernel_size=(3, 1),
                padding="same",
            )
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): wav (B, 1, T).
        Return:
            prediction (Tensor): discriminator prediction (B,).
            features (list(Tensor)): discriminator features (B, 1, T/p, p).
        """

        B, C, T = x.shape

        if T % self.period != 0:
            padding_size = self.period - (T % self.period)
            padding = torch.zeros((B, C, padding_size), device=x.device)
            x = torch.cat([x, padding], dim=2)

        output = x.view(B, C, -1, self.period)
        features = []

        for layer in self.layers:
            output = layer(output)
            features.append(output)

        return output.view(B, -1), features
