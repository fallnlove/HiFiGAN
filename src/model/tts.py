import torch
from torch import Tensor, nn


class TTSModel(nn.Module):
    def __init__(self, melspec_generator, wav_generator):
        super(TTSModel, self).__init__()

        self.melspec_generator = melspec_generator
        self.wav_generator = wav_generator

    def forward(self, *args, **kwargs) -> dict[str, Tensor]:
        """
        Args:
            *args, **kwargs : features to generate melspec(can be text or wav).
        Return:
            output (dict[Tensor]): predicted wav (B, 1, T').
        """

        melspectrogram = self.melspec_generator(*args, **kwargs)

        return self.wav_generator(melspectrogram)
