"""
    This file provided by DLA course.
    https://github.com/markovka17/dla/tree/2024/hw3_nv
"""
import librosa
import torch
import torchaudio
from torch import nn


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sr: int = 22050,
        win_length: int = 1024,
        hop_length: int = 256,
        n_fft: int = 1024,
        f_min: int = 0,
        f_max: int = 8000,
        n_mels: int = 80,
        power: float = 1.0,
        pad_value: float = -11.5129251,
    ):
        super(MelSpectrogram, self).__init__()

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))
        self.pad_value = pad_value

    def forward(self, audio: torch.Tensor, **batch) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, 1, T]
        :return: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio.squeeze(1)).clamp_(min=1e-5).log_()

        return mel
