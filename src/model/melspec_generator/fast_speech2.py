from typing import List

import torch
from torch import nn
from transformers import FastSpeech2ConformerModel, FastSpeech2ConformerTokenizer


class FastSpeech2(nn.Module):
    def __init__(self):
        super(FastSpeech2, self).__init__()

        self.tokenizer = FastSpeech2ConformerTokenizer.from_pretrained(
            "espnet/fastspeech2_conformer"
        )
        self.model = FastSpeech2ConformerModel.from_pretrained(
            "espnet/fastspeech2_conformer"
        )

    def forward(self, text: List[str], **batch) -> torch.Tensor:
        """
        Args:
            text (List[str]): list of strings (B, max_len).
        Returns:
            melpsectrogram (Tensor): melspectrogram (B, n_mels, T)
        """

        tokens = self.tokenizer(text, return_tensors="pt", padding=True).to(
            self._device()
        )

        return self.model(**tokens, return_dict=True)["spectrogram"].transpose(-1, -2)

    @property
    def _device(self):
        return next(self.parameters()).device
