from typing import List

import torch
import torchaudio
from torch import nn


class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()

        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        self.processor = bundle.get_text_processor()
        self.tacotron2 = bundle.get_tacotron2()

    @torch.inference_mode()
    def forward(self, text: List[str], **batch) -> torch.Tensor:
        """
        Args:
            text (List[str]): list of strings (B, max_len).
        Returns:
            melpsectrogram (Tensor): melspectrogram (B, n_mels, T)
        """

        processed, lengths = self.processor(text)
        processed = processed.to(self._device)
        lengths = lengths.to(self._device)
        spec, _, _ = self.tacotron2.infer(processed, lengths)

        return spec

    @property
    def _device(self):
        return next(self.parameters()).device
