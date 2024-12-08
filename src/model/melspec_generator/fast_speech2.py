from typing import List

import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from torch import nn


class FastSpeech2(nn.Module):
    def __init__(self):
        super(FastSpeech2, self).__init__()

        models, _, task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech", arg_overrides={"fp16": False}
        )
        self.model = models[0]
        self.task = task

    def forward(self, text: List[str], **batch) -> torch.Tensor:
        """
        Args:
            text (List[str]): list of strings (B, max_len).
        Returns:
            melpsectrogram (Tensor): melspectrogram (B, n_mels, T)
        """

        tokens = TTSHubInterface.get_model_input(self.task, text)["net_input"].to(
            self._device
        )

        return self.model(tokens["src_tokens"], tokens["src_lengths"])[0].transpose(
            -1, -2
        )

    @property
    def _device(self):
        return next(self.parameters()).device
