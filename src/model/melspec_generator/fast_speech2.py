from typing import List

import nltk
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
        nltk.download("averaged_perceptron_tagger_eng")
        self.model = models[0]
        self.task = task

    def forward(self, text: List[str], **batch) -> torch.Tensor:
        """
        Args:
            text (List[str]): list of strings (B, max_len).
        Returns:
            melpsectrogram (Tensor): melspectrogram (B, n_mels, T)
        """

        tmp = [TTSHubInterface.get_model_input(self.task, i)["net_input"] for i in text]

        tokens = torch.cat([i["src_tokens"] for i in tmp], dim=0).to(self._device)
        lengths = torch.cat([i["src_lengths"] for i in tmp], dim=0).to(self._device)

        return self.model(tokens, lengths)[0].transpose(-1, -2)

    @property
    def _device(self):
        return next(self.parameters()).device
