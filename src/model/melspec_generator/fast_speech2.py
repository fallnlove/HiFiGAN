import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from torch import nn


class FastSpeech2(nn.Module):
    def __init__(self):
        super(FastSpeech2, self).__init__()

        fast_speech2, _, _ = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech", arg_overrides={"fp16": False}
        )
        self.fast_speech2 = fast_speech2

    def forward(
        self, tokens: torch.Tensor, lengths: torch.Tensor, **batch
    ) -> torch.Tensor:
        """
        Args:
            tokens (Tensor): tokens of text (B, len).
            lengths (Tensor): lengths of text (B,).
        Returns:
            melpsectrogram (Tensor): melspectrogram (B, n_mels, T)
        """

        return self.fast_speech2(tokens, lengths)[0].T
