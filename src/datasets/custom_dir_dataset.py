import json
import logging
import random
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomTextDataset(BaseDataset):
    def __init__(
        self,
        data_dir: Optional[str],
        text: Optional[str],
        *args,
        **kwargs,
    ):
        """
        Custom dir dataset.
        Args:
            data_dir (Optional[str]): path to dataset.
            text (Optional[str]): text to synthesize.
        """
        assert (
            data_dir is not None or text is not None
        ), "You should provide path to dataset or text to synthesize"

        if data_dir is not None:
            self._data_dir = Path(data_dir).absolute().resolve()
            index = self._get_or_load_index()
        else:
            index = [
                {
                    "path": "custom_text",
                    "text": text,
                }
            ]

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self):
        index_path = self._data_dir / "index.json"

        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)

        return index

    def _create_index(self):
        random.seed(42)

        index = []
        text_dir = self._data_dir / "transcriptions"
        assert text_dir.exists(), "Dataset is not exits"

        text_paths = set()
        for filename in text_dir.iterdir():
            if str(filename).endswith(".txt"):
                text_paths.add(str(filename))

        for file in tqdm(list(text_paths), desc="Preparing custom dir folders"):
            with open(file) as f:
                text = f.read()
            index.append(
                {
                    "path": file,
                    "text": text,
                }
            )
        return index


class CustomAudioDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        *args,
        **kwargs,
    ):
        """
        Custom dir dataset.
        Args:
            data_dir (str): path to dataset.
        """

        self._data_dir = Path(data_dir).absolute().resolve()

        index = self._get_or_load_index()

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self):
        index_path = self._data_dir / "index.json"

        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)

        return index

    def _create_index(self):
        random.seed(42)

        index = []
        wav_dir = self._data_dir
        assert wav_dir.exists(), "Dataset is not exits"

        wav_paths = set()
        for filename in wav_dir.iterdir():
            if str(filename).endswith(".wav"):
                wav_paths.add(str(filename))

        for file in tqdm(list(wav_paths), desc="Preparing custom dir folders"):
            index.append(
                {
                    "path": file,
                    "text": "",
                }
            )
        return index
