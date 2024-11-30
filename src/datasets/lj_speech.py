import json
import os
import random
import shutil
from pathlib import Path

import torchaudio
import wget
from torch import nn
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "data": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
}


class LJspeechDataset(BaseDataset):
    def __init__(self, part: str, data_dir: str = None, *args, **kwargs):
        """
        Args:
            part (str): part of dataset(train or validation).
            data_dir (str): path to dataset.
        """
        assert part in ["train", "val"]

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        else:
            data_dir = Path(data_dir).absolute().resolve()
        self._data_dir = data_dir

        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        if not arch_path.exists():
            print("Loading dataset")
            wget.download(URL_LINKS["data"], str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"

        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)

        return index

    def _create_index(self, part):
        random.seed(42)

        index = []
        wavs_dir = self._data_dir / "wavs"
        if not wavs_dir.exists():
            self._load_dataset()

        flac_dirs = set()
        for i, filename in enumerate(wavs_dir.iterdir()):
            if part == "train" and i < 2000:
                continue
            if part == "val" and i >= 2000:
                continue
            if str(filename).endswith(".wav"):
                flac_dirs.add(str(filename))

        for wav_file in tqdm(
            list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            t_info = torchaudio.info(wav_file)
            length = t_info.num_frames / t_info.sample_rate
            index.append(
                {
                    "wav_path": wav_file,
                    "audio_len": length,
                }
            )
        return index
