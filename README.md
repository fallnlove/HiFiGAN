# HiFiGAN implementation

## Installation

1. Install dependencies

```bash
pip install -r ./requirements.txt
```

2. Download checkpoints and pre-trained model

```bash
python3 scripts/download_model.py
```

## Training

If you want to reproduce training process run following command.

```bash
pyhton3 train.py -cn=train
```

## Inference

There are two ways to inference HiFiGAN on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) Dataset.

1. Generate wav file from melspectrogram of the ground truth wav.

```bash
pyhton3 synthesize.py -cn=lj_speech_synthesize
```

2. Generate wav file from text using FastSpeech2 model.

```bash
pyhton3 synthesize.py -cn=lj_text_synthesize
```

> [!NOTE]
> Generated audio saved in `data/saved` dir. To customize save path use `inferencer.save_path` option.

## Custom dataset inference

Run following commands to inference HiFiGAN with FastSpeech2 on text data.

1. Inference model in custom dataset.

```bash
python3 synthesize.py -cn=text_synthesize \
datasets.inference.data_dir=PATH_TO_DATASET
```

2. Inference model with text query via command-line.

```bash
python3 synthesize.py -cn=text_synthesize \
datasets.inference.text=YOUR_TEXT
```

Run following command to inference HiFiGAN using ground truth wav's.

```bash
python3 synthesize.py -cn=audio_synthesize \
datasets.inference.data_dir=PATH_TO_DATASET
```

Note: `PATH_TO_DATASET` should contains audio files in `wav` format.

## Metrics

Run following command to calculate WV-MOS of generated audio.

```bash
python3 scripts/wv_mos.py +data_dir=PATH_TO_GEN_AUDIO
```

## Credits

I use [Project Template](https://github.com/Blinorot/pytorch_project_template) for well-structured code.
For TTS inference I use FastSpeech2Conformer pre-trained model from [HuggingFace](https://huggingface.co/docs/transformers/model_doc/fastspeech2_conformer#-transformers-usage).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
