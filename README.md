# SubInPhon

SubInPhon is a research codebase for learning phonological patterns with sequence-to-sequence neural networks.
It supports three experiment modalities:

- `text`: UR-to-SR prediction from symbolic strings
- `feature`: UR-to-SR prediction with phonological feature embeddings
- `audio`: speech-to-text-and-spectrogram modeling from source audio

The project currently centers on generated datasets for two pattern families:

- backness harmony
- final devoicing

## What The Pipeline Does

For each run, the project:

1. builds a language object from `src/languages_config.json`
2. generates harmony/disharmony annotation files under `dataset/`
3. loads one generated CSV for the selected condition
4. splits the dataset into train/validation/test subsets
5. trains or inspects a seq2seq model
6. writes checkpoints, prediction logs, accuracy files, attention plots, and embedding plots under `output/`

The same high-level loop is used across text, feature, and audio experiments, with modality-specific datasets, models, and recorders.

## Repository Layout

- `src/main.py`: CLI entry point
- `src/experiment_runner.py`: experiment orchestration for text, feature, and audio runs
- `src/hyper_params.py`: default experiment settings and output conventions
- `src/language_registry.py`: language construction from config
- `src/languages_config.json`: language inventory and syllable/word-structure specifications
- `src/language_generator.py`: language/pattern generation logic
- `src/text_dataset.py`, `src/feature_dataset.py`, `src/audio_dataset.py`: dataset loaders
- `src/text_network.py`, `src/feature_network.py`: Bahdanau seq2seq models for symbolic and feature-based experiments
- `src/audio_network_t1.py`: audio seq2seq model based on Google Translatotron 1
- `src/audio_network_t2.py`: audio seq2seq model based on Google Translatotron 2
- `src/text_trainer.py`, `src/audio_trainer.py`: training, evaluation, checkpoint loading, and inspection
- `src/text_recorder.py`, `src/audio_recorder.py`: accuracy/prediction logging and plot bookkeeping
- `src/utils.py`: plotting and file-saving helpers
- `misc/`: helper scripts outside the main experiment pipeline

## Installation

Use Python `3.8` to `3.11`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the default experiment:

```bash
python3 src/main.py
```

Run a text experiment on CPU:

```bash
python3 src/main.py --modality text --lang-name EnglishBH_shortened --runs 0:2 --device cpu
```

Run a feature-based experiment:

```bash
python3 src/main.py --modality feature --lang-name EnglishBH_shortened
```

Run an audio experiment with the second audio model:

```bash
python3 src/main.py --modality audio --audio-model t2
```

Resume from a saved checkpoint:

```bash
python3 src/main.py --modality audio --resume-model-file /path/to/model_seq2seq.pth
```

## Current Language Entries

The language registry currently includes:

- `EnglishBH`
- `EnglishBH_shortened`
- `EnglishBH_expanded`
- `EnglishFD`

These entries are defined in `src/languages_config.json`.

`EnglishBH_shortened` also includes an `aud_vowel` variant used by the audio pipeline when available.

## Run Modes

- `train and evaluate`: train on the train split, evaluate on the test split, then run attention and embedding inspection on the test split
- `tuning`: train on the train split, evaluate on the validation split, then run attention and embedding inspection on the validation split
- `inspection`: skip training and run attention/embedding inspection from saved checkpoints

## Command-Line Options

`src/main.py` supports:

- `--modality {text,feature,audio}`
- `--trial-num TRIAL_ID`
- `--runs N | START:STOP[:STEP]`
- `--data-proportion FLOAT`
- `--audio-model {t1,t2}`
- `--n-epochs INT`
- `--save-epochs INT`
- `--base-seed INT`
- `--lang-name LANGUAGE_KEY`
- `--property PROPERTY_LABEL`
- `--run-mode {"train and evaluate","tuning","inspection"}`
- `--pred-log {vowel_only_error,consonant_vowel_error,all_correct_syll}`
- `--device {cpu,cuda}`
- `--resume-model-file PATH`

Notes:

- `--runs 2` means `range(2)`, i.e. runs `0` and `1`
- `--runs 0:2` and `--runs 1:5:2` follow Python range-style semantics
- `--trial-num` defaults to a timestamp in `YYYYMMDDHHMM` format

## Settings Still Controlled In `hyper_params.py`

Some experiment settings are still edited directly in `src/hyper_params.py` rather than exposed as CLI flags:

- `directionality`
- `conditions`
- `audio_root`

You will usually also want to be aware of the current defaults for:

- `lang_name`
- `property`
- `run_mode`
- `pred_log`
- `device`
- `batch_size`
- `n_epochs`

## Data Expectations

### Generated annotation files

Each run writes generated annotation files under:

```text
dataset/{trial_num}_{lang_name}_generated_data/
```

Depending on the language and condition, these include:

- `*_harmony.csv`
- `*_disharmony.csv`
- `*_template_counts.xlsx`

Text and feature experiments read `ur_string` and `sr_string` from these generated CSVs.

### Feature experiments

Feature experiments expect a spreadsheet at:

```text
dataset/EnglishBH_features.xlsx
```

### Audio experiments

Audio experiments expect a local audio directory shaped like:

```text
{audio_root}/{lang_name}/
```

with `.wav` files referenced by the generated CSVs.

The audio embedding inspection path also expects TextGrid segmentations under:

```text
{audio_root}/{lang_name}_segmented/
```

## Outputs

Each experiment writes to:

```text
output/{trial_num}_{lang_name}_{modality_suffix}/
```

where the modality suffix is:

- `txt`
- `fea`
- `aud`

Typical outputs include:

- `run_config.txt`: resolved run arguments plus hyperparameters after CLI overrides
- `*_acc.csv`: epoch-level accuracy/loss logs
- `*_pred.csv`: prediction-level logs
- `*_model_files/`: saved checkpoints
- `*_acc_plots/`: training curves
- `*_att_plots/`: attention visualizations
- `*_embed_plots/`: phoneme embedding plots

Audio runs additionally produce:

- `*_aud_embed.csv`
- `*_aud_vowel_distance.csv`
- `*_pred_embed.png`
- `*_aud_embed.html`
- `*_aud_vowel_distance.html`

## Reproducibility

Each run is seeded with:

```text
base_seed + run_num
```

The experiment runner seeds:

- `random`
- `numpy`
- `torch`
- `torch.cuda` when CUDA is available

## Default Configuration Snapshot

At the time of writing, `src/hyper_params.py` defaults to:

- `lang_name = "EnglishBH"`
- `property = ""`
- `directionality = ["l2r", "r2l"]`
- `conditions = ["harmony", "disharmony"]`
- `run_mode = "train and evaluate"`
- `pred_log = "vowel_only_error"`
- `device = "cuda"`

These are defaults, not requirements.

## Practical Notes

- Checkpoints can now be reloaded with `map_location=hp.device`, so moving between CPU and CUDA runs is supported through the configured device setting.
- The audio pipeline chooses the `aud_vowel` variant automatically when the selected language defines one.

## Minimal Example Workflow

1. choose a language entry
2. set any remaining non-CLI settings in `src/hyper_params.py`
3. run `python3 src/main.py` with modality-specific overrides
4. inspect generated data under `dataset/`
5. inspect metrics, checkpoints, and plots under `output/`
