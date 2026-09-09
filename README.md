# SubInPhon

SubInPhon is a research codebase for learning phonological patterns with sequence-to-sequence neural networks. It supports three experiment modalities:

- `text`: UR-to-SR prediction from symbolic strings
- `feature`: UR-to-SR prediction with phonological feature embeddings
- `audio`: speech-to-text-and-spectrogram modeling from source audio

The project currently centers on generated datasets for two pattern families:

- backness harmony
- final devoicing (under-construction)

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
- `src/text_network.py`, `src/feature_network.py`: Bahdanau seq2seq models for text and feature experiments
- `src/audio_network_t1.py`: audio seq2seq model based on Google Translatotron 1
- `src/audio_network_t2.py`: audio seq2seq model based on Google Translatotron 2
- `src/text_trainer.py`, `src/audio_trainer.py`: training, evaluation, checkpoint loading, and inspection
- `src/text_recorder.py`, `src/audio_recorder.py`: accuracy/prediction logging and plot bookkeeping
- `src/utils.py`: plotting and file-saving helpers
- `misc/`: helper scripts outside the main experiment pipeline

## Installation

Use Python `3.10`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

Run commands from the repository root. Defaults are audio, T1, and `EnglishBH_shortened`.

For text and feature experiments, set `directionality = ["l2r", "r2l"]` in `src/hyper_params.py`. Run the EnglishBH configuration used in the papers with:

```bash
python3 src/main.py --modality text --lang-name EnglishBH --property "" --data-proportion 0.1
python3 src/main.py --modality feature --lang-name EnglishBH --property "" --data-proportion 0.1
```

For the other text and feature configurations, use the language, property, and sampling proportion in the table below. Add `--device cpu` to use CPU instead of CUDA.

For the audio experiment, first set `directionality = ["l2r"]` and configure `audio_root` in `src/hyper_params.py`. Supply the WAV files and TextGrids described under Data Expectations, then run:

```bash
python3 src/main.py --modality audio --lang-name EnglishBH_shortened --property "" --data-proportion 1.0
```

The audio model defaults to `t1`; `--audio-model t2` selects the other implementation. These commands use the default two runs. Set `--runs` and `--trial-num` as needed.

## Current Language Entries

The language registry currently includes:

- `EnglishBH`
- `EnglishBH_shortened`
- `EnglishBH_expanded`
- `EnglishFD`: under-construction

These entries are defined in `src/languages_config.json`.

`EnglishBH_shortened` includes the `aud_vowel` variant (`e` → `eɪ`, `o` → `oʊ`). Generation adds `ur_var` and `sr_var` for audio file references; text and feature experiments use the unchanged base strings.

## Configurations Used in the Papers

The text, feature, and audio experiments used the following settings:

| Experiment | `lang_name` | `property` | `data_proportion` | `directionality` |
| --- | --- | --- | --- | --- |
| text and feature | `EnglishBH` | `""` (none) | `0.1` | `["l2r", "r2l"]` |
| text and feature | `EnglishBH_expanded` | `""` (none) | `0.0001` | `["l2r", "r2l"]` |
| text and feature | `EnglishBH` | `"nonidentical"` | `0.2` | `["l2r", "r2l"]` |
| audio | `EnglishBH_shortened` | `""` (none) | `1.0` | `["l2r"]` |

## Run Modes and Checkpoints

- `train and evaluate`: train on the train split and evaluate on the test split each epoch.
- `tuning`: train on the train split and evaluate on the validation split each epoch.
- `inspection`: skip training and inspect existing checkpoints.

Training is followed by attention and embedding inspection. Inspection mode requires the original trial ID, matching settings, and saved checkpoints in the expected output directories.

Use `--resume-model-file PATH` to continue training from a saved model. It restores weights and the next epoch number, but not optimizer state; `--n-epochs` is the total epoch limit.

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

## Configuration and Defaults

CLI options override settings in `src/hyper_params.py`. Edit that file directly for `directionality`, `conditions`, `audio_root`, split ratios, batch size, and model dimensions.

Current defaults in `src/hyper_params.py` include:

- `lang_name = "EnglishBH_shortened"`
- `property = ""`
- `data_proportion = 1.0`
- `directionality = ["l2r", "r2l"]`
- `conditions = ["harmony", "disharmony"]`
- `run_mode = "train and evaluate"`
- `pred_log = "vowel_only_error"`
- `device = "cuda"`

Use the configurations above to reproduce the dataset settings used in the papers.

## Data Expectations

### Generated annotation files

Each run writes generated annotation files under:

```text
dataset/{trial_num}_{lang_name}[_{property}]_generated_data/
```

Here and in the output path below, `[_{property}]` is included only when the property is nonempty; the brackets are not literal. Each direction and run produces:

- `*_harmony.csv`
- `*_disharmony.csv`
- `*_template_counts.xlsx`

Text and feature experiments read `ur_string` and `sr_string` from these generated CSVs.

### Feature experiments

All three backness harmony entries use the base language name `EnglishBH` to locate the feature spreadsheet:

```text
dataset/EnglishBH_features.xlsx
```

### Audio experiments

Audio experiments expect a local audio directory shaped like:

```text
{audio_root}/{lang_name}/
```

with `<ur_var>.wav` and `<sr_var>.wav` files referenced by the generated CSVs. Use `EnglishBH_shortened` for the current audio experiment; the other registry entries do not generate the variant columns required by the audio loader.

The audio embedding inspection path also expects TextGrid segmentations under:

```text
{audio_root}/{lang_name}_segmented/
```

## Outputs

Each experiment writes to:

```text
output/{trial_num}_{lang_name}[_{property}]_{modality_suffix}/
```

The suffix is `txt` for text, `fea` for feature, and `aud` for audio.

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

The runner seeds Python, NumPy, and PyTorch, including CUDA when available.
