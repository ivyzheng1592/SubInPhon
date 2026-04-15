# SubInPhon

SubInPhon is a research codebase for modeling phonological patterns with sequence-to-sequence neural networks.
It supports text-based learning (UR -> SR), feature-embedded text learning, and audio-to-text/audio modeling.

## Project Layout
- `main.py` : entry point for running experiments
- `hyper_params.py` : global hyperparameters and output layout
- `text_dataset.py`, `feature_dataset.py`, `audio_dataset.py` : dataset loaders
- `text_network.py`, `feature_network.py` : text/feature seq2seq models
- `audio_network_t1.py` : audio seq2seq model (Translatotron-style)
- `text_trainer.py`, `audio_trainer.py` : training/evaluation loops
- `text_recorder.py`, `audio_recorder.py` : metrics + plot recording
- `utils.py` : plotting utilities
- `Dataset/` : data files and language definitions
- `Results/` : model outputs, plots, and logs

## Quick Start
1. Create/activate a Python environment (3.8–3.11).
2. Install requirements:

```bash
pip install torch==2.0.0 torchaudio==2.0.1 torchinfo==1.8.0
```

3. Run a default experiment:

```bash
python main.py
```

4. Override settings from the command line when needed:

```bash
python main.py --modality text --lang-name EnglishBH_shortened --runs 0:2 --device cpu
```

5. Set `property`, `directionality`, `conditions`, and `audio_root` in `hyper_params.py` before running:

```bash
python main.py --modality audio
```

## Run Modes
- `train and evaluate`: train on train set; evaluate on test set; then attention + embedding on test
- `tuning`: train on train set; evaluate on valid set; then attention + embedding on valid
- `evaluate only`: attention + embedding on test (no training)

## CLI Options
- `--modality {text,feature,audio}` selects the experiment type
- `--lang-name` overrides `hp.lang_name`
- `--runs` accepts a count (`2`) or range-style spec (`0:2`, `1:5:2`)
- `--trial-num` sets the output folder prefix; default is a timestamp
- `--run-mode` overrides the run mode without editing `hyper_params.py`
- `--pred-log {vowel_only_error,consonant_vowel_error,all_correct_syll}` overrides `hp.pred_log` for a single run
- `--device {cpu,cuda}` overrides `hp.device` for a single run
- `--resume-model-file` resumes from a saved `*_seq2seq.pth` checkpoint

## Typical Experiment Flow
1. Choose a language entry, optional property, and directionality (e.g., `EnglishBH_shortened` + `""` + `l2r`).
2. Load the dataset from `Dataset/{lang_name}_{directionality}_{condition}.csv` when `property=""`, or `Dataset/{lang_name}_{directionality}_{property}_{condition}.csv` otherwise.
3. Train/evaluate a seq2seq model.
4. Record accuracy, predictions, attention plots, and embeddings in `Results/`.

## Notes
- Audio experiments expect a local audio directory and WAV files referenced in the dataset CSVs.
- `property`, `directionality`, `conditions`, and `audio_root` are currently set directly in `hyper_params.py`.
- Text and feature experiments currently read `ur_string` and `sr_string` from the dataset CSVs.
- `audio_network_t2.py` is experimental and may be incomplete.
- Reproducibility: each run seeds `random`, `numpy`, and `torch` using `base_seed + run_num` (set in `hyper_params.py`).
