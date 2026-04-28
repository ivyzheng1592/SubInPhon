# SubInPhon

SubInPhon is a research codebase for modeling phonological patterns with sequence-to-sequence neural networks.
It supports text-based learning (UR -> SR), feature-embedded text learning, and audio-to-text/audio modeling.

## Project Layout
- `src/main.py` : entry point for running experiments
- `src/hyper_params.py` : global hyperparameters and output layout
- `src/text_dataset.py`, `src/feature_dataset.py`, `src/audio_dataset.py` : dataset loaders
- `src/text_network.py`, `src/feature_network.py` : text/feature seq2seq models
- `src/audio_network_t1.py`, `src/audio_network_t2.py` : audio seq2seq models
- `src/text_trainer.py`, `src/audio_trainer.py` : training/evaluation loops
- `src/text_recorder.py`, `src/audio_recorder.py` : metrics + plot recording
- `src/utils.py` : plotting utilities
- `src/language_registry.py` : language construction and standalone generation example
- `data/` : static data files and generated annotation files
- `results/` : model outputs, plots, and logs
- `misc/` : side scripts that are not part of the main experiment pipeline

## Quick Start
1. Create/activate a Python environment (3.8–3.11).
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run a default experiment:

```bash
python3 src/main.py
```

4. Override settings from the command line when needed:

```bash
python3 src/main.py --modality text --lang-name EnglishBH_shortened --runs 0:2 --device cpu
```

5. Set `property`, `directionality`, `conditions`, and `audio_root` in `hyper_params.py` before running:

```bash
python3 src/main.py --modality audio
```

6. Choose the audio model when needed:

```bash
python3 src/main.py --modality audio --audio-model t2
```

## Run Modes
- `train and evaluate`: train on train set; evaluate on test set; then attention + embedding on test
- `tuning`: train on train set; evaluate on valid set; then attention + embedding on valid
- `inspection`: attention + embedding on test (no training)

## CLI Options
- `--modality {text,feature,audio}` selects the experiment type
- `--lang-name` overrides `hp.lang_name`
- `--property` overrides `hp.property`
- `--runs` accepts a count (`2`) or range-style spec (`0:2`, `1:5:2`)
- `--trial-num` sets the output folder prefix; default is a timestamp
- `--data-proportion` overrides `hp.data_proportion`
- `--gen-data-proportion` overrides `hp.gen_data_proportion`
- `--audio-model {t1,t2}` overrides `hp.audio_model`
- `--n-epochs` overrides `hp.n_epochs`
- `--save-epochs` overrides `hp.save_epochs`
- `--base-seed` overrides `hp.base_seed`
- `--run-mode` overrides the run mode without editing `hyper_params.py`
- `--pred-log {vowel_only_error,consonant_vowel_error,all_correct_syll}` overrides `hp.pred_log` for a single run
- `--device {cpu,cuda}` overrides `hp.device` for a single run
- `--resume-model-file` resumes from a saved `*_seq2seq.pth` checkpoint

## Typical Experiment Flow
1. Choose a language entry, optional property, and directionality (e.g., `EnglishBH_shortened` + `""` + `l2r`).
2. Generate the annotation files under `data/{trial_num}_{lang_name}_generated_data/`.
3. Load the generated harmony or disharmony CSV for the current run.
4. Train/evaluate a seq2seq model.
5. Record accuracy, predictions, attention plots, and embeddings in `results/`.

## Notes
- Audio experiments expect a local audio directory and WAV files referenced in the dataset CSVs.
- `property`, `directionality`, `conditions`, and `audio_root` are currently set directly in `hyper_params.py`.
- Text and feature experiments currently read `ur_string` and `sr_string` from the dataset CSVs.
- Reproducibility: each run seeds `random`, `numpy`, and `torch` using `base_seed + run_num` (set in `hyper_params.py`).
