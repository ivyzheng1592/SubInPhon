# SubInPhon

SubInPhon is a research codebase for modeling phonological patterns with sequence-to-sequence neural networks.
It supports text-based learning (UR -> SR), feature-embedded text learning, and audio-to-text/audio modeling.

## Project Layout
- `main.py` : entry point for running experiments
- `hyper_params.py` : global hyperparameters and output layout
- `text_dataset.py`, `feature_dataset.py`, `audio_dataset.py` : dataset loaders
- `text_network.py`, `feature_network.py` : text/feature seq2seq models
- `audio_network_t1.py` : audio seq2seq model (Translatotron-style)
- `text_run.py`, `audio_run.py` : training/evaluation loops
- `text_record.py`, `audio_record.py` : metrics + plot recording
- `utils.py` : plotting utilities
- `Dataset/` : data files and language definitions
- `Results/` : model outputs, plots, and logs

## Quick Start
1. Create/activate a Python environment (3.8–3.11).
2. Install requirements:

```bash
pip install torch==2.0.0 torchaudio==2.0.1 torchinfo==1.8.0
```

3. Run a default experiment (edit `main.py` to set language/conditions):

```bash
python main.py
```

## Run Modes
- `train and evaluate`: train on train set; evaluate on test set; then attention + embedding on test
- `tuning`: train on train set; evaluate on valid set; then attention + embedding on valid
- `evaluate only`: attention + embedding on test (no training)

## Typical Experiment Flow
1. Choose a language and property (e.g., `EnglishBH` + `shortened`).
2. Load the dataset from `Dataset/*.csv`.
3. Train/evaluate a seq2seq model.
4. Record accuracy, predictions, attention plots, and embeddings in `Results/`.

## Notes
- Audio experiments expect a local audio directory and WAV files referenced in the dataset CSVs.
- `audio_network_t2.py` is experimental and may be incomplete.
- Reproducibility: each run seeds `random`, `numpy`, and `torch` using `base_seed + run_num` (set in `hyper_params.py`).

## Next Cleanup Ideas
- Add a config system for experiment parameters.
- Move dataset helper scripts under a `scripts/` folder.
- Rename `audio_network_t1.py` to `audio_network.py` if it is the primary model.
