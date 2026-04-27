import argparse
from datetime import datetime
import os
import sys
from typing import Optional, Union

from experiment_runner import run_experiment
import hyper_params as hp

"""
Command-line arguments:
- --modality {text,feature,audio}
- --trial-num TRIAL_ID
- --runs N | START:STOP[:STEP]
- --data-proportion FLOAT
- --gen-data-proportion FLOAT
- --n-epochs INT
- --save-epochs INT
- --base-seed SEED
- --lang-name LANGUAGE_KEY
- --property PROPERTY_LABEL
- --run-mode {"train and evaluate","tuning","inspection"}
- --pred-log {vowel_only_error,consonant_vowel_error,all_correct_syll}
- --device {cpu,cuda}
- --resume-model-file PATH_TO_CHECKPOINT

Typical examples:
- python3 src/main.py
- python3 src/main.py --modality text --lang-name EnglishBH_shortened --runs 0:2 --device cpu
- python3 src/main.py --base-seed 2026
- python3 src/main.py --data-proportion 0.2 --gen-data-proportion 0.001 --n-epochs 20 --save-epochs 5
- python3 src/main.py --lang-name EnglishBH_expanded --property nonidentical
- python3 src/main.py --modality audio --resume-model-file /path/to/model_seq2seq.pth

Set these directly in hyper_params.py:
- property
- directionality
- conditions
- audio_root
"""


def _parse_runs(run_spec: Union[str, int, range]) -> range:
    if isinstance(run_spec, range):
        return run_spec
    if isinstance(run_spec, int):
        return range(run_spec)

    run_spec = str(run_spec).strip()
    if ":" in run_spec:
        parts = run_spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid run spec: {run_spec}")
        start = int(parts[0])
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        return range(start, stop, step)

    return range(int(run_spec))


def _resolve_trial_num(trial_num: Optional[str]) -> str:
    return trial_num or datetime.now().strftime("%Y%m%d%H%M")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SubInPhon experiments.")
    parser.add_argument(
        "--modality",
        choices=("text", "feature", "audio"),
        default="audio",
        help="Experiment modality to run.",
    )
    parser.add_argument(
        "--trial-num",
        default=None,
        help="Trial identifier for output folders. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--runs",
        default="2",
        help="Run count or Python-range-style spec. Examples: 2, 0:2, 1:5:2.",
    )
    parser.add_argument(
        "--data-proportion",
        type=float,
        default=None,
        help="Override hp.data_proportion for this run.",
    )
    parser.add_argument(
        "--gen-data-proportion",
        type=float,
        default=None,
        help="Override hp.gen_data_proportion for this run.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=None,
        help="Override hp.n_epochs for this run.",
    )
    parser.add_argument(
        "--save-epochs",
        type=int,
        default=None,
        help="Override hp.save_epochs for this run.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Override hp.base_seed for this run.",
    )
    parser.add_argument("--lang-name", default=None, help="Override hp.lang_name.")
    parser.add_argument(
        "--property",
        default=None,
        help="Override hp.property for this run.",
    )
    parser.add_argument(
        "--run-mode",
        choices=("train and evaluate", "tuning", "inspection"),
        default=None,
        help="Override hp.run_mode.",
    )
    parser.add_argument(
        "--pred-log",
        choices=("vowel_only_error", "consonant_vowel_error", "all_correct_syll"),
        default=None,
        help="Override hp.pred_log for this run.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Override hp.device for this run.",
    )
    parser.add_argument(
        "--resume-model-file",
        default=None,
        help="Resume from a saved *_seq2seq.pth model file.",
    )
    return parser


def _write_run_config(args: argparse.Namespace, trial_num: str, runs: range) -> None:
    modality_suffix = {
        "text": "txt",
        "feature": "fea",
        "audio": "aud",
    }[args.modality]
    result_dir = os.path.join("results", trial_num + "_" + hp.lang_name + "_" + modality_suffix)
    os.makedirs(result_dir, exist_ok=True)
    config_file = os.path.join(result_dir, "run_config.txt")

    hp_items = {}
    for name in dir(hp):
        if name.startswith("__"):
            continue
        value = getattr(hp, name)
        if callable(value):
            continue
        hp_items[name] = value

    with open(config_file, "w") as f:
        f.write("command:\n")
        f.write(" ".join(sys.argv) + "\n\n")

        f.write("input_arguments:\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write(f"resolved_trial_num: {trial_num}\n")
        f.write(f"resolved_runs: {list(runs)}\n\n")

        f.write("hyper_parameters:\n")
        for key in sorted(hp_items):
            f.write(f"{key}: {hp_items[key]}\n")


def main() -> None:
    args = _build_parser().parse_args()

    if args.data_proportion is not None:
        hp.data_proportion = args.data_proportion
    if args.gen_data_proportion is not None:
        hp.gen_data_proportion = args.gen_data_proportion
    if args.n_epochs is not None:
        hp.n_epochs = args.n_epochs
    if args.save_epochs is not None:
        hp.save_epochs = args.save_epochs
    if args.base_seed is not None:
        hp.base_seed = args.base_seed
    if args.lang_name is not None:
        hp.lang_name = args.lang_name
    if args.property is not None:
        hp.property = args.property
    if args.run_mode is not None:
        hp.run_mode = args.run_mode
    if args.pred_log is not None:
        hp.pred_log = args.pred_log
    if args.device is not None:
        hp.device = args.device

    trial_num = _resolve_trial_num(args.trial_num)
    runs = _parse_runs(args.runs)

    print(f"Running modality={args.modality} lang={hp.lang_name}")
    print(
        f"Run mode={hp.run_mode} pred_log={hp.pred_log} device={hp.device} "
        f"data_proportion={hp.data_proportion} gen_data_proportion={hp.gen_data_proportion} "
        f"n_epochs={hp.n_epochs} save_epochs={hp.save_epochs} "
        f"base_seed={hp.base_seed} lang_name={hp.lang_name} property={hp.property} "
        f"runs={list(runs)} trial_num={trial_num}"
    )

    _write_run_config(args, trial_num, runs)

    run_experiment(
        args.modality,
        trial_num,
        runs,
        resume_model_file=args.resume_model_file,
    )


if __name__ == "__main__":
    main()
