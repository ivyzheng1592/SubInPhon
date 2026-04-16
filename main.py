import argparse
from datetime import datetime
from typing import Optional, Union

from experiment_runner import run_experiment
import hyper_params as hp

"""
Command-line arguments:
- --modality {text,feature,audio}
- --trial-num TRIAL_ID
- --runs N | START:STOP[:STEP]
- --lang-name LANGUAGE_KEY
- --run-mode {"train and evaluate","tuning","inspection"}
- --pred-log {vowel_only_error,consonant_vowel_error,all_correct_syll}
- --device {cpu,cuda}
- --resume-model-file PATH_TO_CHECKPOINT

Typical examples:
- python3 main.py
- python3 main.py --modality text --lang-name EnglishBH_shortened --runs 0:2 --device cpu
- python3 main.py --modality audio --resume-model-file /path/to/model_seq2seq.pth

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
    parser.add_argument("--lang-name", default=None, help="Override hp.lang_name.")
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


def main() -> None:
    args = _build_parser().parse_args()

    if args.lang_name is not None:
        hp.lang_name = args.lang_name
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
        f"runs={list(runs)} trial_num={trial_num}"
    )

    run_experiment(
        args.modality,
        trial_num,
        runs,
        resume_model_file=args.resume_model_file,
    )


if __name__ == "__main__":
    main()
