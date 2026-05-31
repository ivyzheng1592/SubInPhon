import gc
from pathlib import Path
from typing import Callable, List, Sequence

import pandas as pd


CV_TRIALS = ["EnglishBH_txt_cv", "EnglishBH_fea_cv"]
V_TRIALS = [
    "EnglishBH_txt_v",
    "EnglishBH_fea_v",
    "EnglishBH_nonidentical_txt_v",
    "EnglishBH_nonidentical_fea_v",
    "EnglishBH_expanded_txt_v",
    "EnglishBH_expanded_fea_v",
]
ALL_ACC_TRIALS = V_TRIALS + CV_TRIALS
RUN_KEYS = ["language", "modality", "directionality", "property", "condition", "run_num"]
CV_PRED_COLUMNS = [
    "language", "modality", "directionality", "property",
    "condition", "run_num", "epoch", "record_type",
    "v1_error", "v2_error", "o1_error", "o2_error",
    "c1_error", "c2_error", "pred_sr_v1", "pred_sr_v2",
]
V_PRED_COLUMNS = [
    "language", "modality", "directionality", "property",
    "condition", "run_num", "epoch", "record_type",
    "v1_error", "v2_error", "v3_error",
    "sr_v1", "sr_v2", "sr_v3",
    "pred_sr_v1", "pred_sr_v2", "pred_sr_v3",
]
VOWEL_BACK = {"i": 0, "ɪ": 0, "e": 0, "ɛ": 0, "u": 1, "ʊ": 1, "o": 1, "ɔ": 1}
VOWEL_HIGH = {"i": 1, "ɪ": 1, "u": 1, "ʊ": 1, "e": 0, "ɛ": 0, "o": 0, "ɔ": 0}
VOWEL_TENSE = {"i": 1, "u": 1, "e": 1, "o": 1, "ɪ": 0, "ʊ": 0, "ɛ": 0, "ɔ": 0}


def resolve_trial_dirs(base_dir: Path, trial: str) -> List[Path]:
    # Finds folders whose names match the canonical trial name or end with it.
    trial_dirs = sorted(
        path for path in base_dir.iterdir()
        if path.is_dir() and (path.name == trial or path.name.endswith("_" + trial))
    )
    print(
        f"resolved {trial} to "
        + (", ".join(str(path.relative_to(base_dir)) for path in trial_dirs) if trial_dirs else "no folders")
    )
    return trial_dirs


def list_matching_files(base_dir: Path, trials: Sequence[str], pattern: str) -> List[Path]:
    # Collects matching files from the requested trial folders.
    files: List[Path] = []
    for trial in trials:
        for trial_dir in resolve_trial_dirs(base_dir, trial):
            files.extend(sorted(trial_dir.rglob(pattern)))
    return files


def label_model(df: pd.DataFrame) -> pd.DataFrame:
    # Labels the model type from the modality column.
    df["model"] = df["modality"].str.contains("txt").map({True: "segment", False: "feature"})
    return df


def label_directionality(df: pd.DataFrame) -> pd.DataFrame:
    # Expands directionality abbreviations into analysis labels.
    df["directionality"] = df["directionality"].map({"l2r": "left-to-right", "r2l": "right-to-left"})
    return df


def label_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Labels each row as full, reduced, or expanded.
    df["property"] = df["property"].fillna("")
    df["dataset"] = "full"
    df.loc[df["property"] == "nonidentical", "dataset"] = "reduced"
    df.loc[df["language"].str.contains("expanded"), "dataset"] = "expanded"
    return df


def read_acc_files(base_dir: Path, trials: Sequence[str], with_trial: bool = False) -> pd.DataFrame:
    # Reads and concatenates accuracy files from the requested trial folders.
    frames: List[pd.DataFrame] = []
    for trial in trials:
        for trial_dir in resolve_trial_dirs(base_dir, trial):
            for file_path in sorted(trial_dir.rglob("*acc.csv")):
                this_run = pd.read_csv(file_path)
                if with_trial:
                    this_run["trial"] = trial
                frames.append(this_run)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_failed_run_list(acc: pd.DataFrame) -> pd.DataFrame:
    # Collects run keys for runs that fail the final test threshold.
    acc = acc.copy()
    acc["property"] = acc["property"].fillna("")
    return (
        acc[
            (acc["epoch"] == 99)
            & (acc["record_type"] == "test")
            & ((acc["acc"] < 0.85) | (acc["loss"] > 0.05))
        ][RUN_KEYS]
        .drop_duplicates()
    )


def filter_failed_runs(df: pd.DataFrame, failed_run_list: pd.DataFrame) -> pd.DataFrame:
    # Removes rows belonging to failed runs.
    df = df.copy()
    df["property"] = df["property"].fillna("")
    return df.merge(
        failed_run_list[RUN_KEYS].drop_duplicates(),
        on=RUN_KEYS,
        how="left",
        indicator=True,
    ).loc[lambda x: x["_merge"] == "left_only"].drop(columns="_merge")


def add_vowel_features(df: pd.DataFrame) -> pd.DataFrame:
    # Adds vowel height, tense, and backness features for source and predicted vowels.
    for prefix in ["sr_v1", "sr_v2", "sr_v3", "pred_sr_v1", "pred_sr_v2", "pred_sr_v3"]:
        df[f"{prefix}_high"] = df[prefix].map(VOWEL_HIGH)
        df[f"{prefix}_tense"] = df[prefix].map(VOWEL_TENSE)
        df[f"{prefix}_back"] = df[prefix].map(VOWEL_BACK)
    return df


def iter_run_chunks(
    file_path: Path,
    usecols: Sequence[str] | Callable[[str], bool],
    output_columns: Sequence[str],
    chunk_size: int = 200000,
):
    # Stores the last run from the previous read chunk.
    carryover = pd.DataFrame(columns=output_columns)
    for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=chunk_size):
        # Prepends the previous incomplete run to the current read chunk.
        if not carryover.empty:
            chunk = pd.concat([carryover, chunk], ignore_index=True)
        chunk["property"] = chunk["property"].fillna("")
        # Gets the run key of the last row in the current chunk.
        last_run = tuple(chunk.iloc[-1][RUN_KEYS])
        run_keys = chunk[RUN_KEYS].apply(tuple, axis=1)
        # Splits the chunk into complete runs and the last run.
        complete_mask = run_keys != last_run
        complete_chunk = chunk.loc[complete_mask].copy()
        carryover = chunk.loc[~complete_mask].copy()
        if complete_chunk.empty:
            del chunk, run_keys, complete_mask, complete_chunk
            gc.collect()
            continue
        # Yields one complete run at a time.
        for _, run_df in complete_chunk.groupby(RUN_KEYS, sort=False):
            yield run_df.reset_index(drop=True)
        del chunk, run_keys, complete_mask, complete_chunk
        gc.collect()
    # Yields the final run after the file has been fully read.
    if not carryover.empty:
        yield carryover.reset_index(drop=True)


def append_csv(df: pd.DataFrame, output_file: Path, first_write: bool, source_file: Path, base_dir: Path) -> bool:
    # Writes a dataframe to a CSV, using append mode after the first write.
    if df.empty:
        return first_write
    df.to_csv(output_file, mode="w" if first_write else "a", index=False, header=first_write)
    run_key = ", ".join(f"{key}={df.iloc[0][key]}" for key in RUN_KEYS if key in df.columns)
    print(
        f"wrote {output_file.relative_to(base_dir)} from {source_file.relative_to(base_dir)}"
        + (f" for {run_key}" if run_key else "")
    )
    return False


def clean_all_acc(base_dir: Path, output_dir: Path) -> None:
    # Builds the cleaned all-accuracy summary across cv and v trials.
    acc = read_acc_files(base_dir, ALL_ACC_TRIALS, with_trial=True)
    failed_run_list = build_failed_run_list(acc)
    acc = filter_failed_runs(acc, failed_run_list)
    acc = label_model(acc)
    acc = label_directionality(acc)
    acc = label_dataset(acc)
    acc["error_record"] = acc["trial"].str.contains("cv").map({True: "cv", False: "v"})
    acc = acc.rename(columns={"record_type": "subset"})
    acc = acc[
        ["language", "model", "directionality", "dataset", "error_record", "condition", "run_num", "subset", "epoch", "loss", "acc"]
    ]
    output_file = output_dir / "cleaned_260518_EnglishBH_all_acc.csv"
    acc.to_csv(output_file, index=False)
    input_files = [path.relative_to(base_dir) for path in list_matching_files(base_dir, ALL_ACC_TRIALS, "*acc.csv")]
    print(f"wrote {output_file.relative_to(base_dir)} from {', '.join(str(path) for path in input_files)}")
    del acc, failed_run_list, input_files
    gc.collect()


def clean_cv_pred(base_dir: Path, output_dir: Path) -> None:
    # Builds the cv prediction summary from per-run prediction data.
    output_file = output_dir / "cleaned_260518_EnglishBH_cv_pred.csv"
    output_file.unlink(missing_ok=True)
    first_write = True

    for trial in CV_TRIALS:
        trial_dirs = resolve_trial_dirs(base_dir, trial)
        acc_files: List[Path] = []
        pred_files: List[Path] = []
        for trial_dir in trial_dirs:
            acc_files.extend(sorted(trial_dir.rglob("*acc.csv")))
            pred_files.extend(sorted(trial_dir.rglob("*pred.csv")))
        print(f"processing {trial} across {len(trial_dirs)} folders with {len(acc_files)} acc files and {len(pred_files)} pred files")

        # Prepares the run-level accuracy totals used to infer syllable-structure errors.
        trial_acc = pd.concat((pd.read_csv(file_path) for file_path in acc_files), ignore_index=True)
        failed_run_list = build_failed_run_list(trial_acc)
        trial_acc = filter_failed_runs(trial_acc, failed_run_list)
        trial_acc = label_model(trial_acc)
        trial_acc = label_directionality(trial_acc)
        trial_acc["dataset"] = "full"
        trial_acc = trial_acc.rename(columns={"record_type": "subset"})
        trial_acc["total_data"] = 167968
        trial_acc["data_split"] = trial_acc["subset"].map({"train": 0.8, "test": 0.1})
        trial_acc["subset_data"] = trial_acc["total_data"] * trial_acc["data_split"]
        trial_acc["total_error"] = (trial_acc["subset_data"] * (1 - trial_acc["acc"])).astype(int)
        trial_acc = trial_acc[
            ["model", "directionality", "dataset", "condition", "run_num", "epoch", "subset", "total_data", "data_split", "subset_data", "total_error"]
        ]

        for pred_file in pred_files:
            print(f"reading {pred_file.relative_to(base_dir)}")
            for this_run in iter_run_chunks(pred_file, CV_PRED_COLUMNS, CV_PRED_COLUMNS):
                # Labels each prediction row and derives consonant and vowel error indicators.
                this_run = filter_failed_runs(this_run, failed_run_list)
                if this_run.empty:
                    continue
                this_run = label_model(this_run)
                this_run = label_directionality(this_run)
                this_run["dataset"] = "full"
                this_run = this_run.rename(columns={"record_type": "subset"})
                run_acc = trial_acc[
                    (trial_acc["model"] == this_run["model"].iat[0])
                    & (trial_acc["directionality"] == this_run["directionality"].iat[0])
                    & (trial_acc["dataset"] == this_run["dataset"].iat[0])
                    & (trial_acc["condition"] == this_run["condition"].iat[0])
                    & (trial_acc["run_num"] == this_run["run_num"].iat[0])
                ]
                this_run["v_error"] = (
                    (this_run["v1_error"] != 0) 
                    | (this_run["v2_error"] != 0)
                ).astype(int)
                this_run["c_error"] = (
                    (this_run["o1_error"] != 0)
                    | (this_run["o2_error"] != 0)
                    | (this_run["c1_error"] != 0)
                    | (this_run["c2_error"] != 0)
                ).astype(int)

                this_summary = (
                    this_run.groupby(
                        ["model", "directionality", "dataset", "condition", "run_num", "epoch", "subset", "c_error", "v_error"],
                        as_index=False,
                    )
                    .size()
                    .rename(columns={"size": "error_num"})
                )

                # Completes the four c_error x v_error combinations for each run, epoch, and subset.
                run_keys = run_acc[
                    ["model", "directionality", "dataset", "condition", "run_num", "epoch", "subset"]
                ].drop_duplicates()
                combos = pd.MultiIndex.from_product([[0, 1], [0, 1]], names=["c_error", "v_error"]).to_frame(index=False)
                this_summary = run_keys.merge(combos, how="cross").merge(
                    this_summary,
                    on=["model", "directionality", "dataset", "condition", "run_num", "epoch", "subset", "c_error", "v_error"],
                    how="left",
                )
                this_summary["error_num"] = pd.to_numeric(this_summary["error_num"], errors="coerce").fillna(0).astype(int)
                this_summary["segment_error"] = this_summary.groupby(
                    ["model", "directionality", "dataset", "condition", "run_num", "epoch", "subset"]
                )["error_num"].transform("sum")
                this_summary = this_summary.merge(
                    trial_acc,
                    on=["model", "directionality", "dataset", "condition", "run_num", "epoch", "subset"],
                    how="left",
                )
                # Replaces the 00 combination with the inferred syllable-structure error count.
                mask = (this_summary["c_error"] == 0) & (this_summary["v_error"] == 0)
                this_summary.loc[mask, "error_num"] = (
                    this_summary.loc[mask, "total_error"] - this_summary.loc[mask, "segment_error"]
                )
                this_summary["error_type"] = this_summary.apply(
                    lambda row: "consonant only"
                    if row["c_error"] == 1 and row["v_error"] == 0
                    else "vowel only"
                    if row["c_error"] == 0 and row["v_error"] == 1
                    else "consonant and vowel"
                    if row["c_error"] == 1 and row["v_error"] == 1
                    else "syllable structure",
                    axis=1,
                )
                this_summary["error_type"] = pd.Categorical(
                    this_summary["error_type"],
                    categories=[
                        "syllable structure",
                        "consonant and vowel",
                        "consonant only",
                        "vowel only",
                    ],
                    ordered=True,
                )
                this_summary = this_summary.drop(columns=["c_error", "v_error", "segment_error"])
                this_summary["error_rate"] = this_summary.apply(
                    lambda row: 0 if row["total_error"] == 0 else row["error_num"] / row["total_error"],
                    axis=1,
                )
                this_summary = this_summary[
                    [
                        "model",
                        "directionality",
                        "dataset",
                        "condition",
                        "run_num",
                        "epoch",
                        "subset",
                        "error_type",
                        "error_num",
                        "error_rate",
                        "total_data",
                        "data_split",
                        "subset_data",
                        "total_error",
                    ]
                ]
                first_write = append_csv(this_summary, output_file, first_write, pred_file, base_dir)
                del this_run, run_acc, this_summary, run_keys, combos, mask
                gc.collect()
        del trial_acc, failed_run_list, trial_dirs, acc_files, pred_files
        gc.collect()


def clean_v_pred(base_dir: Path, output_dir: Path) -> None:
    # Builds the v prediction summary and the vowel-height comparison summary.
    acc = read_acc_files(base_dir, V_TRIALS)
    failed_run_list = build_failed_run_list(acc)
    acc = filter_failed_runs(acc, failed_run_list)
    acc = label_model(acc)
    acc = label_directionality(acc)
    acc = label_dataset(acc)
    acc = acc.rename(columns={"record_type": "subset"})
    pred_summary_file = output_dir / "cleaned_260518_EnglishBH_v_pred.csv"
    vowel_height_file = output_dir / "cleaned_260518_EnglishBH_v_height.csv"
    pred_summary_file.unlink(missing_ok=True)
    vowel_height_file.unlink(missing_ok=True)
    first_pred_write = True
    first_vowel_height_write = True

    for trial in V_TRIALS:
        trial_dirs = resolve_trial_dirs(base_dir, trial)
        pred_files: List[Path] = []
        for trial_dir in trial_dirs:
            pred_files.extend(sorted(trial_dir.rglob("*pred.csv")))
        print(f"processing {trial} across {len(trial_dirs)} folders with {len(pred_files)} pred files")
        for pred_file in pred_files:
            print(f"reading {pred_file.relative_to(base_dir)}")
            for this_run in iter_run_chunks(pred_file, lambda col: col in V_PRED_COLUMNS, V_PRED_COLUMNS):
                # Restores any missing expanded-dataset columns before selecting the analysis columns.
                missing_cols = [col for col in V_PRED_COLUMNS if col not in this_run.columns]
                for col in missing_cols:
                    this_run[col] = pd.NA
                this_run = this_run[V_PRED_COLUMNS].copy()
                this_run["v3_error"] = pd.to_numeric(this_run["v3_error"], errors="coerce").fillna(0).astype(int)
                this_run = filter_failed_runs(this_run, failed_run_list)
                if this_run.empty:
                    continue
                this_run = label_model(this_run)
                this_run = label_directionality(this_run)
                this_run = label_dataset(this_run)
                this_run = this_run.rename(columns={"record_type": "subset"})
                run_acc = acc[
                    (acc["model"] == this_run["model"].iat[0])
                    & (acc["directionality"] == this_run["directionality"].iat[0])
                    & (acc["dataset"] == this_run["dataset"].iat[0])
                    & (acc["condition"] == this_run["condition"].iat[0])
                    & (acc["run_num"] == this_run["run_num"].iat[0])
                ]
                this_run = add_vowel_features(this_run)

                # Derives feature-level vowel error indicators for each prediction row.
                expanded_mask = this_run["dataset"] == "expanded"
                this_run["high_error"] = (
                    (this_run["sr_v1_high"] != this_run["pred_sr_v1_high"])
                    | (this_run["sr_v2_high"] != this_run["pred_sr_v2_high"])
                    | (expanded_mask & (this_run["sr_v3_high"] != this_run["pred_sr_v3_high"]))
                ).astype(int)
                this_run["tense_error"] = (
                    (this_run["sr_v1_tense"] != this_run["pred_sr_v1_tense"])
                    | (this_run["sr_v2_tense"] != this_run["pred_sr_v2_tense"])
                    | (expanded_mask & (this_run["sr_v3_tense"] != this_run["pred_sr_v3_tense"]))
                ).astype(int)
                this_run["back_error"] = (
                    (this_run["sr_v1_back"] != this_run["pred_sr_v1_back"])
                    | (this_run["sr_v2_back"] != this_run["pred_sr_v2_back"])
                    | (expanded_mask & (this_run["sr_v3_back"] != this_run["pred_sr_v3_back"]))
                ).astype(int)

                pred_triplets = this_run["pred_sr_v1_back"].astype("Int64").astype(str).str.cat(
                    [
                        this_run["pred_sr_v2_back"].astype("Int64").astype(str),
                        this_run["pred_sr_v3_back"].astype("Int64").astype(str),
                    ],
                    sep="",
                )
                this_run["harmony_error"] = 0
                this_run.loc[
                    expanded_mask & (this_run["condition"] == "harmony") & ~pred_triplets.isin(["000", "111"]),
                    "harmony_error",
                ] = 1
                this_run.loc[
                    expanded_mask
                    & (this_run["condition"] == "disharmony")
                    & (this_run["directionality"] == "left-to-right")
                    & ~pred_triplets.isin(["100", "011"]),
                    "harmony_error",
                ] = 1
                this_run.loc[
                    expanded_mask
                    & (this_run["condition"] == "disharmony")
                    & (this_run["directionality"] == "right-to-left")
                    & ~pred_triplets.isin(["110", "001"]),
                    "harmony_error",
                ] = 1
                this_run.loc[
                    (~expanded_mask) & (this_run["condition"] == "harmony") & (this_run["pred_sr_v1_back"] != this_run["pred_sr_v2_back"]),
                    "harmony_error",
                ] = 1
                this_run.loc[
                    (~expanded_mask) & (this_run["condition"] == "disharmony") & (this_run["pred_sr_v1_back"] == this_run["pred_sr_v2_back"]),
                    "harmony_error",
                ] = 1

                # Summarizes prediction counts by run, epoch, subset, and vowel error pattern.
                this_summary = (
                    this_run.groupby(
                        [
                            "model", "directionality", "dataset", "condition", "run_num", "epoch", "subset",
                            "v1_error", "v2_error", "v3_error", "high_error", "tense_error", "back_error", "harmony_error",
                        ],
                        as_index=False,
                    )
                    .size()
                    .rename(columns={"size": "error_num"})
                )
                run_keys = run_acc[
                    ["model", "directionality", "dataset", "condition", "run_num", "epoch", "subset"]
                ].drop_duplicates()
                error_keys = this_summary[
                    ["v1_error", "v2_error", "v3_error", "high_error", "tense_error", "back_error", "harmony_error"]
                ].drop_duplicates()
                this_summary = run_keys.merge(error_keys, how="cross").merge(
                    this_summary,
                    on=[
                        "model",
                        "directionality",
                        "dataset",
                        "condition",
                        "run_num",
                        "epoch",
                        "subset",
                        "v1_error",
                        "v2_error",
                        "v3_error",
                        "high_error",
                        "tense_error",
                        "back_error",
                        "harmony_error",
                    ],
                    how="left",
                )
                this_summary["error_num"] = pd.to_numeric(this_summary["error_num"], errors="coerce").fillna(0).astype(int)

                # Summarizes input height-pattern counts for full harmony data.
                this_input_height_summary = this_run[
                    (this_run["dataset"] == "full") & (this_run["condition"] == "harmony")
                ].copy()
                this_input_height_summary["v_high"] = (
                    this_input_height_summary["sr_v1_high"].astype("Int64").astype(str)
                    + this_input_height_summary["sr_v2_high"].astype("Int64").astype(str)
                )
                this_input_height_summary["v_agree"] = (
                    this_input_height_summary["sr_v1_high"] == this_input_height_summary["sr_v2_high"]
                ).astype(int)
                this_input_height_summary = (
                    this_input_height_summary.groupby(
                        ["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"],
                        as_index=False,
                    )
                    .size()
                    .rename(columns={"size": "error_num"})
                )
                this_input_height_summary["error_rate"] = this_input_height_summary.groupby(
                    ["model", "directionality", "dataset", "condition", "run_num"]
                )["error_num"].transform(lambda s: s / s.sum())
                input_run_keys = this_input_height_summary[
                    ["model", "directionality", "dataset", "condition", "run_num"]
                ].drop_duplicates()
                input_combos = this_input_height_summary[["v_high", "v_agree"]].drop_duplicates()
                this_input_height_summary = input_run_keys.merge(input_combos, how="cross").merge(
                    this_input_height_summary,
                    on=["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"],
                    how="left",
                )
                this_input_height_summary["error_num"] = pd.to_numeric(
                    this_input_height_summary["error_num"], errors="coerce"
                ).fillna(0).astype(int)
                this_input_height_summary["error_rate"] = pd.to_numeric(
                    this_input_height_summary["error_rate"], errors="coerce"
                ).fillna(0.0)

                # Summarizes predicted height-pattern counts for full harmony data.
                this_output_height_summary = this_run[
                    (this_run["dataset"] == "full") & (this_run["condition"] == "harmony")
                ].copy()
                this_output_height_summary["v_high"] = (
                    this_output_height_summary["pred_sr_v1_high"].astype("Int64").astype(str)
                    + this_output_height_summary["pred_sr_v2_high"].astype("Int64").astype(str)
                )
                this_output_height_summary["v_agree"] = (
                    this_output_height_summary["pred_sr_v1_high"] == this_output_height_summary["pred_sr_v2_high"]
                ).astype(int)
                this_output_height_summary = (
                    this_output_height_summary.groupby(
                        ["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"],
                        as_index=False,
                    )
                    .size()
                    .rename(columns={"size": "error_num"})
                )
                this_output_height_summary["error_rate"] = this_output_height_summary.groupby(
                    ["model", "directionality", "dataset", "condition", "run_num"]
                )["error_num"].transform(lambda s: s / s.sum())
                output_run_keys = this_output_height_summary[
                    ["model", "directionality", "dataset", "condition", "run_num"]
                ].drop_duplicates()
                output_combos = this_output_height_summary[["v_high", "v_agree"]].drop_duplicates()
                this_output_height_summary = output_run_keys.merge(output_combos, how="cross").merge(
                    this_output_height_summary,
                    on=["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"],
                    how="left",
                )
                this_output_height_summary["error_num"] = pd.to_numeric(
                    this_output_height_summary["error_num"], errors="coerce"
                ).fillna(0).astype(int)
                this_output_height_summary["error_rate"] = pd.to_numeric(
                    this_output_height_summary["error_rate"], errors="coerce"
                ).fillna(0.0)

                # Combines the input and predicted height-pattern summaries.
                this_height_summary = this_input_height_summary.merge(
                    this_output_height_summary,
                    on=["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"],
                    how="outer",
                    suffixes=("_input", "_output"),
                )
                for col in ["error_num_input", "error_num_output"]:
                    this_height_summary[col] = pd.to_numeric(this_height_summary[col], errors="coerce").fillna(0).astype(int)
                for col in ["error_rate_input", "error_rate_output"]:
                    this_height_summary[col] = pd.to_numeric(this_height_summary[col], errors="coerce").fillna(0.0)
                this_height_summary["error_num_diff"] = (
                    this_height_summary["error_num_output"] - this_height_summary["error_num_input"]
                )
                this_height_summary["error_rate_diff"] = (
                    this_height_summary["error_rate_output"] - this_height_summary["error_rate_input"]
                )

                first_pred_write = append_csv(this_summary, pred_summary_file, first_pred_write, pred_file, base_dir)
                first_vowel_height_write = append_csv(
                    this_height_summary,
                    vowel_height_file,
                    first_vowel_height_write,
                    pred_file,
                    base_dir,
                )
                del (
                    this_run,
                    run_acc,
                    missing_cols,
                    expanded_mask,
                    pred_triplets,
                    this_summary,
                    run_keys,
                    error_keys,
                    this_input_height_summary,
                    input_run_keys,
                    input_combos,
                    this_output_height_summary,
                    output_run_keys,
                    output_combos,
                    this_height_summary,
                )
                gc.collect()
        del trial_dirs, pred_files
        gc.collect()
    del acc, failed_run_list
    gc.collect()


def main() -> None:
    base_dir = "/media/ldlmdl/A2AAE4B1AAE482E1/SSD_Documents/subinphon/results"
    output_dir = "/media/ldlmdl/A2AAE4B1AAE482E1/SSD_Documents/subinphon/data"

    clean_all_acc(base_dir, output_dir)
    clean_cv_pred(base_dir, output_dir)
    clean_v_pred(base_dir, output_dir)


if __name__ == "__main__":
    main()
