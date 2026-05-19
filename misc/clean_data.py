import argparse
from pathlib import Path
from typing import List, Sequence, Set, Tuple

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
VOWEL_BACK = {"i": 0, "ɪ": 0, "e": 0, "ɛ": 0, "u": 1, "ʊ": 1, "o": 1, "ɔ": 1}
VOWEL_HIGH = {"i": 1, "ɪ": 1, "u": 1, "ʊ": 1, "e": 0, "ɛ": 0, "o": 0, "ɔ": 0}
VOWEL_TENSE = {"i": 1, "u": 1, "e": 1, "o": 1, "ɪ": 0, "ʊ": 0, "ɛ": 0, "ɔ": 0}
FAILED_RUN_KEYS = ["language", "modality", "directionality", "property", "condition", "run_num"]
CV_PRED_COLUMNS = [
    "language", "modality", "directionality", "property", "condition", "run_num", "epoch", "record_type",
    "v1_error", "v2_error", "o1_error", "o2_error", "c1_error", "c2_error", "pred_sr_v1", "pred_sr_v2",
]
V_PRED_COLUMNS = [
    "language", "modality", "directionality", "property", "condition", "run_num", "epoch", "record_type",
    "v1_error", "v2_error", "v3_error", "sr_v1", "sr_v2", "sr_v3", "pred_sr_v1", "pred_sr_v2", "pred_sr_v3",
]


def list_matching_files(base_dir: Path, trials: Sequence[str], pattern: str) -> List[Path]:
    files: List[Path] = []
    for trial in trials:
        files.extend(sorted((base_dir / trial).rglob(pattern)))
    return files


def label_dataset(language: pd.Series, property_col: pd.Series) -> pd.Series:
    property_col = property_col.fillna("")
    return pd.Series(pd.NA, index=language.index).mask(
        language.str.contains("expanded"),
        "expanded",
    ).mask(
        ~language.str.contains("expanded") & property_col.eq("nonidentical"),
        "reduced",
    ).fillna("full")


def label_model(modality: pd.Series) -> pd.Series:
    return modality.str.contains("txt").map({True: "segment", False: "feature"})


def label_directionality(directionality: pd.Series) -> pd.Series:
    return directionality.map({"l2r": "left-to-right", "r2l": "right-to-left"})


def read_acc_files(base_dir: Path, trials: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for trial in trials:
        for file_path in sorted((base_dir / trial).rglob("*acc.csv")):
            frame = pd.read_csv(file_path)
            frame["trial"] = trial
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def filter_failed_runs(df: pd.DataFrame, acc: pd.DataFrame) -> pd.DataFrame:
    failed = acc[
        (acc["epoch"] == 99)
        & (acc["record_type"] == "test")
        & ((acc["acc"] < 0.85) | (acc["loss"] > 0.05))
    ][FAILED_RUN_KEYS].drop_duplicates()
    failed["property"] = failed["property"].fillna("")
    failed_runs: Set[Tuple[str, str, str, str, str, int]] = {
        (
            str(row.language),
            str(row.modality),
            str(row.directionality),
            str(row.property),
            str(row.condition),
            int(row.run_num),
        )
        for row in failed.itertuples(index=False)
    }
    property_col = df["property"].fillna("")
    keys = list(
        zip(
            df["language"].astype(str),
            df["modality"].astype(str),
            df["directionality"].astype(str),
            property_col.astype(str),
            df["condition"].astype(str),
            df["run_num"].astype(int),
        )
    )
    keep_mask = [key not in failed_runs for key in keys]
    filtered = df.loc[keep_mask].copy()
    filtered["property"] = property_col.loc[keep_mask].to_numpy()
    return filtered


def clean_all_acc(base_dir: Path, output_dir: Path) -> None:
    acc = read_acc_files(base_dir, ALL_ACC_TRIALS)
    input_files = [path.relative_to(base_dir) for path in list_matching_files(base_dir, ALL_ACC_TRIALS, "*acc.csv")]
    acc = filter_failed_runs(acc, acc)
    acc["model"] = label_model(acc["modality"])
    acc["directionality"] = label_directionality(acc["directionality"])
    acc["dataset"] = label_dataset(acc["language"], acc["property"])
    acc["error_record"] = acc["trial"].str.contains("cv").map({True: "cv", False: "v"})
    acc = acc.rename(columns={"record_type": "subset"})
    acc = acc[
        ["language", "model", "directionality", "dataset", "error_record", "condition", "run_num", "subset", "epoch", "loss", "acc"]
    ]
    output_file = output_dir / "cleaned_260518_EnglishBH_all_acc.csv"
    acc.to_csv(output_file, index=False)
    print(f"wrote {output_file.relative_to(base_dir)} from {', '.join(str(path) for path in input_files)}")


def build_cv_acc_summary(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    acc = read_acc_files(base_dir, CV_TRIALS)
    raw_acc = acc.copy()
    acc = filter_failed_runs(acc, acc)
    acc["model"] = label_model(acc["modality"])
    acc["directionality"] = label_directionality(acc["directionality"])
    acc["dataset"] = "full"
    acc = acc.rename(columns={"record_type": "subset"})
    acc = acc[acc["subset"].isin(["train", "test"])].copy()
    acc["total_data"] = 167968
    acc["data_split"] = acc["subset"].map({"train": 0.8, "test": 0.1})
    acc["subset_data"] = acc["total_data"] * acc["data_split"]
    acc["total_error"] = (acc["subset_data"] * (1 - acc["acc"])).astype(int)
    acc_summary = (
        acc.groupby(["model", "directionality", "dataset", "condition", "run_num"], as_index=False)["total_error"]
        .sum()
    )
    return raw_acc, acc_summary


def summarize_cv_pred_chunk(chunk: pd.DataFrame, acc_summary: pd.DataFrame) -> pd.DataFrame:
    chunk["model"] = label_model(chunk["modality"])
    chunk["directionality"] = label_directionality(chunk["directionality"])
    chunk["dataset"] = "full"
    chunk = chunk.rename(columns={"record_type": "subset"})
    chunk["v_error"] = ((chunk["v1_error"] != 0) | (chunk["v2_error"] != 0)).astype(int)
    chunk["c_error"] = (
        (chunk["o1_error"] != 0)
        | (chunk["o2_error"] != 0)
        | (chunk["c1_error"] != 0)
        | (chunk["c2_error"] != 0)
    ).astype(int)
    chunk["pred_sr_v1_back"] = chunk["pred_sr_v1"].map(VOWEL_BACK)
    chunk["pred_sr_v2_back"] = chunk["pred_sr_v2"].map(VOWEL_BACK)
    chunk["harmony_error"] = 0
    harmony_mask = (chunk["condition"] == "harmony") & (chunk["pred_sr_v1_back"] != chunk["pred_sr_v2_back"])
    disharmony_mask = (chunk["condition"] == "disharmony") & (chunk["pred_sr_v1_back"] == chunk["pred_sr_v2_back"])
    chunk.loc[harmony_mask | disharmony_mask, "harmony_error"] = 1

    grouped = (
        chunk.groupby(
            ["model", "directionality", "dataset", "condition", "run_num", "epoch", "subset", "c_error", "v_error"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "error_num"})
    )

    run_keys = grouped[["model", "directionality", "dataset", "condition", "run_num"]].drop_duplicates()
    combos = pd.MultiIndex.from_product([[0, 1], [0, 1]], names=["c_error", "v_error"]).to_frame(index=False)
    complete = run_keys.merge(combos, how="cross")
    grouped = complete.merge(
        grouped,
        on=["model", "directionality", "dataset", "condition", "run_num", "c_error", "v_error"],
        how="left",
    )
    grouped["error_num"] = grouped["error_num"].fillna(0).astype(int)
    grouped["segment_error"] = grouped.groupby(
        ["model", "directionality", "dataset", "condition", "run_num"]
    )["error_num"].transform("sum")
    grouped = grouped.merge(
        acc_summary,
        on=["model", "directionality", "dataset", "condition", "run_num"],
        how="left",
    )
    mask = (grouped["c_error"] == 0) & (grouped["v_error"] == 0)
    grouped.loc[mask, "error_num"] = grouped.loc[mask, "total_error"] - grouped.loc[mask, "segment_error"]
    grouped["error_type"] = grouped.apply(
        lambda row: "consonant only"
        if row["c_error"] == 1 and row["v_error"] == 0
        else "vowel only"
        if row["c_error"] == 0 and row["v_error"] == 1
        else "consonant and vowel"
        if row["c_error"] == 1 and row["v_error"] == 1
        else "syllable structure",
        axis=1,
    )
    grouped = grouped.drop(columns=["c_error", "v_error", "segment_error"])
    grouped["error_rate"] = grouped.apply(
        lambda row: 0 if row["total_error"] == 0 else row["error_num"] / row["total_error"],
        axis=1,
    )
    return grouped


def add_vowel_features(df: pd.DataFrame) -> pd.DataFrame:
    for prefix in ["sr_v1", "sr_v2", "sr_v3", "pred_sr_v1", "pred_sr_v2", "pred_sr_v3"]:
        df[f"{prefix}_high"] = df[prefix].map(VOWEL_HIGH)
        df[f"{prefix}_tense"] = df[prefix].map(VOWEL_TENSE)
        df[f"{prefix}_back"] = df[prefix].map(VOWEL_BACK)
    return df


def summarize_v_pred_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk["model"] = label_model(chunk["modality"])
    chunk["directionality"] = label_directionality(chunk["directionality"])
    chunk["dataset"] = label_dataset(chunk["language"], chunk["property"])
    chunk = chunk.rename(columns={"record_type": "subset"})
    chunk = add_vowel_features(chunk)

    expanded_mask = chunk["dataset"] == "expanded"
    chunk["high_error"] = (
        (chunk["sr_v1_high"] != chunk["pred_sr_v1_high"])
        | (chunk["sr_v2_high"] != chunk["pred_sr_v2_high"])
        | (expanded_mask & (chunk["sr_v3_high"] != chunk["pred_sr_v3_high"]))
    ).astype(int)
    chunk["tense_error"] = (
        (chunk["sr_v1_tense"] != chunk["pred_sr_v1_tense"])
        | (chunk["sr_v2_tense"] != chunk["pred_sr_v2_tense"])
        | (expanded_mask & (chunk["sr_v3_tense"] != chunk["pred_sr_v3_tense"]))
    ).astype(int)
    chunk["back_error"] = (
        (chunk["sr_v1_back"] != chunk["pred_sr_v1_back"])
        | (chunk["sr_v2_back"] != chunk["pred_sr_v2_back"])
        | (expanded_mask & (chunk["sr_v3_back"] != chunk["pred_sr_v3_back"]))
    ).astype(int)
    chunk["harmony_error"] = 0

    pred_triplets = chunk["pred_sr_v1_back"].astype("Int64").astype(str).str.cat(
        [chunk["pred_sr_v2_back"].astype("Int64").astype(str), chunk["pred_sr_v3_back"].astype("Int64").astype(str)],
        sep="",
    )
    expanded_harmony = expanded_mask & (chunk["condition"] == "harmony")
    expanded_dish_l2r = expanded_mask & (chunk["condition"] == "disharmony") & (chunk["directionality"] == "left-to-right")
    expanded_dish_r2l = expanded_mask & (chunk["condition"] == "disharmony") & (chunk["directionality"] == "right-to-left")
    non_expanded = chunk["dataset"] != "expanded"

    chunk.loc[expanded_harmony & ~pred_triplets.isin(["000", "111"]), "harmony_error"] = 1
    chunk.loc[expanded_dish_l2r & ~pred_triplets.isin(["100", "011"]), "harmony_error"] = 1
    chunk.loc[expanded_dish_r2l & ~pred_triplets.isin(["110", "001"]), "harmony_error"] = 1
    chunk.loc[
        non_expanded & (chunk["condition"] == "harmony") & (chunk["pred_sr_v1_back"] != chunk["pred_sr_v2_back"]),
        "harmony_error",
    ] = 1
    chunk.loc[
        non_expanded & (chunk["condition"] == "disharmony") & (chunk["pred_sr_v1_back"] == chunk["pred_sr_v2_back"]),
        "harmony_error",
    ] = 1

    return (
        chunk.groupby(
            [
                "model", "directionality", "dataset", "condition", "run_num", "epoch", "subset",
                "v1_error", "v2_error", "high_error", "tense_error", "back_error", "harmony_error",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "error_num"})
    )


def summarize_height_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk["model"] = label_model(chunk["modality"])
    chunk["directionality"] = label_directionality(chunk["directionality"])
    chunk["dataset"] = label_dataset(chunk["language"], chunk["property"])
    chunk = add_vowel_features(chunk)
    chunk = chunk[(chunk["dataset"] == "full") & (chunk["condition"] == "harmony")].copy()

    input_height = chunk[["model", "directionality", "dataset", "condition", "run_num"]].copy()
    input_height["v_high"] = chunk["sr_v1_high"].astype("Int64").astype(str) + chunk["sr_v2_high"].astype("Int64").astype(str)
    input_height["v_agree"] = (chunk["sr_v1_high"] == chunk["sr_v2_high"]).astype(int)
    input_height = input_height.groupby(
        ["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"],
        as_index=False,
    ).size().rename(columns={"size": "error_num_input"})
    input_height["error_rate_input"] = input_height.groupby(
        ["model", "directionality", "dataset", "condition", "run_num"]
    )["error_num_input"].transform(lambda s: s / s.sum())

    output_height = chunk[["model", "directionality", "dataset", "condition", "run_num"]].copy()
    output_height["v_high"] = (
        chunk["pred_sr_v1_high"].astype("Int64").astype(str) + chunk["pred_sr_v2_high"].astype("Int64").astype(str)
    )
    output_height["v_agree"] = (chunk["pred_sr_v1_high"] == chunk["pred_sr_v2_high"]).astype(int)
    output_height = output_height.groupby(
        ["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"],
        as_index=False,
    ).size().rename(columns={"size": "error_num_output"})
    output_height["error_rate_output"] = output_height.groupby(
        ["model", "directionality", "dataset", "condition", "run_num"]
    )["error_num_output"].transform(lambda s: s / s.sum())

    all_keys = pd.concat(
        [
            input_height[["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"]],
            output_height[["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    height = all_keys.merge(
        input_height,
        on=["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"],
        how="left",
    ).merge(
        output_height,
        on=["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"],
        how="left",
    )
    for col in ["error_num_input", "error_rate_input", "error_num_output", "error_rate_output"]:
        height[col] = height[col].fillna(0)
    height["error_num_diff"] = height["error_num_output"] - height["error_num_input"]
    height["error_rate_diff"] = height["error_rate_output"] - height["error_rate_input"]
    return height


def clean_cv_pred(base_dir: Path, output_dir: Path) -> None:
    acc, acc_summary = build_cv_acc_summary(base_dir)
    input_files = list_matching_files(base_dir, CV_TRIALS, "*pred.csv")
    output_file = output_dir / "cleaned_260518_EnglishBH_cv_pred.csv"
    output_file.unlink(missing_ok=True)
    first_write = True

    for file_path in input_files:
        for chunk in pd.read_csv(file_path, usecols=CV_PRED_COLUMNS, chunksize=200000):
            chunk = filter_failed_runs(chunk, acc)
            if chunk.empty:
                continue
            summary = summarize_cv_pred_chunk(chunk, acc_summary)
            first_write = append_csv(summary, output_file, first_write, file_path, base_dir)


def clean_v_pred(base_dir: Path, output_dir: Path) -> None:
    acc = read_acc_files(base_dir, V_TRIALS)
    input_files = list_matching_files(base_dir, V_TRIALS, "*pred.csv")
    pred_output = output_dir / "cleaned_260518_EnglishBH_v_pred.csv"
    height_output = output_dir / "cleaned_260518_EnglishBH_v_height.csv"
    pred_output.unlink(missing_ok=True)
    height_output.unlink(missing_ok=True)
    first_pred_write = True
    first_height_write = True

    for file_path in input_files:
        for chunk in pd.read_csv(file_path, chunksize=200000):
            missing_cols = [col for col in V_PRED_COLUMNS if col not in chunk.columns]
            for col in missing_cols:
                chunk[col] = pd.NA
            chunk = chunk[V_PRED_COLUMNS].copy()
            chunk = filter_failed_runs(chunk, acc)
            if chunk.empty:
                continue
            pred_summary = summarize_v_pred_chunk(chunk)
            height_summary = summarize_height_chunk(chunk)
            first_pred_write = append_csv(pred_summary, pred_output, first_pred_write, file_path, base_dir)
            first_height_write = append_csv(height_summary, height_output, first_height_write, file_path, base_dir)


def append_csv(df: pd.DataFrame, output_file: Path, first_write: bool, input_file: Path, base_dir: Path) -> bool:
    if df.empty:
        return first_write
    df.to_csv(output_file, mode="w" if first_write else "a", index=False, header=first_write)
    print(f"wrote {output_file.relative_to(base_dir)} from {input_file.relative_to(base_dir)}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean SubInPhon result CSVs in Python.")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Directory containing the trial folders listed in the original R script.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the cleaned CSV files will be written.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_all_acc(base_dir, output_dir)
    clean_cv_pred(base_dir, output_dir)
    clean_v_pred(base_dir, output_dir)


if __name__ == "__main__":
    main()
