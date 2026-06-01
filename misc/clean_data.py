import gc
from pathlib import Path

import numpy as np
import pandas as pd

RUN_KEY_COLUMNS = [
    "language",
    "modality",
    "directionality",
    "property",
    "condition",
    "run_num",
]
V_REQUIRED_COLUMNS = [
    "v1_error",
    "v2_error",
    "v3_error",
    "sr_v1",
    "sr_v2",
    "sr_v3",
    "pred_sr_v1",
    "pred_sr_v2",
    "pred_sr_v3",
]
CV_REQUIRED_COLUMNS = [
    "v1_error",
    "v2_error",
    "o1_error",
    "o2_error",
    "c1_error",
    "c2_error",
    "pred_sr_v1",
    "pred_sr_v2",
]
HIGH_VOWELS = {"i", "ɪ", "u", "ʊ"}
MID_VOWELS = {"e", "ɛ", "o", "ɔ"}
TENSE_VOWELS = {"i", "u", "e", "o"}
LAX_VOWELS = {"ɪ", "ʊ", "ɛ", "ɔ"}
FRONT_VOWELS = {"i", "ɪ", "e", "ɛ"}
BACK_VOWELS = {"u", "ʊ", "o", "ɔ"}


def filter_included_runs(df: pd.DataFrame, included_run_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or included_run_df.empty:
        return df.copy()

    filtered_df = df.copy()
    filtered_df["property"] = filtered_df["property"].fillna("")
    filtered_df = filtered_df.merge(
        included_run_df[RUN_KEY_COLUMNS].drop_duplicates(),
        on=RUN_KEY_COLUMNS,
        how="inner",
    )
    return filtered_df.reset_index(drop=True)


def append_csv(df: pd.DataFrame, output_file: Path, first_write: bool) -> bool:
    if df.empty:
        return first_write

    df.to_csv(output_file, mode="w" if first_write else "a", index=False, header=first_write)
    del df
    gc.collect()
    return False


def iter_runs_from_pred_file(file_path: Path, chunk_size: int = 200000):
    carryover = pd.DataFrame()

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk["property"] = chunk["property"].fillna("")

        if not carryover.empty:
            chunk = pd.concat([carryover, chunk], ignore_index=True)

        run_keys = chunk[RUN_KEY_COLUMNS].apply(tuple, axis=1)
        last_run_key = run_keys.iloc[-1]
        complete_mask = run_keys != last_run_key

        complete_chunk = chunk.loc[complete_mask].copy()
        carryover = chunk.loc[~complete_mask].copy()

        if complete_chunk.empty:
            continue

        for _, run_df in complete_chunk.groupby(RUN_KEY_COLUMNS, sort=False):
            yield run_df.reset_index(drop=True)

    if not carryover.empty:
        yield carryover.reset_index(drop=True)


def is_included_run(run_df: pd.DataFrame, included_run_df: pd.DataFrame) -> bool:
    if run_df.empty or included_run_df.empty:
        return False

    run_key_df = run_df[RUN_KEY_COLUMNS].drop_duplicates()
    return not run_key_df.merge(
        included_run_df[RUN_KEY_COLUMNS].drop_duplicates(),
        on=RUN_KEY_COLUMNS,
        how="inner",
    ).empty


def relabel_columns(df: pd.DataFrame) -> pd.DataFrame:
    labeled_df = df.copy()
    labeled_df["model"] = labeled_df["modality"].map({"txt": "segment"}).fillna("feature")
    labeled_df["directionality"] = labeled_df["directionality"].map(
        {"l2r": "left-to-right"}
    ).fillna("right-to-left")
    labeled_df["dataset"] = "full"
    labeled_df.loc[labeled_df["language"].str.contains("expanded", na=False), "dataset"] = "expanded"
    labeled_df.loc[labeled_df["property"] == "nonidentical", "dataset"] = "reduced"
    return labeled_df


def ensure_columns(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    normalized_df = df.copy()
    for column in required_columns:
        if column not in normalized_df.columns:
            normalized_df[column] = pd.NA
    return normalized_df


def _map_high(series: pd.Series) -> pd.Series:
    return series.map(lambda value: 1 if value in HIGH_VOWELS else 0 if value in MID_VOWELS else pd.NA)


def _map_tense(series: pd.Series) -> pd.Series:
    return series.map(lambda value: 1 if value in TENSE_VOWELS else 0 if value in LAX_VOWELS else pd.NA)


def _map_back(series: pd.Series) -> pd.Series:
    return series.map(lambda value: 0 if value in FRONT_VOWELS else 1 if value in BACK_VOWELS else pd.NA)


def build_v_high(first: pd.Series, second: pd.Series, directionality: pd.Series) -> pd.Series:
    l2r = first.astype("Int64").astype(str) + second.astype("Int64").astype(str)
    r2l = second.astype("Int64").astype(str) + first.astype("Int64").astype(str)
    return pd.Series(
        np.where(directionality == "right-to-left", r2l, l2r),
        index=first.index,
    )


def clean_v_run_df(run_df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = run_df.copy()

    for prefix in ["sr_v1", "sr_v2", "sr_v3", "pred_sr_v1", "pred_sr_v2", "pred_sr_v3"]:
        cleaned_df[f"{prefix}_high"] = _map_high(cleaned_df[prefix])
        cleaned_df[f"{prefix}_tense"] = _map_tense(cleaned_df[prefix])
        cleaned_df[f"{prefix}_back"] = _map_back(cleaned_df[prefix])

    cleaned_df["high_error"] = 0
    cleaned_df.loc[
        (cleaned_df["dataset"] == "expanded")
        & (
            (cleaned_df["sr_v1_high"] != cleaned_df["pred_sr_v1_high"])
            | (cleaned_df["sr_v2_high"] != cleaned_df["pred_sr_v2_high"])
            | (cleaned_df["sr_v3_high"] != cleaned_df["pred_sr_v3_high"])
        ),
        "high_error",
    ] = 1
    cleaned_df.loc[
        (cleaned_df["dataset"] != "expanded")
        & (
            (cleaned_df["sr_v1_high"] != cleaned_df["pred_sr_v1_high"])
            | (cleaned_df["sr_v2_high"] != cleaned_df["pred_sr_v2_high"])
        ),
        "high_error",
    ] = 1

    cleaned_df["tense_error"] = 0
    cleaned_df.loc[
        (cleaned_df["dataset"] == "expanded")
        & (
            (cleaned_df["sr_v1_tense"] != cleaned_df["pred_sr_v1_tense"])
            | (cleaned_df["sr_v2_tense"] != cleaned_df["pred_sr_v2_tense"])
            | (cleaned_df["sr_v3_tense"] != cleaned_df["pred_sr_v3_tense"])
        ),
        "tense_error",
    ] = 1
    cleaned_df.loc[
        (cleaned_df["dataset"] != "expanded")
        & (
            (cleaned_df["sr_v1_tense"] != cleaned_df["pred_sr_v1_tense"])
            | (cleaned_df["sr_v2_tense"] != cleaned_df["pred_sr_v2_tense"])
        ),
        "tense_error",
    ] = 1

    cleaned_df["back_error"] = 0
    cleaned_df.loc[
        (cleaned_df["dataset"] == "expanded")
        & (
            (cleaned_df["sr_v1_back"] != cleaned_df["pred_sr_v1_back"])
            | (cleaned_df["sr_v2_back"] != cleaned_df["pred_sr_v2_back"])
            | (cleaned_df["sr_v3_back"] != cleaned_df["pred_sr_v3_back"])
        ),
        "back_error",
    ] = 1
    cleaned_df.loc[
        (cleaned_df["dataset"] != "expanded")
        & (
            (cleaned_df["sr_v1_back"] != cleaned_df["pred_sr_v1_back"])
            | (cleaned_df["sr_v2_back"] != cleaned_df["pred_sr_v2_back"])
        ),
        "back_error",
    ] = 1

    pred_back_triplet = (
        cleaned_df["pred_sr_v1_back"].astype("Int64").astype(str)
        + cleaned_df["pred_sr_v2_back"].astype("Int64").astype(str)
        + cleaned_df["pred_sr_v3_back"].astype("Int64").astype(str)
    )
    cleaned_df["harmony_error"] = 0
    cleaned_df.loc[
        (cleaned_df["dataset"] == "expanded")
        & (cleaned_df["condition"] == "harmony")
        & ~pred_back_triplet.isin(["000", "111"]),
        "harmony_error",
    ] = 1
    cleaned_df.loc[
        (cleaned_df["dataset"] == "expanded")
        & (cleaned_df["condition"] == "disharmony")
        & (cleaned_df["directionality"] == "left-to-right")
        & ~pred_back_triplet.isin(["100", "011"]),
        "harmony_error",
    ] = 1
    cleaned_df.loc[
        (cleaned_df["dataset"] == "expanded")
        & (cleaned_df["condition"] == "disharmony")
        & (cleaned_df["directionality"] == "right-to-left")
        & ~pred_back_triplet.isin(["110", "001"]),
        "harmony_error",
    ] = 1
    cleaned_df.loc[
        (cleaned_df["dataset"] != "expanded")
        & (cleaned_df["condition"] == "harmony")
        & (cleaned_df["pred_sr_v1_back"] != cleaned_df["pred_sr_v2_back"]),
        "harmony_error",
    ] = 1
    cleaned_df.loc[
        (cleaned_df["dataset"] != "expanded")
        & (cleaned_df["condition"] == "disharmony")
        & (cleaned_df["pred_sr_v1_back"] == cleaned_df["pred_sr_v2_back"]),
        "harmony_error",
    ] = 1

    return cleaned_df


def clean_cv_run_df(run_df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = run_df.copy()

    cleaned_df["v_error"] = (
        ~((cleaned_df["v1_error"] == 0) & (cleaned_df["v2_error"] == 0))
    ).astype(int)
    cleaned_df["c_error"] = (
        ~(
            (cleaned_df["o1_error"] == 0)
            & (cleaned_df["o2_error"] == 0)
            & (cleaned_df["c1_error"] == 0)
            & (cleaned_df["c2_error"] == 0)
        )
    ).astype(int)

    return cleaned_df


def build_cv_acc_df(filtered_acc_df: pd.DataFrame) -> pd.DataFrame:
    cv_acc_df = filtered_acc_df[filtered_acc_df["error_record"] == "cv"].copy()
    if cv_acc_df.empty:
        return cv_acc_df

    cv_acc_df["total_data"] = 167968
    cv_acc_df["data_split"] = cv_acc_df["subset"].map({"train": 0.8, "test": 0.1})
    cv_acc_df["subset_data"] = cv_acc_df["total_data"] * cv_acc_df["data_split"]
    cv_acc_df["total_error"] = (cv_acc_df["subset_data"] * (1 - cv_acc_df["acc"])).astype(int)
    return cv_acc_df


def build_v_acc_df(filtered_acc_df: pd.DataFrame) -> pd.DataFrame:
    return filtered_acc_df[filtered_acc_df["error_record"] == "v"].copy()


def summarize_v_run(run_df: pd.DataFrame, v_acc_df: pd.DataFrame) -> pd.DataFrame:
    run_acc_df = v_acc_df[
        (v_acc_df["model"] == run_df["model"].iat[0])
        & (v_acc_df["directionality"] == run_df["directionality"].iat[0])
        & (v_acc_df["dataset"] == run_df["dataset"].iat[0])
        & (v_acc_df["condition"] == run_df["condition"].iat[0])
        & (v_acc_df["run_num"] == run_df["run_num"].iat[0])
    ].copy()

    if run_acc_df.empty:
        return pd.DataFrame()

    summary_df = (
        run_df.groupby(
            [
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
            as_index=False,
        )
        .size()
        .rename(columns={"size": "error_num"})
    )

    run_keys = run_acc_df[
        ["model", "directionality", "dataset", "condition", "run_num", "epoch", "subset"]
    ].drop_duplicates()
    error_keys = summary_df[
        ["v1_error", "v2_error", "v3_error", "high_error", "tense_error", "back_error", "harmony_error"]
    ].drop_duplicates()
    summary_df = run_keys.merge(error_keys, how="cross").merge(
        summary_df,
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
    summary_df["error_num"] = pd.to_numeric(summary_df["error_num"], errors="coerce").fillna(0).astype(int)
    return summary_df.reset_index(drop=True)


def summarize_v_height_run(run_df: pd.DataFrame) -> pd.DataFrame:
    harmony_df = run_df[
        (run_df["dataset"] == "full") & (run_df["condition"] == "harmony")
    ].copy()

    if harmony_df.empty:
        return pd.DataFrame()

    input_height_df = harmony_df.copy()
    input_height_df["v_high"] = build_v_high(
        input_height_df["sr_v1_high"],
        input_height_df["sr_v2_high"],
        input_height_df["directionality"],
    )
    input_height_df["v_agree"] = pd.NA
    input_height_df.loc[input_height_df["sr_v1_high"] == input_height_df["sr_v2_high"], "v_agree"] = 1
    input_height_df.loc[input_height_df["sr_v1_high"] != input_height_df["sr_v2_high"], "v_agree"] = 0
    input_height_df = (
        input_height_df.groupby(
            ["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "error_num"})
    )
    input_height_df["error_rate"] = input_height_df.groupby(
        ["model", "directionality", "dataset", "condition", "run_num"]
    )["error_num"].transform(lambda s: s / s.sum() if s.sum() else 0)
    input_run_keys = input_height_df[
        ["model", "directionality", "dataset", "condition", "run_num"]
    ].drop_duplicates()
    input_combos = input_height_df[["v_high", "v_agree"]].drop_duplicates()
    input_height_df = input_run_keys.merge(input_combos, how="cross").merge(
        input_height_df,
        on=["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"],
        how="left",
    )
    input_height_df["error_num"] = pd.to_numeric(input_height_df["error_num"], errors="coerce").fillna(0).astype(int)
    input_height_df["error_rate"] = pd.to_numeric(input_height_df["error_rate"], errors="coerce").fillna(0.0)

    output_height_df = harmony_df.copy()
    output_height_df["v_high"] = (
        build_v_high(
            output_height_df["pred_sr_v1_high"],
            output_height_df["pred_sr_v2_high"],
            output_height_df["directionality"],
        )
    )
    output_height_df["v_agree"] = pd.NA
    output_height_df.loc[
        output_height_df["pred_sr_v1_high"] == output_height_df["pred_sr_v2_high"], "v_agree"
    ] = 1
    output_height_df.loc[
        output_height_df["pred_sr_v1_high"] != output_height_df["pred_sr_v2_high"], "v_agree"
    ] = 0
    output_height_df = (
        output_height_df.groupby(
            ["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "error_num"})
    )
    output_height_df["error_rate"] = output_height_df.groupby(
        ["model", "directionality", "dataset", "condition", "run_num"]
    )["error_num"].transform(lambda s: s / s.sum() if s.sum() else 0)
    output_run_keys = output_height_df[
        ["model", "directionality", "dataset", "condition", "run_num"]
    ].drop_duplicates()
    output_combos = output_height_df[["v_high", "v_agree"]].drop_duplicates()
    output_height_df = output_run_keys.merge(output_combos, how="cross").merge(
        output_height_df,
        on=["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"],
        how="left",
    )
    output_height_df["error_num"] = pd.to_numeric(output_height_df["error_num"], errors="coerce").fillna(0).astype(int)
    output_height_df["error_rate"] = pd.to_numeric(output_height_df["error_rate"], errors="coerce").fillna(0.0)

    v_height_df = input_height_df.merge(
        output_height_df,
        on=["model", "directionality", "dataset", "condition", "run_num", "v_high", "v_agree"],
        how="outer",
        suffixes=("_input", "_output"),
    )
    for column in ["error_num_input", "error_num_output"]:
        v_height_df[column] = pd.to_numeric(v_height_df[column], errors="coerce").fillna(0).astype(int)
    for column in ["error_rate_input", "error_rate_output"]:
        v_height_df[column] = pd.to_numeric(v_height_df[column], errors="coerce").fillna(0.0)
    v_height_df["error_num_diff"] = v_height_df["error_num_output"] - v_height_df["error_num_input"]
    v_height_df["error_rate_diff"] = v_height_df["error_rate_output"] - v_height_df["error_rate_input"]
    return v_height_df.reset_index(drop=True)


def summarize_cv_run(run_df: pd.DataFrame, cv_acc_df: pd.DataFrame) -> pd.DataFrame:
    run_acc_df = cv_acc_df[
        (cv_acc_df["model"] == run_df["model"].iat[0])
        & (cv_acc_df["directionality"] == run_df["directionality"].iat[0])
        & (cv_acc_df["dataset"] == run_df["dataset"].iat[0])
        & (cv_acc_df["condition"] == run_df["condition"].iat[0])
        & (cv_acc_df["run_num"] == run_df["run_num"].iat[0])
    ].copy()

    if run_acc_df.empty:
        return pd.DataFrame()

    summary_df = (
        run_df.groupby(
            ["model", "directionality", "dataset", "condition", "run_num", "epoch", "subset", "c_error", "v_error"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "error_num"})
    )

    run_keys = run_acc_df[
        ["model", "directionality", "dataset", "condition", "run_num", "epoch", "subset"]
    ].drop_duplicates()
    error_combos = pd.MultiIndex.from_product([[0, 1], [0, 1]], names=["c_error", "v_error"]).to_frame(index=False)
    summary_df = run_keys.merge(error_combos, how="cross").merge(
        summary_df,
        on=["model", "directionality", "dataset", "condition", "run_num", "epoch", "subset", "c_error", "v_error"],
        how="left",
    )
    summary_df["error_num"] = pd.to_numeric(summary_df["error_num"], errors="coerce").fillna(0).astype(int)

    summary_df["segment_error"] = summary_df.groupby(
        ["model", "directionality", "dataset", "condition", "run_num", "epoch", "subset"]
    )["error_num"].transform("sum")
    summary_df = summary_df.merge(
        run_acc_df,
        on=["model", "directionality", "dataset", "condition", "run_num", "epoch", "subset"],
        how="left",
    )
    summary_df = summary_df.drop(columns=["language", "error_record", "loss", "acc"])
    summary_df.loc[
        (summary_df["c_error"] == 0) & (summary_df["v_error"] == 0),
        "error_num",
    ] = (
        summary_df.loc[(summary_df["c_error"] == 0) & (summary_df["v_error"] == 0), "total_error"]
        - summary_df.loc[(summary_df["c_error"] == 0) & (summary_df["v_error"] == 0), "segment_error"]
    )
    summary_df["error_type"] = pd.Categorical(
        summary_df.apply(
            lambda row: "consonant only"
            if row["c_error"] == 1 and row["v_error"] == 0
            else "vowel only"
            if row["c_error"] == 0 and row["v_error"] == 1
            else "consonant and vowel"
            if row["c_error"] == 1 and row["v_error"] == 1
            else "syllable structure",
            axis=1,
        ),
        categories=[
            "syllable structure",
            "consonant and vowel",
            "consonant only",
            "vowel only",
        ],
        ordered=True,
    )
    summary_df = summary_df.drop(columns=["c_error", "v_error", "segment_error"])
    summary_df["error_rate"] = summary_df.apply(
        lambda row: 0 if row["total_error"] == 0 else row["error_num"] / row["total_error"],
        axis=1,
    )

    return summary_df[
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
    ].reset_index(drop=True)


def filter_failed_runs(df: pd.DataFrame, failed_run_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or failed_run_df.empty:
        return df.copy()

    filtered_df = df.copy()
    filtered_df["property"] = filtered_df["property"].fillna("")
    filtered_df = filtered_df.merge(
        failed_run_df[RUN_KEY_COLUMNS].drop_duplicates(),
        on=RUN_KEY_COLUMNS,
        how="left",
        indicator=True,
    )
    filtered_df = (
        filtered_df[filtered_df["_merge"] == "left_only"]
        .drop(columns="_merge")
        .reset_index(drop=True)
    )
    return filtered_df


def clean_acc(
    results_dir: Path,
    data_dir: Path,
    min_acc: float,
    max_loss: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    acc_files = sorted(results_dir.rglob("*acc.csv"))

    if not acc_files:
        print(f"No acc.csv files found under: {results_dir}")
        empty_df = pd.DataFrame()
        return empty_df, empty_df

    frames = []
    for file_path in acc_files:
        print(f"Reading acc file: {file_path}")
        this_df = pd.read_csv(file_path)
        top_level_folder = file_path.relative_to(results_dir).parts[0]
        this_df["error_record"] = "cv" if "_cv" in top_level_folder else "v"
        frames.append(this_df)

    all_acc_df = pd.concat(frames, ignore_index=True)
    all_acc_df["property"] = all_acc_df["property"].fillna("")

    failed_run_keys_df = (
        all_acc_df[
            (all_acc_df["epoch"] == 99)
            & (
                (all_acc_df["acc"] < min_acc)
                | (all_acc_df["loss"] > max_loss)
            )
        ][RUN_KEY_COLUMNS]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    failed_run_df = (
        all_acc_df[all_acc_df["epoch"] == 99]
        .merge(failed_run_keys_df, on=RUN_KEY_COLUMNS, how="inner")
        .sort_values(RUN_KEY_COLUMNS + ["record_type"])
        .reset_index(drop=True)
    )
    failed_run_file = data_dir / "cleaned_260531_EnglishBH_failed_run.csv"
    failed_run_df.to_csv(failed_run_file, index=False)
    print(f"Saved {len(failed_run_df)} failed runs to: {failed_run_file}")

    filtered_acc_df = all_acc_df.merge(
        failed_run_keys_df,
        on=RUN_KEY_COLUMNS,
        how="left",
        indicator=True,
    )
    filtered_acc_df = (
        filtered_acc_df[filtered_acc_df["_merge"] == "left_only"]
        .drop(columns="_merge")
        .reset_index(drop=True)
    )
    included_run_df = filtered_acc_df[RUN_KEY_COLUMNS].drop_duplicates().reset_index(drop=True)
    filtered_acc_df = relabel_columns(filtered_acc_df)
    filtered_acc_df = filtered_acc_df.rename(columns={"record_type": "subset"})
    filtered_acc_df = filtered_acc_df[
        [
            "language",
            "model",
            "directionality",
            "dataset",
            "error_record",
            "condition",
            "run_num",
            "epoch",
            "subset",
            "loss",
            "acc",
        ]
    ]
    filtered_acc_file = data_dir / "cleaned_260531_EnglishBH_all_acc.csv"
    filtered_acc_df.to_csv(filtered_acc_file, index=False)

    filtered_run_df = (
        filtered_acc_df[
            [
                "language",
                "model",
                "directionality",
                "dataset",
                "error_record",
                "condition",
                "run_num",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "language",
                "model",
                "directionality",
                "dataset",
                "error_record",
                "condition",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n"})
    )
    filtered_run_file = data_dir / "cleaned_260531_EnglishBH_run_list.csv"
    filtered_run_df.to_csv(filtered_run_file, index=False)

    print(
        f"Combined {len(acc_files)} acc files into {len(all_acc_df)} rows, "
        f"then kept {len(filtered_acc_df)} rows after removing failed runs"
    )
    print(f"Saved filtered acc data to: {filtered_acc_file}")
    print(f"Saved filtered run summary to: {filtered_run_file}")
    return filtered_acc_df, included_run_df


def clean_v_pred(results_dir: Path, data_dir: Path, included_run_df: pd.DataFrame, filtered_acc_df: pd.DataFrame) -> None:
    v_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir() and path.name.endswith("_v"))
    pred_files = sorted(file_path for folder in v_dirs for file_path in folder.rglob("*pred.csv"))
    v_acc_df = build_v_acc_df(filtered_acc_df)
    output_file = data_dir / "cleaned_260531_EnglishBH_v_pred.csv"
    v_height_file = data_dir / "cleaned_260531_EnglishBH_v_height.csv"
    output_file.unlink(missing_ok=True)
    v_height_file.unlink(missing_ok=True)
    first_write = True
    first_v_height_write = True

    if not pred_files:
        print(f"No pred.csv files found under v folders in: {results_dir}")
        return

    for file_path in pred_files:
        print(f"Reading v pred file: {file_path}")
        for run_df in iter_runs_from_pred_file(file_path):
            if not is_included_run(run_df, included_run_df):
                continue
            run_df = ensure_columns(run_df, V_REQUIRED_COLUMNS)
            run_df = relabel_columns(run_df)
            run_df = run_df.rename(columns={"record_type": "subset"})
            run_df = clean_v_run_df(run_df)
            summary_df = summarize_v_run(run_df, v_acc_df)
            v_height_df = summarize_v_height_run(run_df)
            first_write = append_csv(summary_df, output_file, first_write)
            first_v_height_write = append_csv(v_height_df, v_height_file, first_v_height_write)
            if not summary_df.empty:
                print(
                    "Appended v summary for "
                    f"language={run_df['language'].iat[0]}, "
                    f"model={run_df['model'].iat[0]}, "
                    f"directionality={run_df['directionality'].iat[0]}, "
                    f"dataset={run_df['dataset'].iat[0]}, "
                    f"condition={run_df['condition'].iat[0]}, "
                    f"run_num={run_df['run_num'].iat[0]}"
                )
            if not v_height_df.empty:
                print(
                    "Appended v height summary for "
                    f"language={run_df['language'].iat[0]}, "
                    f"model={run_df['model'].iat[0]}, "
                    f"directionality={run_df['directionality'].iat[0]}, "
                    f"dataset={run_df['dataset'].iat[0]}, "
                    f"condition={run_df['condition'].iat[0]}, "
                    f"run_num={run_df['run_num'].iat[0]}"
                )

    print(f"Saved v summary data to: {output_file}")
    print(f"Saved v height data to: {v_height_file}")


def clean_cv_pred(results_dir: Path, data_dir: Path, included_run_df: pd.DataFrame, filtered_acc_df: pd.DataFrame) -> None:
    cv_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir() and path.name.endswith("_cv"))
    pred_files = sorted(file_path for folder in cv_dirs for file_path in folder.rglob("*pred.csv"))
    cv_acc_df = build_cv_acc_df(filtered_acc_df)
    output_file = data_dir / "cleaned_260531_EnglishBH_cv_pred.csv"
    output_file.unlink(missing_ok=True)
    first_write = True

    if not pred_files:
        print(f"No pred.csv files found under cv folders in: {results_dir}")
        return

    for file_path in pred_files:
        print(f"Reading cv pred file: {file_path}")
        for run_df in iter_runs_from_pred_file(file_path):
            if not is_included_run(run_df, included_run_df):
                continue
            run_df = ensure_columns(run_df, CV_REQUIRED_COLUMNS)
            run_df = relabel_columns(run_df)
            run_df = run_df.rename(columns={"record_type": "subset"})
            run_df = clean_cv_run_df(run_df)
            summary_df = summarize_cv_run(run_df, cv_acc_df)
            first_write = append_csv(summary_df, output_file, first_write)
            if not summary_df.empty:
                print(
                    "Appended cv summary for "
                    f"language={run_df['language'].iat[0]}, "
                    f"model={run_df['model'].iat[0]}, "
                    f"directionality={run_df['directionality'].iat[0]}, "
                    f"dataset={run_df['dataset'].iat[0]}, "
                    f"condition={run_df['condition'].iat[0]}, "
                    f"run_num={run_df['run_num'].iat[0]}"
                )

    print(f"Saved cv summary data to: {output_file}")


def main() -> None:
    # Edit these paths as needed for a given run.
    results_dir = Path("/media/ldlmdl/A2AAE4B1AAE482E1/SSD_Documents/subinphon/results")
    data_dir = Path("/media/ldlmdl/A2AAE4B1AAE482E1/SSD_Documents/subinphon/data")
    min_acc = 0.8
    max_loss = 0.1

    results_dir = results_dir.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()

    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading files from: {results_dir}")
    print(f"Saving files to: {data_dir}")

    filtered_acc_df, included_run_df = clean_acc(results_dir, data_dir, min_acc, max_loss)
    clean_v_pred(results_dir, data_dir, included_run_df, filtered_acc_df)
    clean_cv_pred(results_dir, data_dir, included_run_df, filtered_acc_df)


if __name__ == "__main__":
    main()
