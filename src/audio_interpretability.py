"""Extract final-checkpoint vowel representations for classification and RSA."""

import base64
import hashlib
import html
import json
import os
from pathlib import Path
import random
import re
import tempfile
import warnings

import numpy as np
from plotly.offline import get_plotlyjs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Subset

import hyper_params as hp


REPRESENTATIONS = {
    "encoder_hidden": "Encoder hidden",
    "decoder_context": "Decoder attention context",
    "decoder_hidden": "Decoder LSTM hidden",
    "synthesizer_input": "Synthesizer context / upsampled representation",
    "synthesizer_hidden": "Synthesizer LSTM hidden",
}

CLASSIFICATION_TASKS = {
    "V1_backness": "Underlying V1 backness",
    "V1_height": "Underlying V1 height",
    "V2_backness": "Underlying V2 backness",
    "V2_height": "Underlying V2 height",
}


BASELINES = {
    "input_backness": "Input backness", "target_backness": "Target backness",
    "predicted_backness": "Predicted backness", "input_acoustic": "Input mean spectrum",
    "target_acoustic": "Target mean spectrum", "predicted_acoustic": "Predicted mean spectrum",
}


def _stimulus_keys(dataset, indices):
    counts, keys = {}, []
    for index in indices:
        row = dataset.annotations.iloc[index]
        identity = json.dumps([str(row[key]) for key in ("ur_var", "sr_var", "ur_string", "sr_string")], ensure_ascii=False)
        occurrence = counts.get(identity, 0)
        counts[identity] = occurrence + 1
        keys.append(identity + f"#{occurrence}")
    return keys


def _check_test_match(left, right):
    if sorted(left["test_stimuli"]) != sorted(right["test_stimuli"]):
        raise ValueError(f"T1/T2 test stimuli do not match: {left['trial']} / {right['trial']}, "
                         f"condition {left['condition']}, run {left['run']}")


def _same_run(left, right):
    return all(left[key] == right[key] for key in ("language", "condition", "direction", "run"))



def _classification_split(recorder, indices):
    words = [recorder.dataset.annotations.iloc[index].ur_string for index in indices]
    unique_words = sorted(set(words))
    features = {}
    for word in unique_words:
        vowels = [char for char in word if char in recorder.language.vowel]
        if len(vowels) == 2:
            features[word] = [
                recorder.language.vowel[vowel][feature]
                for vowel in vowels for feature in (0, 1)
            ]
    eligible_words = [word for word in unique_words if word in features]
    labels = {}
    targets = np.full((len(words), 4), -1, dtype=int)
    for task_index, task in enumerate(CLASSIFICATION_TASKS):
        classes = sorted({features[word][task_index] for word in eligible_words})
        labels[task] = classes
        lookup = {label: i for i, label in enumerate(classes)}
        for i, word in enumerate(words):
            if word in features:
                targets[i, task_index] = lookup[features[word][task_index]]
    word_folds = {}
    strategy = "Unavailable: fewer than five eligible underlying word types"
    if len(eligible_words) >= 5:
        strata = ["|".join(features[word]) for word in eligible_words]
        _, counts = np.unique(strata, return_counts=True)
        if counts.min() >= 5:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=hp.base_seed)
            strategy = "Five folds stratified by the four underlying vowel features"
        else:
            splitter = KFold(n_splits=5, shuffle=True, random_state=hp.base_seed)
            strategy = "Five shuffled word folds; joint feature strata too small for stratification"
        for fold, (_, test) in enumerate(splitter.split(eligible_words, strata)):
            for index in test:
                word_folds[eligible_words[index]] = fold
    folds = np.array([word_folds.get(word, -1) for word in words], dtype=int)
    manifest = {
        "seed": hp.base_seed, "fold_count": 5, "strategy": strategy,
        "classes": labels,
        "words": [{"ur": word, "fold": word_folds.get(word, -1), "features": features.get(word)} for word in unique_words],
    }
    manifest["split_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
    return targets, folds, manifest


def _classify_vowels(vectors, rows, targets, folds):
    results = {}
    for name, values in vectors.items():
        results[name] = {}
        for task_index, task in enumerate(CLASSIFICATION_TASKS):
            position = task.split("_")[0]
            selected = np.array([i for i, row in enumerate(rows) if row["position"] == position], dtype=int)
            word_indices = np.array([rows[i]["test_index"] for i in selected], dtype=int)
            x = values[selected]
            y = targets[word_indices, task_index]
            word_folds = folds[word_indices]
            valid = np.isfinite(x).all(axis=1) & (word_folds >= 0) & (y >= 0)
            predictions = np.full(len(y), -1, dtype=int)
            folds_used = 0
            for fold in range(5):
                train = valid & (word_folds != fold)
                test = valid & (word_folds == fold)
                if not test.any() or len(np.unique(y[train])) != 2:
                    continue
                scaler = StandardScaler()
                train_values = scaler.fit_transform(x[train])
                test_values = scaler.transform(x[test])
                classifier = LogisticRegression(
                    C=1.0, class_weight="balanced", solver="liblinear",
                    max_iter=1000, random_state=hp.base_seed,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    classifier.fit(train_values, y[train])
                if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                    continue
                predictions[test] = classifier.predict(test_values)
                folds_used += 1
            evaluated = valid & (predictions >= 0)
            score = None
            if valid.any() and evaluated.sum() == valid.sum() and len(np.unique(y[evaluated])) == 2:
                score = float(balanced_accuracy_score(y[evaluated], predictions[evaluated]))
            results[name][task] = {
                "score": score, "available": int(valid.sum()), "evaluated": int(evaluated.sum()),
                "folds_used": folds_used,
                "class_counts": [int((y[evaluated] == value).sum()) for value in (0, 1)],
            }
    return results


def _row_indices(subset, dataset):
    indices = np.arange(len(subset))
    while isinstance(subset, Subset):
        indices = np.asarray(subset.indices)[indices]
        subset = subset.dataset
    if subset is not dataset:
        raise ValueError("The evaluation loader must reference the recorder's dataset")
    return indices.tolist()


def _reference_intervals(recorder, row, side, cache):
    word = row[f"{side}_string"]
    ref = row[f"{side}_var"]
    path = Path(hp.audio_root) / (hp.lang_name + "_segmented") / (ref + ".TextGrid")
    if path not in cache:
        cache[path] = recorder.read_vowel_intervals(str(path))
    intervals = cache[path]
    if len(intervals) != 2:
        raise ValueError(f"Expected two vowel intervals in {path}")
    if recorder.validate_vowel_label([item[2] for item in intervals], word) != [
        char for char in word if char in recorder.language.vowel
    ]:
        raise ValueError(f"Vowel labels do not match {word} in {path}")
    return intervals


def _pool_interval(sequence, interval, acoustic_mask):
    start, end, _ = interval
    if not (np.isfinite(start) and np.isfinite(end) and 0 <= start < end):
        raise ValueError(f"Invalid vowel interval: {interval}")
    left = int(round(start * hp.sample_rate / hp.hop_length))
    right = int(round(end * hp.sample_rate / hp.hop_length))
    right = max(left + 1, right)
    if left >= len(sequence) or right > len(sequence):
        raise ValueError(f"Vowel interval exceeds the acoustic window: {interval}")
    selected = acoustic_mask[left:right]
    if not selected.any():
        raise ValueError(f"Vowel interval contains no non-padding frames: {interval}")
    return sequence[left:right][selected].mean(dim=0).numpy()


def evaluate_representations(model, recorder, dataloader, checkpoint, report_file):
    """Extract and classify final-checkpoint vowels without teacher forcing."""
    dataset = recorder.dataset
    architecture = "t2" if hasattr(model, "gaussian_upsample") else "t1"
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    match = re.search(r"_epoch(-?\d+)_seq2seq\.pth$", checkpoint.name)
    if match is None:
        raise ValueError(f"Cannot parse checkpoint epoch: {checkpoint}")
    final_epoch = int(match.group(1))
    indices = _row_indices(dataloader.dataset, dataset)
    if not indices:
        raise ValueError("No test examples to analyze")
    targets, folds, split = _classification_split(recorder, indices)
    stimulus_keys = _stimulus_keys(dataset, indices)
    report_file = Path(report_file).resolve()
    archive_dir = report_file.parent / (report_file.stem + "_vowels_data")
    current = {"architecture": architecture, "trial": recorder.trial_num, "language": recorder.lang_name,
               "condition": recorder.condition, "direction": recorder.directionality,
               "run": recorder.run_num, "test_stimuli": stimulus_keys}
    for archive in archive_dir.glob("*.npz"):
        with np.load(archive, allow_pickle=False) as saved:
            previous = json.loads(str(saved["metadata"].item()))
        if previous.get("rsa_schema") and previous["architecture"] != architecture and _same_run(current, previous):
            _check_test_match(current, previous)
    batch_size = hp.batch_size
    base_collate = dataset._get_collate_fn()

    def collate(batch):
        count = len(batch)
        padded_batch = batch + [batch[-1]] * (batch_size - count)
        return base_collate(padded_batch), count

    loader = DataLoader(
        dataloader.dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        collate_fn=collate, generator=torch.Generator().manual_seed(hp.base_seed),
    )
    captures = {name: [] for name in REPRESENTATIONS}
    vectors = {name: [] for name in REPRESENTATIONS}
    acoustic_vectors = {name: [] for name in ("input_acoustic", "target_acoustic", "predicted_acoustic")}
    metadata, skipped, handles, interval_cache = [], [], [], {}
    classification = {
        "method": "L2 logistic regression, C=1, balanced class weights; within-fold standardization",
        "score": "Balanced accuracy of pooled out-of-fold predictions; chance = 0.5",
        "fold_count": 5, "split_id": split["split_id"], "split": split,
        "padding": "Source mask for encoder; target mask for acoustic synthesis; target PAD/SOS/EOS excluded from decoder steps",
    }

    def capture(name, select):
        def hook(module, args, output):
            captures[name].append(select(output).detach().cpu().clone())
        return hook

    modules = {
        "encoder_hidden": (model.encoder.rnn, lambda output: output[0]),
        "decoder_context": (
            model.attention if architecture == "t2" else model.single_attention,
            lambda output: output[0],
        ),
        "decoder_hidden": (model.decoder.rnn, lambda output: output[0]),
        "synthesizer_input": (
            model.gaussian_upsample if architecture == "t2" else model.multi_attention,
            lambda output: output[0],
        ),
        "synthesizer_hidden": (model.synthesizer.rnn, lambda output: output[0]),
    }
    previous_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    previous_mode = model.training
    random_state = random.getstate()
    try:
        model.eval()
        for name, (module, select) in modules.items():
            handles.append(module.register_forward_hook(capture(name, select)))
        print(f"Interpretability epoch {final_epoch}: extracting {len(indices)} test words", flush=True)
        model.load_state_dict(torch.load(checkpoint, map_location=next(model.parameters()).device, weights_only=True))
        offset = 0
        dimensions = {}
        with torch.no_grad():
            for batch, count in loader:
                for values in captures.values():
                    values.clear()
                output = model(batch, 0.0, 0.0)
                predicted_text = output[1].detach().cpu()
                predicted_audio = output[2].detach().cpu()
                sequences = {}
                for name, values in captures.items():
                    if not values:
                        raise RuntimeError(f"No representations captured for {name}")
                    if name == "synthesizer_input" and architecture == "t2":
                        sequences[name] = values[0]
                    else:
                        sequences[name] = torch.cat(values, dim=0).transpose(0, 1)
                    dimensions[name] = sequences[name].shape[2]
                source_mask = batch[1].detach().cpu().ne(-100).any(dim=2).squeeze(1)
                target_mask = batch[3].detach().cpu().ne(-100).any(dim=2).squeeze(1)
                for item in range(count):
                    row_index = int(indices[offset + item])
                    row = dataset.annotations.iloc[row_index]
                    try:
                        underlying = [char for char in row.ur_string if char in recorder.language.vowel]
                        target_positions = [i for i, char in enumerate(row.sr_string) if char in recorder.language.vowel]
                        if len(underlying) != 2 or len(target_positions) != 2:
                            raise ValueError("This analysis requires exactly two vowels per word")
                        source_intervals = _reference_intervals(recorder, row, "ur", interval_cache)
                        target_intervals = _reference_intervals(recorder, row, "sr", interval_cache)
                        pooled, acoustic_pooled = [], []
                        for vowel_index in range(2):
                            position = target_positions[vowel_index]
                            if position >= sequences["decoder_hidden"].shape[1]:
                                raise ValueError("Target vowel position exceeds decoder sequence")
                            values = {
                                "encoder_hidden": _pool_interval(sequences["encoder_hidden"][item], source_intervals[vowel_index], source_mask[item]),
                                "decoder_context": sequences["decoder_context"][item, position].numpy(),
                                "decoder_hidden": sequences["decoder_hidden"][item, position].numpy(),
                                "synthesizer_input": _pool_interval(sequences["synthesizer_input"][item], target_intervals[vowel_index], target_mask[item]),
                                "synthesizer_hidden": _pool_interval(sequences["synthesizer_hidden"][item], target_intervals[vowel_index], target_mask[item]),
                            }
                            if not all(np.isfinite(value).all() for value in values.values()):
                                raise ValueError("Non-finite representation")
                            pooled.append(values)
                            acoustic_pooled.append({
                                "input_acoustic": _pool_interval(batch[1][item, 0].detach().cpu().transpose(0, 1), source_intervals[vowel_index], source_mask[item]),
                                "target_acoustic": _pool_interval(batch[3][item, 0].detach().cpu().transpose(0, 1), target_intervals[vowel_index], target_mask[item]),
                                "predicted_acoustic": _pool_interval(predicted_audio[item, 0].transpose(0, 1), target_intervals[vowel_index], target_mask[item]),
                            })
                    except (OSError, ValueError, RuntimeError) as error:
                        skipped.append({"dataset_index": row_index, "ur": row.ur_string, "reason": str(error)})
                        continue
                    for vowel_index, values in enumerate(pooled):
                        vowel = underlying[vowel_index]
                        backness, height, _ = recorder.language.vowel[vowel]
                        position = target_positions[vowel_index]
                        predicted_token = dataset.sr_alphabet.idx2char[int(predicted_text[position + 1, item])]
                        target_vowel = row.sr_string[position]
                        predicted_features = recorder.language.vowel.get(predicted_token)
                        metadata.append({
                            "dataset_index": row_index, "test_index": offset + item, "ur": row.ur_string,
                            "sr": row.sr_string, "stimulus_key": stimulus_keys[offset + item],
                            "target_backness": recorder.language.vowel[target_vowel][0],
                            "predicted_token": predicted_token,
                            "predicted_backness": predicted_features[0] if predicted_features else None,
                            "position": "V" + str(vowel_index + 1),
                            "vowel": vowel, "backness": backness, "height": height,
                        })
                        for name, value in values.items():
                            vectors[name].append(value)
                        for name, value in acoustic_pooled[vowel_index].items():
                            acoustic_vectors[name].append(value)
                offset += count
    finally:
        for handle in handles:
            handle.remove()
        model.load_state_dict(previous_weights)
        model.train(previous_mode)
        random.setstate(random_state)

    arrays = {
        name: np.stack(values).astype(np.float32) if values else np.empty((0, dimensions[name]), dtype=np.float32)
        for name, values in vectors.items()
    }
    classification["representations"] = _classify_vowels(arrays, metadata, targets, folds)
    info = {
        "schema": "final_vowel_classification_v1", "epoch": final_epoch,
        "rsa_schema": "vowel_rsa_v1", "test_stimuli": stimulus_keys,
        "architecture": architecture, "trial": recorder.trial_num,
        "language": recorder.lang_name, "condition": recorder.condition,
        "direction": recorder.directionality, "run": recorder.run_num,
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "teacher_forcing": {"text": 0.0, "audio": 0.0},
        "test_items": len(indices), "included_words": len(metadata) // 2,
        "skipped": skipped, "rows": metadata,
        "alignment": "Reference source/target TextGrid vowel means; reference target text positions",
    }
    arrays.update({name: np.stack(values).astype(np.float32) if values else np.empty((0, hp.n_mels), dtype=np.float32)
                   for name, values in acoustic_vectors.items()})
    rsa = _rsa_run(arrays, metadata)
    archive_dir.mkdir(parents=True, exist_ok=True)
    identity = {key: info[key] for key in ("architecture", "trial", "language", "condition", "direction", "run")}
    archive = archive_dir / (hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20] + ".npz")
    with tempfile.NamedTemporaryFile(dir=archive_dir, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(
            temporary_path, metadata=np.array(json.dumps(info)),
            classification=np.array(json.dumps(classification, allow_nan=False)),
            rsa=np.array(json.dumps(rsa, allow_nan=False)), **arrays,
        )
        os.replace(temporary_path, archive)
    finally:
        temporary_path.unlink(missing_ok=True)
    write_report(archive_dir, report_file)
    print(f"Interpretability: final epoch {final_epoch}, {info['included_words']}/{len(indices)} words classified; {report_file}")
    return {
        "report": str(report_file), "archive": str(archive),
        "included_words": info["included_words"], "skipped_words": len(skipped),
        "epoch": final_epoch, "split_id": split["split_id"],
    }


def _mean_scores(groups, representation, task, architecture, condition):
    matching = [group for group in groups if group["info"]["architecture"] == architecture
                and group["info"]["condition"] == condition]
    scored = [(group, group["classification"]["representations"][representation][task]["score"])
              for group in matching]
    scored = [(group, score) for group, score in scored if score is not None]
    return {
        "mean": float(np.mean([score for _, score in scored])) if scored else None,
        "runs": len(scored), "total_runs": len(matching),
        "details": "; ".join(f"{group['info']['trial']} / run {group['info']['run']}: {score:.3f}"
                             for group, score in scored),
    }


def rebuild_report(report_file=None):
    """Render the shared HTML from cached final-vowel classification results."""
    report_file = Path(hp.interpretability_report if report_file is None else report_file).resolve()
    write_report(report_file.parent / (report_file.stem + "_vowels_data"), report_file)
    return str(report_file)


def write_report(archive_dir, report_file):
    """Plot mean per-run accuracies for both architectures and all trials."""
    groups = []
    for archive in sorted(Path(archive_dir).glob("*.npz")):
        with np.load(archive, allow_pickle=False) as saved:
            info = json.loads(str(saved["metadata"].item()))
            if info.get("schema") != "final_vowel_classification_v1":
                continue
            group = {"info": info, "classification": json.loads(str(saved["classification"].item()))}
            if info.get("rsa_schema"):
                group["vectors"] = {name: saved[name].copy() for name in (*REPRESENTATIONS, "input_acoustic", "target_acoustic", "predicted_acoustic")}
                group["rsa"] = json.loads(str(saved["rsa"].item()))
            groups.append(group)
    groups.sort(key=lambda group: (group["info"]["architecture"], group["info"]["condition"],
                                   str(group["info"]["trial"]), group["info"]["run"]))
    rsa_html = _rsa_report(groups)
    conditions = sorted({group["info"]["condition"] for group in groups})
    plots = []
    for task, label in CLASSIFICATION_TASKS.items():
        titles = []
        for name, representation in REPRESENTATIONS.items():
            for architecture in ("t1", "t2"):
                title = representation
                if name == "synthesizer_input":
                    title = "Synthesizer attention context" if architecture == "t1" else "Upsampled representation"
                titles.append(f"{architecture.upper()} · {title}")
        figure = make_subplots(rows=5, cols=2, subplot_titles=titles, vertical_spacing=0.07)
        for row, name in enumerate(REPRESENTATIONS, 1):
            for column, architecture in enumerate(("t1", "t2"), 1):
                for index, condition in enumerate(conditions):
                    stats = _mean_scores(groups, name, task, architecture, condition)
                    figure.add_trace(go.Bar(
                        x=[condition], y=[stats["mean"]], name=condition, legendgroup=condition,
                        showlegend=row == 1 and column == 1,
                        marker_color=("#2469b2", "#d87727")[index % 2],
                        text=[f"{stats['mean']:.3f}" if stats["mean"] is not None else ""], textposition="outside",
                        customdata=[[stats["runs"], stats["total_runs"], html.escape(stats["details"])]],
                        hovertemplate=("%{x}<br>Mean balanced accuracy: %{y:.3f}<br>Scored runs: %{customdata[0]} / %{customdata[1]}"
                                       "<br>%{customdata[2]}<extra></extra>"),
                    ), row=row, col=column)
                figure.add_hline(y=0.5, line_dash="dot", line_color="#89919c", row=row, col=column)
                figure.update_yaxes(range=[0, 1.08], title_text="Balanced accuracy" if column == 1 else None,
                                    row=row, col=column)
                figure.update_xaxes(categoryorder="array", categoryarray=conditions, row=row, col=column)
        figure.update_layout(title=label, height=1450, barmode="group", template="plotly_white",
                             margin=dict(t=100, b=50), legend=dict(orientation="h", y=1.05))
        plots.append(figure.to_html(full_html=False, include_plotlyjs=False, div_id=task,
                                    config={"responsive": True, "displaylogo": False}))
    rows = []
    for group in groups:
        info = group["info"]
        values = [info[key] for key in ("architecture", "trial", "language", "condition", "run", "epoch")]
        values += [f"{info['included_words']} / {info['test_items']}", group["classification"]["split_id"]]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>")
    excluded = [f"{group['info']['architecture']} / {group['info']['trial']} / {group['info']['condition']} / run {group['info']['run']}: "
                f"{item['dataset_index']} {item['ur']} — {item['reason']}"
                for group in groups for item in group["info"]["skipped"]]
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Vowel classification and RSA</title>
<style>body{font:15px system-ui;margin:28px;color:#202630;background:#f7f8fa}section{background:white;margin:24px 0}
p{max-width:1000px;line-height:1.6}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd2da;text-align:left}
pre{white-space:pre-wrap} .table{overflow-x:auto}</style><h1>Final-checkpoint vowel classification and RSA</h1>
<p>Each figure has five representation rows and T1/T2 columns. Bars show the arithmetic mean of per-run balanced accuracies
for each condition across all included trials. Classifiers are fitted separately for each run and condition using fixed five-fold
word-type cross-validation within the original test set. Repeated underlying words share a fold; standardization is fitted on training folds only.
The classifier is L2 logistic regression (C=1, balanced class weights). The dotted line marks chance (0.5).
Hover over bars for contributing run scores and counts. Unavailable scores are excluded from means; missing results have no bar.</p>
<p>Only the requested final checkpoint is loaded, without text or audio teacher forcing. Each vowel predicts its own underlying
backness or height. Acoustic vectors are averaged over reference vowel intervals after excluding -100 padding frames;
decoder vectors use reference target-text vowel positions. Reference locations are also used for incorrect predictions.
T1/T2 trials need not have matching identifiers. No direction selector is used.</p>
<script>__PLOTLY__</script>__PLOTS____RSA__
<h2>Included runs</h2><div class="table"><table><thead><tr><th>Model</th><th>Trial</th><th>Language</th><th>Condition</th><th>Run</th>
<th>Epoch</th><th>Included / test words</th><th>Split ID</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<details><summary>Excluded words</summary><pre>__EXCLUDED__</pre></details></html>'''
    document = document.replace("__PLOTLY__", get_plotlyjs()).replace("__PLOTS__", "".join(f"<section>{plot}</section>" for plot in plots))
    document = document.replace("__RSA__", rsa_html)
    document = document.replace("__ROWS__", "".join(rows)).replace("__EXCLUDED__", html.escape("\n".join(excluded) or "None"))
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    Path(report_file).write_text(document, encoding="utf-8")


def _vowel_rdms(vectors, rows, position):
    indices = sorted((i for i, row in enumerate(rows) if row["position"] == position),
                     key=lambda i: rows[i]["stimulus_key"])
    selected = [rows[i] for i in indices]
    values = {name: array[indices] for name, array in vectors.items()}
    for name, field in (("input_backness", "backness"), ("target_backness", "target_backness"),
                        ("predicted_backness", "predicted_backness")):
        labels = [row[field] for row in selected]
        classes = sorted({label for label in labels if label is not None})
        values[name] = np.array([classes.index(label) if label is not None else np.nan for label in labels], dtype=float).reshape(-1, 1)
    return values, selected


def _rank_rdm(values, metric):
    distances = pdist(values, metric=metric)
    ranks = rankdata(distances)
    ranks -= ranks.mean() if len(ranks) else 0
    norm = float(np.linalg.norm(ranks))
    return ranks, norm


def _correlate_ranks(left, right, count, total):
    result = {"rho": None, "stimuli": count, "excluded": total - count, "pairs": count * (count - 1) // 2}
    if count >= 3 and left[1] > 0 and right[1] > 0:
        result["rho"] = float(np.clip(np.dot(left[0], right[0]) / (left[1] * right[1]), -1, 1))
    return result


def _rsa_run(vectors, rows):
    results = {}
    for position in ("V1", "V2"):
        values, selected = _vowel_rdms(vectors, rows, position)
        cache = {}
        results[position] = {}
        for baseline in BASELINES:
            valid = np.isfinite(values[baseline]).all(axis=1)
            metric = "hamming" if baseline.endswith("backness") else "euclidean"
            reference = _rank_rdm(values[baseline][valid], metric)
            for name in REPRESENTATIONS:
                key = (name, valid.tobytes())
                if key not in cache:
                    cache[key] = _rank_rdm(values[name][valid], "euclidean")
                results[position].setdefault(name, {})[baseline] = _correlate_ranks(
                    cache[key], reference, int(valid.sum()), len(selected))
    return results


def _cross_model_rsa(left, right):
    _check_test_match(left["info"], right["info"])
    results = {}
    for position in ("V1", "V2"):
        left_values, left_rows = _vowel_rdms(left["vectors"], left["info"]["rows"], position)
        right_values, right_rows = _vowel_rdms(right["vectors"], right["info"]["rows"], position)
        if [row["stimulus_key"] for row in left_rows] != [row["stimulus_key"] for row in right_rows]:
            raise ValueError(f"T1/T2 included vowel stimuli do not match for {position}, "
                             f"condition {left['info']['condition']}, run {left['info']['run']}")
        results[position] = {
            name: _correlate_ranks(_rank_rdm(left_values[name], "euclidean"),
                                   _rank_rdm(right_values[name], "euclidean"), len(left_rows), len(left_rows))
            for name in REPRESENTATIONS
        }
    return results


def _rsa_average(items):
    valid = [item for item in items if item["rho"] is not None]
    return {
        "mean": float(np.mean([item["rho"] for item in valid])) if valid else None,
        "runs": len(valid), "total": len(items),
        "stimuli": ", ".join(str(item["stimuli"]) for item in items),
        "excluded": ", ".join(str(item["excluded"]) for item in items),
    }


def _packed_vectors(values):
    values = np.asarray(values, dtype="<f4")
    return {"shape": list(values.shape), "data": base64.b64encode(values.tobytes()).decode("ascii")}


def _rsa_report(groups):
    available = [group for group in groups if "rsa" in group]
    if not available:
        return "<h2>Vowel RSA</h2><p>No RSA results yet. Run interpretability for both architectures to collect the new baselines.</p>"
    paired = []
    for left in available:
        if left["info"]["architecture"] != "t1":
            continue
        for right in available:
            if right["info"]["architecture"] == "t2" and _same_run(left["info"], right["info"]):
                paired.append({"condition": left["info"]["condition"], "results": _cross_model_rsa(left, right)})
    plots = []
    for condition in sorted({group["info"]["condition"] for group in available}):
        for position in ("V1", "V2"):
            means, hover, labels = [], [], []
            for architecture in ("t1", "t2"):
                selected = [group for group in available if group["info"]["architecture"] == architecture
                            and group["info"]["condition"] == condition]
                for name, label in REPRESENTATIONS.items():
                    stats = [_rsa_average([group["rsa"][position][name][baseline] for group in selected]) for baseline in BASELINES]
                    labels.append(f"{architecture.upper()} · {label}")
                    means.append([item["mean"] for item in stats])
                    hover.append([[item["runs"], item["total"], item["stimuli"], item["excluded"]] for item in stats])
            figure = go.Figure(go.Heatmap(
                z=means, x=list(BASELINES.values()), y=labels, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                customdata=hover, colorbar=dict(title="Spearman rho"), hoverongaps=False,
                hovertemplate=("%{y}<br>%{x}<br>Mean rho: %{z:.3f}<br>Estimable runs: %{customdata[0]} / %{customdata[1]}"
                               "<br>Stimuli per run: %{customdata[2]}<br>Excluded per run: %{customdata[3]}<extra></extra>")))
            figure.update_layout(title=f"{html.escape(condition)} · {position}: representation–baseline RSA",
                                 height=650, margin=dict(l=320, b=150), yaxis=dict(autorange="reversed"))
            plots.append(figure.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True}))
            selected_pairs = [pair for pair in paired if pair["condition"] == condition]
            stats = [_rsa_average([pair["results"][position][name] for pair in selected_pairs]) for name in REPRESENTATIONS]
            figure = go.Figure(go.Bar(
                x=list(REPRESENTATIONS.values()), y=[item["mean"] for item in stats],
                customdata=[[item["runs"], item["total"], item["stimuli"]] for item in stats],
                hovertemplate="%{x}<br>Mean rho: %{y:.3f}<br>Estimable run pairs: %{customdata[0]} / %{customdata[1]}<br>Stimuli per pair: %{customdata[2]}<extra></extra>"))
            figure.update_layout(title=f"{html.escape(condition)} · {position}: corresponding T1–T2 components",
                                 yaxis=dict(title="Spearman rho", range=[-1, 1]), height=450, margin=dict(b=130))
            plots.append(figure.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True}))
    display = []
    diagnostics = []
    for group in available:
        info = group["info"]
        entry = {"label": f"{info['architecture'].upper()} · {info['trial']} · {info['condition']} · run {info['run']}", "positions": {}}
        for position in ("V1", "V2"):
            values, rows = _vowel_rdms(group["vectors"], info["rows"], position)
            entry["positions"][position] = {
                "labels": [f"{row['ur']} → {row['sr']} · {row['stimulus_key']}" for row in rows],
                "vectors": {name: _packed_vectors(value) for name, value in values.items()},
            }
            undefined = sum(row["predicted_backness"] is None for row in rows)
            diagnostics.append(f"{entry['label']} · {position}: {len(rows)} included vowels; {undefined} undefined predicted-backness tokens excluded only from predicted-backness RSA.")
        display.append(entry)
    payload = json.dumps(display, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    names = json.dumps({**BASELINES, **REPRESENTATIONS})
    viewer = r'''<h2>Individual vowel RDMs</h2>
<p>Choose a model/run, vowel, and RDM. Rows and columns are the same canonically ordered test stimuli. Undefined predicted backness appears blank.
Backness distance is 0 for equal classes and 1 otherwise; acoustic and hidden-vector distances are Euclidean.</p>
<div><label>Model / trial / condition / run <select id="rdm-run"></select></label>
<label>Vowel <select id="rdm-vowel"><option>V1</option><option>V2</option></select></label>
<label>RDM <select id="rdm-kind"></select></label></div><p id="rdm-status"></p><div id="individual-rdm" style="height:800px"></div>
<script>
const rdmGroups=__RDM_DATA__,rdmNames=__RDM_NAMES__;
const rdmById=id=>document.getElementById(id);
const rdmEscape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
rdmGroups.forEach((group,index)=>{const option=document.createElement('option');option.value=index;option.textContent=group.label;rdmById('rdm-run').append(option)});
Object.entries(rdmNames).forEach(([name,label])=>{const option=document.createElement('option');option.value=name;option.textContent=label;rdmById('rdm-kind').append(option)});
let rdmGeneration=0;
function renderRdm(){
 const generation=++rdmGeneration;
 rdmById('rdm-status').textContent='Computing selected RDM…';
 requestAnimationFrame(()=>setTimeout(()=>{
  if(generation!==rdmGeneration)return;
  const group=rdmGroups[Number(rdmById('rdm-run').value)],position=rdmById('rdm-vowel').value,key=rdmById('rdm-kind').value;
  const selected=group.positions[position],packed=selected.vectors[key],raw=atob(packed.data),bytes=new Uint8Array(raw.length);
  for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
  const view=new DataView(bytes.buffer),n=packed.shape[0],d=packed.shape[1],values=new Float32Array(n*d);
  for(let i=0;i<values.length;i++)values[i]=view.getFloat32(4*i,true);
  const valid=Array.from({length:n},(_,i)=>{for(let k=0;k<d;k++)if(!Number.isFinite(values[i*d+k]))return false;return true});
  const matrix=Array.from({length:n},()=>new Array(n).fill(null)),backness=key.endsWith('backness');
  for(let i=0;i<n;i++){if(!valid[i])continue;matrix[i][i]=0;
   for(let j=i+1;j<n;j++){if(!valid[j])continue;let distance=0;
    if(backness)distance=values[i*d]===values[j*d]?0:1;
    else {for(let k=0;k<d;k++){const difference=values[i*d+k]-values[j*d+k];distance+=difference*difference}distance=Math.sqrt(distance)}
    matrix[i][j]=distance;matrix[j][i]=distance;
   }
  }
  const labels=selected.labels.map(rdmEscape);
  Plotly.react('individual-rdm',[{type:'heatmap',z:matrix,x:labels,y:labels,colorscale:'Viridis',zmin:0,zmax:backness?1:undefined,hoverongaps:false,
    hovertemplate:'%{y}<br>%{x}<br>Distance: %{z:.4f}<extra></extra>'}],
   {title:rdmEscape(group.label)+' · '+position+' · '+rdmNames[key],xaxis:{showticklabels:false,title:'Test stimuli'},
    yaxis:{showticklabels:false,autorange:'reversed',title:'Test stimuli'},margin:{t:80},height:800},{responsive:true});
  rdmById('rdm-status').textContent=`${n} vowel stimuli; ${valid.filter(Boolean).length} valid for this RDM.`;
 },0));
}
['rdm-run','rdm-vowel','rdm-kind'].forEach(id=>rdmById(id).addEventListener('change',renderRdm));renderRdm();
</script>'''
    viewer = viewer.replace("__RDM_DATA__", payload).replace("__RDM_NAMES__", names)
    description = '''<h2>Vowel RSA</h2><p>RSA correlates the upper triangles of RDMs using Spearman correlation.
Input/target/predicted backness and input/target/postnet-predicted mean spectra form six baselines; predicted baselines belong to the selected model.
Wrong vowel predictions are retained; only undefined backness is omitted from predicted-backness comparisons.
Correlations are computed per run and condition, separately for V1/V2, then averaged arithmetically across runs.
Cross-model comparisons use only functionally corresponding components and pair by language, condition, direction, and run number.
All matching trial pairs are included; their full test sets and included vowel stimuli must match exactly.
Missing partner runs remain pending. Blank correlations are not estimable (constant RDM or fewer than three valid stimuli).
These are descriptive correspondences, not evidence of uniquely phonological or causal coding; no pairwise p-values are reported.</p>'''
    return (description + "".join(f"<section>{plot}</section>" for plot in plots) + viewer
            + "<details><summary>RSA counts and exclusions</summary><pre>"
            + html.escape("\n".join(diagnostics)) + "</pre></details>")
