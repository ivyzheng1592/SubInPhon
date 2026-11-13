"""Extract representations and render PCA, RSA, and timestep classification."""

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
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from sklearn.decomposition import PCA
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


def _classify_sequences(sequences, masks, targets, folds):
    results = {}
    for name, values in sequences.items():
        steps = values.shape[1]
        tasks = {
            task: {"scores": [None] * steps, "available": [0] * steps,
                   "evaluated": [0] * steps, "folds_used": [0] * steps,
                   "class_counts": [[0, 0] for _ in range(steps)]}
            for task in CLASSIFICATION_TASKS
        }
        for step in range(steps):
            valid = masks[name][:, step] & np.isfinite(values[:, step]).all(axis=1)
            valid &= (folds >= 0) & (targets >= 0).all(axis=1)
            predictions = np.full(targets.shape, -1, dtype=int)
            for task_index, task in enumerate(CLASSIFICATION_TASKS):
                tasks[task]["available"][step] = int(valid.sum())
            if valid.sum() < 5:
                continue
            for fold in range(5):
                train = valid & (folds != fold)
                test = valid & (folds == fold)
                if not train.any() or not test.any():
                    continue
                scaler = StandardScaler()
                train_values = scaler.fit_transform(values[train, step])
                test_values = scaler.transform(values[test, step])
                for task_index, task in enumerate(CLASSIFICATION_TASKS):
                    if len(np.unique(targets[train, task_index])) != 2:
                        continue
                    classifier = LogisticRegression(
                        C=1.0, class_weight="balanced", solver="liblinear",
                        max_iter=1000, random_state=hp.base_seed,
                    )
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        classifier.fit(train_values, targets[train, task_index])
                    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                        continue
                    predictions[test, task_index] = classifier.predict(test_values)
                    tasks[task]["folds_used"][step] += 1
            for task_index, task in enumerate(CLASSIFICATION_TASKS):
                evaluated = valid & (predictions[:, task_index] >= 0)
                labels = targets[evaluated, task_index]
                tasks[task]["evaluated"][step] = int(evaluated.sum())
                tasks[task]["class_counts"][step] = [int((labels == value).sum()) for value in (0, 1)]
                if evaluated.sum() == valid.sum() and len(np.unique(labels)) == 2:
                    tasks[task]["scores"][step] = float(balanced_accuracy_score(labels, predictions[evaluated, task_index]))
        results[name] = {
            "timesteps": list(range(1, steps + 1)) if name.startswith("decoder") else list(range(steps)),
            "tasks": tasks,
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
    """Classify all saved checkpoints and pool final-checkpoint vowels without teacher forcing."""
    dataset = recorder.dataset
    architecture = "t2" if hasattr(model, "gaussian_upsample") else "t1"
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    match = re.search(r"_epoch(-?\d+)_seq2seq\.pth$", checkpoint.name)
    if match is None:
        raise ValueError(f"Cannot parse checkpoint epoch: {checkpoint}")
    final_epoch = int(match.group(1))
    prefix = checkpoint.name[:match.start()]
    checkpoints = []
    for path in checkpoint.parent.glob(prefix + "_epoch*_seq2seq.pth"):
        found = re.fullmatch(re.escape(prefix) + r"_epoch(-?\d+)_seq2seq\.pth", path.name)
        if found and int(found.group(1)) <= final_epoch:
            checkpoints.append((int(found.group(1)), path))
    checkpoints.sort()
    indices = _row_indices(dataloader.dataset, dataset)
    if not indices:
        raise ValueError("No test examples to analyze")
    targets, folds, split = _classification_split(recorder, indices)
    batch_size = hp.batch_size
    base_collate = dataset._get_collate_fn()
    text_steps = max(len(dataset.annotations.iloc[index].sr_string) + 1 for index in indices)

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
    metadata, skipped, handles, interval_cache = [], [], [], {}
    classification = {
        "method": "L2 logistic regression, C=1, balanced class weights; within-fold standardization",
        "score": "Balanced accuracy of pooled out-of-fold predictions; chance = 0.5",
        "fold_count": 5, "split_id": split["split_id"], "split": split,
        "checkpoints": [],
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
        for epoch, path in checkpoints:
            print(f"Interpretability epoch {epoch}: extracting {len(indices)} test words", flush=True)
            model.load_state_dict(torch.load(path, map_location=next(model.parameters()).device, weights_only=True))
            full_sequences, sequence_masks = {}, {}
            offset = 0
            with torch.no_grad():
                for batch, count in loader:
                    for values in captures.values():
                        values.clear()
                    output = model(batch, 0.0, 0.0)
                    sequences = {}
                    for name, values in captures.items():
                        if not values:
                            raise RuntimeError(f"No representations captured for {name}")
                        if name == "synthesizer_input" and architecture == "t2":
                            sequences[name] = values[0]
                        else:
                            sequences[name] = torch.cat(values, dim=0).transpose(0, 1)
                    source_mask = batch[1].detach().cpu().ne(-100).any(dim=2).squeeze(1)
                    target_mask = batch[3].detach().cpu().ne(-100).any(dim=2).squeeze(1)
                    text_mask = batch[2][1:].detach().cpu().transpose(0, 1)
                    text_mask = (
                        text_mask.ne(hp.special_tokens.index(hp.pad_token))
                        & text_mask.ne(hp.special_tokens.index(hp.sos_token))
                        & text_mask.ne(hp.special_tokens.index(hp.eos_token))
                    )
                    for name, sequence in sequences.items():
                        steps = text_steps if name.startswith("decoder") else sequence.shape[1]
                        if name not in full_sequences:
                            full_sequences[name] = np.zeros((len(indices), steps, sequence.shape[2]), dtype=np.float32)
                            sequence_masks[name] = np.zeros((len(indices), steps), dtype=bool)
                        mask = source_mask if name == "encoder_hidden" else text_mask if name.startswith("decoder") else target_mask
                        length = sequence.shape[1]
                        full_sequences[name][offset:offset + count, :length] = sequence[:count].numpy()
                        sequence_masks[name][offset:offset + count, :length] = mask[:count].numpy()
                    if epoch == final_epoch:
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
                                pooled = []
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
                            except (OSError, ValueError, RuntimeError) as error:
                                skipped.append({"dataset_index": row_index, "ur": row.ur_string, "reason": str(error)})
                                continue
                            _, _, predicted = recorder.tensor2string(batch[0][:, item], batch[2][:, item], output[1][:, item])
                            for vowel_index, values in enumerate(pooled):
                                vowel = underlying[vowel_index]
                                backness, height, _ = recorder.language.vowel[vowel]
                                metadata.append({
                                    "dataset_index": row_index, "ur": row.ur_string,
                                    "sr": row.sr_string, "prediction": predicted,
                                    "correct": predicted == row.sr_string,
                                    "word_structure": "-".join(
                                        "".join("V" if char in recorder.language.vowel else "C" for char in syllable)
                                        for syllable in row.ur_syll.split(".")
                                    ),
                                    "position": "V" + str(vowel_index + 1),
                                    "vowel": vowel, "backness": backness, "height": height,
                                })
                                for name, value in values.items():
                                    vectors[name].append(value)
                    offset += count
            print(f"Interpretability epoch {epoch}: classifying four features at each timestep", flush=True)
            classification["checkpoints"].append({
                "epoch": epoch, "checkpoint": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "representations": _classify_sequences(full_sequences, sequence_masks, targets, folds),
            })
            dimensions = {name: value.shape[2] for name, value in full_sequences.items()}
            del full_sequences, sequence_masks
    finally:
        for handle in handles:
            handle.remove()
        model.load_state_dict(previous_weights)
        model.train(previous_mode)
        random.setstate(random_state)

    info = {
        "architecture": architecture, "trial": recorder.trial_num,
        "language": recorder.lang_name, "condition": recorder.condition,
        "direction": recorder.directionality, "run": recorder.run_num,
        "checkpoint": str(checkpoint), "checkpoint_sha256": classification["checkpoints"][-1]["sha256"],
        "teacher_forcing": {"text": 0.0, "audio": 0.0},
        "test_items": len(indices), "included_words": len(metadata) // 2,
        "skipped": skipped, "rows": metadata,
        "alignment": "Reference source/target TextGrid vowel means; reference target text positions",
    }
    report_file = Path(report_file).resolve()
    archive_dir = report_file.parent / (report_file.stem + "_data")
    archive_dir.mkdir(parents=True, exist_ok=True)
    identity = {key: info[key] for key in ("architecture", "trial", "language", "condition", "direction", "run", "checkpoint")}
    archive = archive_dir / (hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20] + ".npz")
    with tempfile.NamedTemporaryFile(dir=archive_dir, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        arrays = {
            name: np.stack(values).astype(np.float32) if values else np.empty((0, dimensions[name]), dtype=np.float32)
            for name, values in vectors.items()
        }
        analysis = _analyze(arrays, metadata)
        np.savez_compressed(
            temporary_path, metadata=np.array(json.dumps(info)),
            analysis=np.array(json.dumps(analysis, allow_nan=False)),
            classification=np.array(json.dumps(classification, allow_nan=False)), **arrays,
        )
        os.replace(temporary_path, archive)
    finally:
        temporary_path.unlink(missing_ok=True)
    write_report(archive_dir, report_file)
    print(f"Interpretability: {len(checkpoints)} checkpoints classified; final PCA/RSA {info['included_words']}/{len(indices)} words; {report_file}")
    return {
        "report": str(report_file), "archive": str(archive),
        "included_words": info["included_words"], "skipped_words": len(skipped),
        "classified_epochs": [epoch for epoch, _ in checkpoints], "split_id": split["split_id"],
    }


def _rsa(vectors, rows, position):
    indices = [i for i, row in enumerate(rows) if position == "Both" or row["position"] == position]
    values = vectors[indices]
    result = {"count": len(indices), "features": {}}
    distances = pdist(values, metric="euclidean") if len(values) >= 2 else np.array([])
    ranks = rankdata(distances)
    rank_std = float(ranks.std()) if len(ranks) else 0.0
    for feature in ("backness", "height"):
        labels = np.array([rows[i][feature] for i in indices])
        classes, labels = np.unique(labels, return_inverse=True)
        if len(classes) < 2 or len(distances) < 2 or np.ptp(distances) == 0:
            result["features"][feature] = {"rho": None, "pairs": len(distances)}
            continue
        different = pdist(labels[:, None], metric="hamming").astype(bool)
        if not different.any() or different.all():
            result["features"][feature] = {"rho": None, "pairs": len(distances)}
            continue
        proportion = float(different.mean())
        rho = float(
            (ranks[different].mean() - ranks[~different].mean())
            * np.sqrt(proportion * (1 - proportion)) / rank_std
        )
        result["features"][feature] = {
            "rho": rho if np.isfinite(rho) else None, "pairs": len(distances),
            "same_mean": float(distances[~different].mean()),
            "different_mean": float(distances[different].mean()),
        }
    return result


def _analyze(vectors, rows):
    panels = {}
    for name in REPRESENTATIONS:
        values = vectors[name].astype(np.float64)
        if len(values) == 0:
            panels[name] = {
                "coordinates": [], "variance": [0.0, 0.0],
                "rsa": {position: _rsa(values, rows, position) for position in ("Both", "V1", "V2")},
            }
            continue
        components = min(2, *values.shape)
        if np.any(values != values[0]):
            pca = PCA(n_components=components, svd_solver="full")
            coordinates = pca.fit_transform(values)
            variance = pca.explained_variance_ratio_.tolist()
        else:
            coordinates = np.zeros((len(values), components))
            variance = [0.0] * components
        if components == 1:
            coordinates = np.pad(coordinates, ((0, 0), (0, 1)))
            variance.append(0.0)
        panels[name] = {
            "coordinates": coordinates.round(6).tolist(), "variance": variance,
            "rsa": {position: _rsa(values, rows, position) for position in ("Both", "V1", "V2")},
        }
    return panels


def write_report(archive_dir, report_file):
    """Combine final vowel analyses and all-checkpoint classifier heatmaps."""
    groups = []
    for archive in sorted(Path(archive_dir).glob("*.npz")):
        with np.load(archive, allow_pickle=False) as saved:
            info = json.loads(str(saved["metadata"].item()))
            panels = json.loads(str(saved["analysis"].item())) if "analysis" in saved else _analyze(saved, info["rows"])
            classification = json.loads(str(saved["classification"].item())) if "classification" in saved else None
            groups.append({"info": info, "panels": panels, "classification": classification})
    payload = json.dumps(groups, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    title = "Vowel representations: PCA, RSA, and classification"
    document = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>body{font:15px system-ui;margin:28px;color:#202630;background:#f7f8fa}h1{font-size:24px}
.controls{display:flex;gap:18px;flex-wrap:wrap}label{display:grid;gap:5px}select{padding:8px;max-width:560px}
.plots{display:grid;grid-template-columns:3fr 2fr;gap:18px;margin-top:18px}.plot{background:white;min-height:480px}
.heatmaps{display:grid;grid-template-columns:1fr 1fr;gap:18px}.heatmap{background:white;min-height:340px}
#details{line-height:1.6;margin-top:16px}summary{cursor:pointer}#skipped{white-space:pre-wrap}
@media(max-width:900px){.plots,.heatmaps{grid-template-columns:1fr}}</style>
<h1>__TITLE__</h1><div class="controls">
<label>Model / condition / run<select id="group"></select></label>
<label>Representation<select id="representation"></select></label>
<label>Color by<select id="feature"><option value="backness">Underlying backness</option><option value="height">Underlying height</option></select></label>
<label>Vowel position<select id="position"><option>Both</option><option>V1</option><option>V2</option></select></label></div>
<div class="plots"><div id="pca" class="plot"></div><div id="rsa" class="plot"></div></div>
<div id="details"></div><details><summary>Excluded words</summary><div id="skipped"></div></details>
<h2>Classification across saved epochs</h2><div id="classification-details"></div>
<div class="heatmaps"><div id="V1_backness" class="heatmap"></div><div id="V1_height" class="heatmap"></div>
<div id="V2_backness" class="heatmap"></div><div id="V2_height" class="heatmap"></div></div>
<p>Classifier cells show balanced accuracy of held-out word predictions from fixed five-fold cross-validation within the model test set.
Standardization is fitted separately on each training fold. Chance is 0.5; blank cells have insufficient samples, classes, or valid fits.
Acoustic timestep indices start at 0; decoder indices start at 1 after text SOS. Padding is excluded without compressing the timeline.
Saved epoch -1 is the untrained checkpoint. Classification uses unpooled sequences without TextGrid segmentation.</p>
<p>PCA and RSA use the final checkpoint only. PCA is fitted jointly to V1 and V2 separately for each checkpoint and representation; axes are not shared across models.
RSA is Spearman correlation between full-space Euclidean distances and binary feature differences (same = 0, different = 1).
It measures feature-related geometry, not classifier accuracy or causal use. Pairwise observations are not independent; no pairwise p-values are reported.
Pooling uses reference intervals and positions, including for incorrect predictions. Acoustic and text padding are excluded.</p>
<script>__PLOTLY__</script><script>
const groups=__DATA__,names=__NAMES__,tasks=__TASKS__;
const byId=id=>document.getElementById(id);
const escapeText=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
groups.forEach((g,i)=>{let o=document.createElement('option');o.value=i;
o.textContent=`${g.info.architecture.toUpperCase()} · ${g.info.condition} · ${g.info.direction} · run ${g.info.run} · ${g.info.trial}`;byId('group').append(o)});
Object.entries(names).forEach(([key,label])=>{let o=document.createElement('option');o.value=key;o.textContent=label;byId('representation').append(o)});
function render(){
 const g=groups[Number(byId('group').value)],key=byId('representation').value,p=g.panels[key],feature=byId('feature').value,position=byId('position').value;
 const labels=[...new Set(g.info.rows.map(row=>row[feature]))].sort(),colors=['#2469b2','#d87727'];
 let traces=[];
 labels.forEach((label,color)=>['V1','V2'].forEach(vowel=>{
  if(position!=='Both'&&position!==vowel)return;
  let ids=g.info.rows.map((row,i)=>row[feature]===label&&row.position===vowel?i:-1).filter(i=>i>=0);
  traces.push({type:'scattergl',mode:'markers',name:`${label} · ${vowel}`,x:ids.map(i=>p.coordinates[i][0]),y:ids.map(i=>p.coordinates[i][1]),
   marker:{color:colors[color%colors.length],symbol:vowel==='V1'?'circle':'diamond',size:7,opacity:.65},
   text:ids.map(i=>{const r=g.info.rows[i];return `${escapeText(r.ur)} → ${escapeText(r.sr)}<br>${r.position}: /${escapeText(r.vowel)}/ · ${escapeText(r.backness)}, ${escapeText(r.height)}<br>Prediction: ${escapeText(r.prediction)} · ${r.correct?'correct':'incorrect'}<br>${escapeText(r.word_structure)} · item ${r.dataset_index}`}),hovertemplate:'%{text}<extra></extra>'});
 }));
 const representation=key==='synthesizer_input'?(g.info.architecture==='t1'?'Synthesizer attention context':'Upsampled representation'):names[key];
 Plotly.react('pca',traces,{title:`${representation}: vowel PCA`,xaxis:{title:`PC1 (${(100*p.variance[0]).toFixed(1)}%)`},yaxis:{title:`PC2 (${(100*p.variance[1]).toFixed(1)}%)`},margin:{t:65},legend:{orientation:'h'}},{responsive:true});
 const stats=p.rsa[position],features=['backness','height'];
 Plotly.react('rsa',[{type:'bar',x:features,y:features.map(f=>stats.features[f].rho),marker:{color:features.map(f=>f===feature?'#2469b2':'#8e9cad')},
 text:features.map(f=>stats.features[f].rho===null?'Not estimable':stats.features[f].rho.toFixed(3)),textposition:'auto',
 customdata:features.map(f=>[stats.features[f].pairs,stats.features[f].same_mean??null,stats.features[f].different_mean??null]),
 hovertemplate:'%{x}<br>Spearman rho: %{y:.3f}<br>Pairs: %{customdata[0]}<br>Same-feature mean distance: %{customdata[1]:.3f}<br>Different-feature mean distance: %{customdata[2]:.3f}<extra></extra>'}],
 {title:`Feature RSA · ${position} · ${stats.count} vowels`,yaxis:{title:'Spearman rho',range:[-1,1]},margin:{t:65}},{responsive:true});
 const correct=g.info.rows.filter(row=>row.position==='V1'&&row.correct).length;
 byId('details').textContent=`Final PCA/RSA: ${g.info.language} · ${g.info.included_words}/${g.info.test_items} words included · ${g.info.skipped.length} excluded · ${correct} correct words. Teacher forcing: text 0, audio 0. Checkpoint: ${g.info.checkpoint}`;
 byId('skipped').textContent=g.info.skipped.length?g.info.skipped.map(row=>`${row.dataset_index}: ${row.ur} — ${row.reason}`).join('\n'):'None';
 const classification=g.classification,checkpoints=classification?.checkpoints??[];
 byId('classification-details').textContent=checkpoints.length?`${classification.method}. ${classification.split.strategy}. Split ID: ${classification.split_id}. All five components use the same word folds. Hover for sample counts and fold coverage.`:'No classifier results saved for this group; rerun interpretability evaluation.';
 Object.entries(tasks).forEach(([task,label])=>{
  const timesteps=checkpoints.length?checkpoints[0].representations[key].timesteps:[];
  const stats=checkpoints.map(checkpoint=>checkpoint.representations[key].tasks[task]);
  const classes=classification?.split.classes[task]??['class 0','class 1'];
  Plotly.react(task,[{type:'heatmap',x:timesteps,y:checkpoints.map(checkpoint=>String(checkpoint.epoch)),
   z:stats.map(item=>item.scores),zmin:0,zmax:1,colorscale:'Greys',connectgaps:false,hoverongaps:false,
   customdata:stats.map(item=>item.scores.map((_,i)=>[item.available[i],item.evaluated[i],item.folds_used[i],...item.class_counts[i]])),
   colorbar:{title:'Balanced accuracy'},hovertemplate:`Epoch %{y}<br>Timestep %{x}<br>Balanced accuracy: %{z:.3f}<br>Available words: %{customdata[0]}<br>Held-out words scored: %{customdata[1]}<br>Folds used: %{customdata[2]} / 5<br>${escapeText(classes[0])}: %{customdata[3]} · ${escapeText(classes[1])}: %{customdata[4]}<extra></extra>`}],
   {title:label,xaxis:{title:key.startsWith('decoder')?'Target text timestep':key==='encoder_hidden'?'Source acoustic frame':'Target acoustic frame'},
    yaxis:{title:'Saved epoch',type:'category',autorange:'reversed'},margin:{t:55,l:70},height:340},{responsive:true});
 });
}
['group','representation','feature','position'].forEach(id=>byId(id).addEventListener('change',render));render();
</script></html>'''
    document = document.replace("__TITLE__", html.escape(title)).replace("__PLOTLY__", get_plotlyjs())
    document = document.replace("__DATA__", payload).replace("__NAMES__", json.dumps(REPRESENTATIONS))
    document = document.replace("__TASKS__", json.dumps(CLASSIFICATION_TASKS))
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    Path(report_file).write_text(document, encoding="utf-8")
