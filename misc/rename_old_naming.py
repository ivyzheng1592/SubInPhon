import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple


KNOWN_PROPERTIES = {"nonidentical"}
RESULT_MODALITIES = {"txt", "fea", "aud"}
CONDITIONS = {"harmony", "disharmony"}
DIRECTIONALITIES = {"l2r", "r2l"}


def join_name_parts(parts: Sequence[str]) -> str:
    return "_".join(part for part in parts if part)


def parse_old_generated_dir(dir_name: str) -> Optional[Tuple[str, str]]:
    suffix = "_generated_data"
    if not dir_name.endswith(suffix):
        return None
    stem = dir_name[: -len(suffix)]
    parts = stem.split("_")
    if len(parts) < 2 or parts[-1] in KNOWN_PROPERTIES:
        return None
    return parts[0], "_".join(parts[1:])


def parse_old_results_dir(dir_name: str) -> Optional[Tuple[str, str, str]]:
    parts = dir_name.split("_")
    if len(parts) < 3 or parts[-1] not in RESULT_MODALITIES:
        return None
    if len(parts) >= 4 and parts[-2] in KNOWN_PROPERTIES:
        return None
    return parts[0], "_".join(parts[1:-1]), parts[-1]


def parse_old_generated_file(file_name: str) -> Optional[Dict[str, str]]:
    match = re.match(
        r"^(?P<registry>.+?)_(?P<directionality>l2r|r2l)"
        r"(?:_(?P<property>[^_]+))?_run(?P<run_num>\d+)_(?P<kind>harmony|disharmony|template_counts)"
        r"(?P<ext>\.csv|\.xlsx)$",
        file_name,
    )
    if not match:
        return None
    info = match.groupdict()
    info["property"] = info["property"] or ""
    return info


def parse_old_run_component(component: str) -> Optional[Dict[str, str]]:
    match = re.match(
        r"^(?P<language>.+?)_(?P<modality>txt|fea|aud)_(?P<directionality>l2r|r2l)"
        r"(?:_(?P<property>[^_]+))?_(?P<condition>harmony|disharmony)_run(?P<run_num>\d+)"
        r"(?P<suffix>.*)$",
        component,
    )
    if not match:
        return None
    info = match.groupdict()
    info["property"] = info["property"] or ""
    return info


def build_new_generated_name(info: Dict[str, str]) -> str:
    stem = join_name_parts([info["registry"], info["property"], info["directionality"]])
    return f"{stem}_run{info['run_num']}_{info['kind']}{info['ext']}"


def build_new_run_stem(info: Dict[str, str]) -> str:
    return join_name_parts(
        [
            info["language"],
            info["property"],
            info["modality"],
            info["directionality"],
            info["condition"],
            "run" + info["run_num"],
        ]
    )


def transform_old_result_component(component: str, lang: str, modality: str, property_label: str) -> str:
    run_info = parse_old_run_component(component)
    if run_info and run_info["language"] == lang and run_info["modality"] == modality:
        return build_new_run_stem(run_info) + run_info["suffix"]

    old_result_root = f"{lang}_{modality}"
    new_result_root = join_name_parts([lang, property_label, modality])
    if component.startswith(old_result_root):
        return new_result_root + component[len(old_result_root):]
    return component


def infer_property_from_path_components(path: Path, lang: str, modality: str) -> Optional[str]:
    for component in path.parts:
        run_info = parse_old_run_component(component)
        if run_info and run_info["language"] == lang and run_info["modality"] == modality:
            return run_info["property"]
    return None


def property_from_run_config(file_path: Path) -> Optional[str]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("property:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def property_from_csv(file_path: Path) -> Optional[str]:
    try:
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row.get("property", "") or ""
    except OSError:
        return None
    return ""


def move_file(source: Path, target: Path, apply: bool) -> None:
    if source == target:
        print(f"skip unchanged {source}")
        return
    if target.exists():
        print(f"skip existing target {source} -> {target}")
        return
    print(f"{source} -> {target}")
    if not apply:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def cleanup_empty_dirs(root: Path, apply: bool) -> None:
    directories = sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True)
    for directory in directories:
        try:
            next(directory.iterdir())
        except StopIteration:
            print(f"remove empty dir {directory}")
            if apply:
                directory.rmdir()
        except OSError:
            continue

    try:
        next(root.iterdir())
    except StopIteration:
        print(f"remove empty dir {root}")
        if apply:
            root.rmdir()
    except OSError:
        pass


def migrate_data_root(project_root: Path, apply: bool) -> Set[Path]:
    changed_roots: Set[Path] = set()
    dataset_root = project_root / "dataset"
    if not dataset_root.exists():
        return changed_roots

    for directory in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        parsed = parse_old_generated_dir(directory.name)
        if not parsed:
            continue
        trial_num, lang_name = parsed
        for file_path in sorted(p for p in directory.iterdir() if p.is_file()):
            file_info = parse_old_generated_file(file_path.name)
            if not file_info:
                continue
            property_label = file_info["property"]
            target_dir = dataset_root / join_name_parts([trial_num, lang_name, property_label, "generated_data"])
            target_file = target_dir / build_new_generated_name(file_info)
            move_file(file_path, target_file, apply)
            changed_roots.add(directory)
            changed_roots.add(target_dir)
    return changed_roots


def migrate_output_root(project_root: Path, apply: bool) -> Set[Path]:
    changed_roots: Set[Path] = set()
    output_root = project_root / "output"
    if not output_root.exists():
        return changed_roots

    for directory in sorted(p for p in output_root.iterdir() if p.is_dir()):
        parsed = parse_old_results_dir(directory.name)
        if not parsed:
            continue
        trial_num, lang_name, modality = parsed
        old_result_root = f"{lang_name}_{modality}"

        for file_path in sorted(p for p in directory.rglob("*") if p.is_file()):
            if file_path.name.startswith("."):
                continue

            relative_parts = file_path.relative_to(directory).parts
            if len(relative_parts) == 1 and file_path.name in {old_result_root + "_acc.csv", old_result_root + "_pred.csv"}:
                property_label = property_from_csv(file_path) or ""
                target_dir = output_root / join_name_parts([trial_num, lang_name, property_label, modality])
                target_root = join_name_parts([lang_name, property_label, modality])
                if file_path.name.endswith("_acc.csv"):
                    target_file = target_dir / f"{target_root}_acc.csv"
                else:
                    target_file = target_dir / f"{target_root}_pred.csv"
                move_file(file_path, target_file, apply)
                changed_roots.add(directory)
                changed_roots.add(target_dir)
                continue

            property_label = infer_property_from_path_components(file_path, lang_name, modality)
            if property_label is None and file_path.name == "run_config.txt":
                property_label = property_from_run_config(file_path)
            if property_label is None:
                property_label = ""

            target_dir = output_root / join_name_parts([trial_num, lang_name, property_label, modality])
            transformed_parts = [
                transform_old_result_component(component, lang_name, modality, property_label)
                for component in relative_parts
            ]
            target_file = target_dir.joinpath(*transformed_parts)
            move_file(file_path, target_file, apply)
            changed_roots.add(directory)
            changed_roots.add(target_dir)

    return changed_roots


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename old SubInPhon output paths to the new property-aware scheme.")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root containing the dataset/ and output/ folders.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the renaming. Without this flag, the script runs in dry-run mode.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    changed_roots = set()
    changed_roots.update(migrate_data_root(project_root, args.apply))
    changed_roots.update(migrate_output_root(project_root, args.apply))

    for root in sorted(changed_roots):
        cleanup_empty_dirs(root, args.apply)

    if not args.apply:
        print("Dry run only. Re-run with --apply to perform the renaming.")


if __name__ == "__main__":
    main()
