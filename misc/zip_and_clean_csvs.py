#!/usr/bin/env python3
"""
Find CSV files that had corresponding .csv.gz files, remove all .csv.gz files,
and pack the selected CSV files into a single zip archive.

Configure `root_folder` and `zip_name` inline in `main()`.
"""

from pathlib import Path
import zipfile


def find_all_csvs(root_folder: Path):
    """Return list of all .csv Paths under `root_folder` (recursive)."""
    root_folder = root_folder.expanduser().resolve()
    return sorted(root_folder.rglob("*.csv"))


def remove_all_csv_gz(root_folder: Path) -> int:
    """Remove all files ending with .csv.gz under root_folder. Returns number removed."""
    root_folder = root_folder.expanduser().resolve()
    gz_files = list(root_folder.rglob("*.csv.gz"))
    removed = 0
    for gz in gz_files:
        try:
            gz.unlink()
            print(f"Removed: {gz}")
            removed += 1
        except Exception as e:
            print(f"Failed to remove {gz}: {e}")
    print(f"Total .csv.gz files removed: {removed}")
    return removed


def zip_csv_files(csv_files, zip_path: Path, root_folder: Path):
    """Create a zip at `zip_path` containing the files in `csv_files`.

    Files will be stored with paths relative to `root_folder`.
    """
    zip_path = zip_path.expanduser().resolve()
    root_folder = root_folder.expanduser().resolve()

    if not csv_files:
        print("No CSV files provided to zip.")
        return 0

    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        added = 0
        for f in csv_files:
            try:
                arcname = f.relative_to(root_folder)
            except Exception:
                arcname = f.name
            zf.write(f, arcname.as_posix())
            print(f"Added to zip: {f} -> {arcname}")
            added += 1

    print(f"Created zip archive: {zip_path} (files added: {added})")
    return added


def main():
    # Inline configuration
    root_folder = Path("/home/ldlmdl/Documents/SubInPhon/results/202606251603_EnglishBH_shortened_aud/EnglishBH_shortened_aud_embed_plots")
    zip_name = "EnglishBH_shortened_aud_l2r_embed_53-3_t2.zip"  # output zip placed inside root_folder

    root = root_folder.expanduser().resolve()
    zip_path = root / zip_name

    # 1) Collect all CSVs under the root (before removal)
    selected_csvs = find_all_csvs(root)
    print(f"Found {len(selected_csvs)} CSV files under {root}.")

    # 2) Remove all .csv.gz files
    #remove_all_csv_gz(root)

    # 3) Zip the selected CSVs into one archive
    zip_csv_files(selected_csvs, zip_path, root)


if __name__ == "__main__":
    main()
