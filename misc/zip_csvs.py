#!/usr/bin/env python3
"""Zip all CSV files under a folder into one archive."""

from pathlib import Path
import zipfile


def find_csv_files(root_folder: Path) -> list[Path]:
    """Return all CSV files under `root_folder`."""
    return sorted(root_folder.rglob("*.csv"))


def zip_csv_files(root_folder: Path, zip_path: Path) -> int:
    """Create one zip archive containing all CSV files under `root_folder`."""
    csv_files = find_csv_files(root_folder)

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for csv_file in csv_files:
            archive_name = csv_file.relative_to(root_folder)
            archive.write(csv_file, archive_name.as_posix())
            print(f"Added to zip: {csv_file} -> {archive_name}")

    print(f"Created zip archive: {zip_path} (files added: {len(csv_files)})")
    return len(csv_files)


def main() -> None:
    # Set the folder containing the CSV files to archive.
    root_folder = Path(
        "/home/ldlmdl/Documents/SubInPhon/results/202606251603_EnglishBH_shortened_aud/EnglishBH_shortened_aud_embed_plots"
    )
    # Set the output zip file path.
    zip_path = root_folder / "EnglishBH_shortened_aud_l2r_embed_53-3_t2.zip"

    print(f"Reading CSV files from: {root_folder}")
    zip_csv_files(root_folder, zip_path)


if __name__ == "__main__":
    main()
