from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build_archive(results_dir: Path, output_path: Path) -> int:
    """Archive the results directory and return the file count."""
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    archived_files = 0
    resolved_output = output_path.resolve()

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(results_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.resolve() == resolved_output:
                continue

            arcname = path.relative_to(results_dir.parent).as_posix()
            archive.write(path, arcname=arcname)
            archived_files += 1

    return archived_files


def main():
    parser = argparse.ArgumentParser(
        description="Package Prolepsis run artifacts into one archive",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Results directory to archive",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Archive path to create",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else Path(f"{results_dir.name}_download.zip")

    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    archived_files = build_archive(results_dir, output_path)

    print(f"Created artifact archive: {output_path}")
    print(f"Archived files: {archived_files}")


if __name__ == "__main__":
    main()
