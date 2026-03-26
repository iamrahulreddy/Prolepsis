"""Build the minimal archive needed to run Prolepsis in a cloud environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = "prolepsis.zip"

ESSENTIAL_FILES = (
    "run.sh",
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "scripts/package_results_archive.py",
    "scripts/generate_prompts.py",
)

ESSENTIAL_DIRS = (
    "benchmark",
    "prolepsis",
    "tests",
)

SKIP_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
}

SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
}


def _iter_archive_paths():
    """Yield project files that are required by the release bundle."""
    for relative_path in ESSENTIAL_FILES:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {relative_path}")
        yield path

    for relative_dir in ESSENTIAL_DIRS:
        directory = ROOT / relative_dir
        if not directory.is_dir():
            raise FileNotFoundError(f"Required directory not found: {relative_dir}")

        for path in sorted(directory.rglob("*")):
            if path.is_dir():
                continue
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            if path.suffix in SKIP_SUFFIXES:
                continue
            yield path


def build_archive(output_path: Path) -> list[str]:
    """Create the release archive and return archived paths."""
    archived_paths: list[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in _iter_archive_paths():
            arcname = path.relative_to(ROOT).as_posix()
            archive.write(path, arcname=arcname)
            archived_paths.append(arcname)

    return archived_paths


def main():
    parser = argparse.ArgumentParser(
        description="Build the cloud release archive for Prolepsis",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_ARCHIVE,
        help="Archive path to create",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    archived_paths = build_archive(output_path)

    print(f"Created archive: {output_path}")
    print(f"Archived files: {len(archived_paths)}")
    for path in archived_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
