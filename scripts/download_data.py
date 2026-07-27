#!/usr/bin/env python3
"""Download all TalkPlayData Challenge datasets to a local directory.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --force
    python scripts/download_data.py --output-dir /data/talkplay
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import load_dataset

DATASETS = [
    {
        "hf_name": "talkpl-ai/TalkPlayData-Challenge-Dataset",
        "local_dir": "challenge-dataset",
    },
    {
        "hf_name": "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        "local_dir": "track-metadata",
    },
    {
        "hf_name": "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
        "local_dir": "user-metadata",
    },
    {
        "hf_name": "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        "local_dir": "track-embeddings",
    },
    {
        "hf_name": "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        "local_dir": "user-embeddings",
    },
    {
        "hf_name": "talkpl-ai/TalkPlayData-Challenge-Blind-A",
        "local_dir": "blind-a",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all TalkPlayData Challenge datasets to input/."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="input",
        help="Base directory for downloaded datasets (default: input)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even if the local directory already exists",
    )
    return parser.parse_args()


def download_all(output_dir: str, force: bool) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, ds_info in enumerate(DATASETS, 1):
        target = output_path / ds_info["local_dir"]
        label = f"[{i}/{len(DATASETS)}] {ds_info['hf_name']}"

        if target.exists() and not force:
            print(f"{label} -- SKIPPED (already exists at {target})")
            continue

        print(f"{label} -- downloading ...")
        dataset_dict = load_dataset(ds_info["hf_name"])

        for split_name, split_ds in dataset_dict.items():
            print(f"  split '{split_name}': {len(split_ds)} rows")

        temp_target = target.with_name(f".{target.name}.downloading")
        if temp_target.exists():
            shutil.rmtree(temp_target)

        dataset_dict.save_to_disk(str(temp_target))

        if target.exists():
            shutil.rmtree(target)
        temp_target.rename(target)

        print(f"  Saved to {target}")

    print(f"\nDone. All datasets are in: {output_path.resolve()}")


def main() -> int:
    args = parse_args()
    download_all(args.output_dir, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
