from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> None:
    print("[postprocess] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Post-process generated ECG samples into labeled .pt datasets using an external pipeline."
    )
    parser.add_argument("--input", required=True, help="Path to generated .npy file")
    parser.add_argument("--output", required=True, help="Path to output .pt dataset")
    parser.add_argument(
        "--annotation-file",
        default=None,
        help="Optional annotation file for direct label assignment",
    )
    parser.add_argument(
        "--use-model-for-label",
        action="store_true",
        help="Use a pretrained classifier to assign labels after beat extraction",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint of pretrained classifier used for label assignment",
    )
    parser.add_argument(
        "--dataset-type",
        default="mit",
        choices=["mit", "ptb"],
        help="Dataset type forwarded to downstream preprocess script",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "src/preprocessing/downtream_preprocess.py",
        "--signal_file",
        str(input_path),
        "--out_pt",
        str(output_path),
        "--dataset_type",
        args.dataset_type,
    ]

    if args.annotation_file:
        cmd.extend(["--annotation_file", args.annotation_file])
    elif args.use_model_for_label:
        cmd.append("--use_model_for_label")
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])

    run_command(cmd)
    print(f"[postprocess] Saved labeled dataset to: {output_path}")


if __name__ == "__main__":
    main()
