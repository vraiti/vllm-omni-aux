"""Compare transformer debug checkpoints between two runs.

Usage:
    python compare_transformer_debug.py <dir_a> <dir_b>
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def compare(dir_a: Path, dir_b: Path):
    files_a = sorted(dir_a.glob("tf_*.pt"))
    names = [f.stem for f in files_a]

    if not names:
        print(f"No tf_*.pt files found in {dir_a}")
        return

    for name in names:
        fa = dir_a / f"{name}.pt"
        fb = dir_b / f"{name}.pt"
        if not fb.exists():
            print(f"{name}: MISSING in {dir_b}")
            continue

        ta = torch.load(fa, map_location="cpu", weights_only=True).float()
        tb = torch.load(fb, map_location="cpu", weights_only=True).float()

        if ta.shape != tb.shape:
            print(f"{name}: SHAPE MISMATCH {ta.shape} vs {tb.shape}")
            continue

        diff = (ta - tb).abs()
        cos = F.cosine_similarity(
            ta.flatten().unsqueeze(0), tb.flatten().unsqueeze(0),
        ).item()

        label = name.replace("tf_", "")
        print(f"{label:40s}  cos={cos:10.6f}  max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    compare(Path(sys.argv[1]), Path(sys.argv[2]))
