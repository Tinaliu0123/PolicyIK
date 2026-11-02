"""Average the last 20 checkpoints of a training run with exponential weights to obtain the deployed policy."""
from __future__ import annotations

import argparse
import glob
import os
import re

import torch

LAST_EPOCH = 13000
EPOCHS = list(range(LAST_EPOCH, 11099, -100))   # 20 checkpoints, newest first
HALF_LIFE = 400.0


def find_checkpoints(run_dir: str) -> list[str]:
    """rl_games periodic saves <run_dir>/<time>/nn/last_<name>_ep_<epoch>_rew_<reward>.pth, one per epoch."""
    paths = []
    for ep in EPOCHS:
        pat = re.compile(rf"_ep_{ep}_rew_-?[0-9]+(\.[0-9]+)?\.pth$")
        hits = [p for p in glob.glob(os.path.join(run_dir, "*", "nn", f"last_*_ep_{ep}_rew_*.pth")) if pat.search(p)]
        if len(hits) != 1:
            raise FileNotFoundError(f"epoch {ep}: expected one periodic checkpoint in {run_dir}, found {hits}")
        paths.append(hits[0])
    return paths


def average(paths: list[str], weights: list[float]) -> dict:
    w = torch.tensor(weights, dtype=torch.float64)
    w = w / w.sum()
    base, acc = None, None
    for wi, path in zip(w.tolist(), paths):
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = state["model"]
        if acc is None:
            base = state
            acc = {k: (v.double() * wi if v.dtype.is_floating_point else v.clone()) for k, v in model.items()}
            continue
        for k, v in model.items():
            if v.dtype.is_floating_point:
                acc[k] += v.double() * wi
            elif not torch.equal(acc[k], v):
                raise ValueError(f"integer tensor {k} differs between checkpoints")
    base["model"] = {k: (acc[k].to(v.dtype) if v.dtype.is_floating_point else acc[k]) for k, v in base["model"].items()}
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="log directory of the final training segment")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    weights = [0.5 ** ((LAST_EPOCH - ep) / HALF_LIFE) for ep in EPOCHS]
    torch.save(average(find_checkpoints(a.run_dir), weights), a.out)
    print(f"averaged {len(EPOCHS)} checkpoints (epochs {EPOCHS[-1]}-{EPOCHS[0]}) -> {a.out}")


if __name__ == "__main__":
    main()
