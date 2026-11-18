"""从已完成实验复制预算、种子和数据配置，仅补跑理论 RMS 对照。"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from kda_hybrid_lm.models import HybridLM, ModelConfig
from kda_hybrid_lm.train.trainer import seed_all


REPLAY_KEYS = ("steps", "search_steps", "seeds", "tune_seed", "shuffle_seed",
               "adam_lrs", "scaled_lrs", "batch_size", "accumulation", "length",
               "layers", "hidden_dim", "intermediate_dim", "heads",
               "eval_every", "eval_batches", "trace_every", "threads")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Completed original compare directory")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/tinystories")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    reference = args.reference.resolve()
    launch = read_json(reference / "launch.json")
    if launch["mode"] != "compare":
        parser.error("--reference must be a completed real-data compare experiment")
    old_data = read_json(reference / "data-metadata.json")
    data = read_json(args.data_dir / "metadata.json")
    for key in ("vocab_size", "token_counts", "sha256", "tokenizer_sha256"):
        if old_data[key] != data[key]:
            parser.error(f"Prepared data differs from reference: {key}")
    for split, expected in old_data["sha256"].items():
        tokens = np.load(args.data_dir / f"{split}.npy", mmap_mode="r")
        if hashlib.sha256(tokens).hexdigest() != expected:
            parser.error(f"Token file hash differs from reference: {split}")
    config = ModelConfig(vocab_size=data["vocab_size"], hidden_dim=launch["hidden_dim"],
                         intermediate_dim=launch["intermediate_dim"], num_heads=launch["heads"],
                         layer_types=("softmax",) * launch["layers"])
    model_dict = json.loads(json.dumps(asdict(config)))
    baseline = {r["seed"]: r for r in read_json(reference / "summary.json")
                if r["optimizer"] == "adam_s"}
    if set(baseline) != set(launch["seeds"]):
        parser.error("Reference is missing Adam-S repeat seeds")
    # 在长训练开始前核对初始化，包括调参种子；构造模型不会改变后续重置的 RNG。
    for seed in [launch["tune_seed"], *launch["seeds"]]:
        folder = (reference / "search/adam_s-0" if seed == launch["tune_seed"] else
                  reference / "repeat" / f"adam_s-seed{seed}")
        old = read_json(folder / "config.json")
        if old["model"] != model_dict or old["torch_version"] != str(torch.__version__):
            parser.error("Model configuration or PyTorch version differs from reference")
        seed_all(seed)
        model = HybridLM(config)
        digest = hashlib.sha256()
        for name, p in model.named_parameters():
            digest.update(name.encode())
            digest.update(p.detach().numpy().tobytes())
        if digest.hexdigest() != old["initial_sha256"]:
            parser.error(f"Initial parameters differ from reference for seed {seed}")
        del model
    output = (args.output or ROOT / "outputs/metrics" /
              datetime.now(timezone.utc).strftime("theory-%Y%m%d-%H%M%S-%f")).resolve()
    command = [sys.executable, str(ROOT / "scripts/train_lm.py"), "--mode", "compare",
               "--optimizers", "adam_s_theory", "--data-dir", str(args.data_dir.resolve()),
               "--device", args.device, "--output", str(output)]
    for key in REPLAY_KEYS:
        value = launch[key]
        command.extend(["--" + key.replace("_", "-"),
                        *(str(v) for v in (value if isinstance(value, list) else [value]))])
    if args.no_plots:
        command.append("--no-plots")
    print("Reference data and all initial parameter hashes verified.", flush=True)
    print("Running only adam_s_theory with the original search and repeat budgets.", flush=True)
    subprocess.run(command, check=True)
    results = read_json(output / "summary.json")
    checks = [{"seed": row["seed"],
               "same_initialization": row["initial_sha256"] == baseline[row["seed"]]["initial_sha256"],
               "same_tokens": row["tokens"] == baseline[row["seed"]]["tokens"]}
              for row in results]
    (output / "reference_check.json").write_text(json.dumps({
        "reference": str(reference), "data_hashes_verified": True, "pairs": checks,
        "replayed_settings": {key: launch[key] for key in REPLAY_KEYS},
    }, indent=2), encoding="utf-8")
    if not all(c["same_initialization"] and c["same_tokens"] for c in checks):
        raise RuntimeError("Pairing check failed; inspect reference_check.json")
    print(f"Pairing verified. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
