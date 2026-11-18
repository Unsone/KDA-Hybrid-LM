import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from kda_hybrid_lm.train.plot import load_results, read_traces


ROOT = Path(__file__).resolve().parents[2]


def test_theory_supplement_replays_reference_and_relocated_logs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    tokens = np.tile(np.arange(8, dtype=np.uint16), 20)
    metadata = dict(vocab_size=8, token_counts={}, sha256={}, tokenizer_sha256="fixture")
    for split in ("train", "validation", "test"):
        np.save(data / f"{split}.npy", tokens)
        metadata["token_counts"][split] = len(tokens)
        metadata["sha256"][split] = hashlib.sha256(tokens).hexdigest()
    (data / "metadata.json").write_text(json.dumps(metadata))
    baseline, theory = tmp_path / "baseline", tmp_path / "theory"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/train_lm.py"), "--mode", "compare",
        "--optimizers", "adam_s", "--data-dir", str(data), "--output", str(baseline),
        "--device", "cpu", "--steps", "2", "--search-steps", "1", "--seeds", "42", "43",
        "--adam-lrs", "0.01", "--scaled-lrs", "0.01", "--batch-size", "2",
        "--accumulation", "1", "--length", "4", "--layers", "1", "--hidden-dim", "8",
        "--intermediate-dim", "16", "--heads", "2", "--eval-every", "2",
        "--eval-batches", "1", "--no-plots",
    ], check=True, capture_output=True, text=True)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_theory_control.py"),
        "--reference", str(baseline), "--data-dir", str(data), "--output", str(theory),
        "--device", "cpu", "--no-plots",
    ], check=True, capture_output=True, text=True)
    checks = json.loads((theory / "reference_check.json").read_text())
    assert all(r["same_initialization"] and r["same_tokens"] for r in checks["pairs"])
    assert len(json.loads((theory / "search.json").read_text())) == 1
    for root in (baseline, theory):
        rows = json.loads((root / "summary.json").read_text())
        for row in rows:
            row["path"] = "/server/path/no/longer/available"
        (root / "summary.json").write_text(json.dumps(rows))
    selected = load_results([baseline, theory], ["adam_s", "adam_s_theory"])
    assert len(selected) == 4
    keys, traces = read_traces(Path(selected[0]["path"]) / "weights.csv")
    assert "grad_rms" in keys
    assert traces["embedding.weight"].shape == (3, len(keys) + 1)
    # 数据不一致时在长训练开始前拒绝，不能无声复用不同数据。
    np.save(data / "train.npy", tokens + 1)
    rejected = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_theory_control.py"),
        "--reference", str(baseline), "--data-dir", str(data), "--device", "cpu",
    ], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "Token file hash differs" in rejected.stderr
