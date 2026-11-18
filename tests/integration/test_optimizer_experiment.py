import csv
import json

import numpy as np

from kda_hybrid_lm.models import ModelConfig
from kda_hybrid_lm.train.trainer import run_training


def test_paired_runs_record_same_initialization_data_and_real_updates(tmp_path):
    config = ModelConfig(vocab_size=8, hidden_dim=8, intermediate_dim=16,
                         num_heads=2, layer_types=("softmax",), dropout=0.2)
    data = np.tile(np.arange(8, dtype=np.uint16), 20)
    results = []
    first_losses = []
    for kind in ("adam", "adam_s", "shuffled", "adam_s_theory"):
        path = tmp_path / kind
        results.append(run_training(
            config, data, data, path, kind=kind, lr=0.01, seed=13, shuffle_seed=29,
            steps=2, batch_size=2, accumulation=2, length=4,
            eval_every=2, eval_batches=2, device="cpu"))
        rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        first_losses.append(rows[1]["train_loss"])
        assert results[-1]["tokens"] == 32
        assert rows[-1]["validation"]["tokens"] == 16
        with (path / "weights.csv").open() as source:
            weights = list(csv.DictReader(source))
        initial = [r for r in weights if r["step"] == "0"]
        updates = [r for r in weights if r["step"] == "1"]
        assert all(float(r["norm_ratio"]) == 1 for r in initial)
        assert all(float(r["relative_update"]) > 0 for r in updates)
        assert (path / "last.pt").exists()
    assert len({r["initial_sha256"] for r in results}) == 1
    # 第一步损失相同同时覆盖模型初始化、数据采样和 Dropout RNG 的配对。
    assert len(set(first_losses)) == 1
