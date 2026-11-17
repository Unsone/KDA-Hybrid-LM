import csv
import math

import pytest
import torch
from torch import nn

from kda_hybrid_lm.train.optim import build_optimizer
from kda_hybrid_lm.train.trainer import WeightTrace


@pytest.mark.parametrize("kind", ["adam", "adam_s", "shuffled"])
def test_trace_matches_gradient_moments_and_actual_update(tmp_path, kind):
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.bias.zero_()
    optimizer, _ = build_optimizer(model, kind, 0.01)
    trace = WeightTrace(model, optimizer, tmp_path / "trace.csv")
    before = trace.snapshot()
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    model.bias.grad = torch.tensor([-2.0])
    optimizer.step()
    trace.write(1, before)
    trace.close()
    with (tmp_path / "trace.csv").open() as source:
        rows = list(csv.DictReader(source))
    initial = {r["parameter"]: r for r in rows if r["step"] == "0"}
    updated = {r["parameter"]: r for r in rows if r["step"] == "1"}
    assert set(updated) == {"weight", "bias"}
    assert float(initial["weight"]["momentum_norm"]) == 0
    assert math.isnan(float(initial["weight"]["grad_norm"]))
    assert math.isnan(float(initial["weight"]["momentum_hat_norm"]))
    assert math.isnan(float(updated["bias"]["norm_ratio"]))
    assert math.isnan(float(updated["bias"]["relative_update"]))
    row = updated["weight"]
    expected = dict(
        grad_norm=5, grad_rms=5 / math.sqrt(2),
        momentum_norm=0.5, momentum_hat_norm=5,
        second_moment_norm=math.sqrt(0.009 ** 2 + 0.016 ** 2),
        second_moment_hat_norm=math.sqrt(9 ** 2 + 16 ** 2),
        sqrt_second_moment_hat_rms=math.sqrt(12.5),
        adam_direction_norm=math.sqrt(2), pre_norm=math.sqrt(5),
    )
    for key, value in expected.items():
        assert float(row[key]) == pytest.approx(value, rel=1e-5), key
    assert int(row["parameter_step"]) == 1
    actual_delta = (model.weight.detach() - before[0]).norm().item()
    assert float(row["update_norm"]) == pytest.approx(actual_delta)
    assert float(row["relative_update"]) == pytest.approx(actual_delta / math.sqrt(5))
    assert actual_delta == pytest.approx(float(row["effective_lr"]) * math.sqrt(2), rel=1e-5)


def test_trace_uses_parameter_step_and_marks_missing_gradient(tmp_path):
    model = nn.Linear(2, 1)
    optimizer, _ = build_optimizer(model, "adam", 0.01)
    trace = WeightTrace(model, optimizer, tmp_path / "trace.csv")
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        before = trace.snapshot()
        if step == 1:
            model.weight.grad = torch.ones_like(model.weight)
        model.bias.grad = torch.ones_like(model.bias)
        optimizer.step()
        trace.write(step, before)
    trace.close()
    with (tmp_path / "trace.csv").open() as source:
        row = next(r for r in csv.DictReader(source) if r["step"] == "2" and r["parameter"] == "weight")
    assert int(row["parameter_step"]) == 1
    assert float(row["momentum_hat_norm"]) == pytest.approx(math.sqrt(2))
    assert math.isnan(float(row["grad_norm"]))
    assert math.isnan(float(row["adam_direction_norm"]))
    assert float(row["update_norm"]) == 0
