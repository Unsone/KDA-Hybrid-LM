"""重绘一个或多个已完成实验，可筛选优化器，支持从服务器下载的结果。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kda_hybrid_lm.train.optim import OPTIMIZERS
from kda_hybrid_lm.train.plot import load_results, plot_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="Experiment directories containing summary.json")
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS,
                        default=["adam", "adam_s"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = load_results(args.runs, args.optimizers)
    output = args.output or args.runs[0] / ("figures_" + "_".join(args.optimizers))
    print(f"Plotting {len(results)} runs into {output}", flush=True)
    plot_results(results, output)
    (output / "plot_manifest.json").write_text(json.dumps({
        "runs": [str(p.resolve()) for p in args.runs], "optimizers": args.optimizers,
        "selected": [{k: r[k] for k in ("optimizer", "seed", "lr", "initial_sha256", "path")}
                     for r in results],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
