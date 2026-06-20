from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bss_handson.plot_style import apply_axis_style


def load_iteration_metrics(metrics_path: str | Path = "results/bss_example/iteration_metrics.json") -> pd.DataFrame:
    rows = []
    records = json.loads(Path(metrics_path).read_text())
    for record in records:
        iteration = int(record["iteration"])
        metrics = record["metrics"]
        for source_index, value in enumerate(metrics["sdr"]):
            rows.append(
                {
                    "iteration": iteration,
                    "source": str(source_index),
                    "sdr": value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", default="results/bss_example/iteration_metrics.json")
    parser.add_argument("--output", default="results/metrics_by_niter.png")
    parser.add_argument(
        "--style",
        nargs="+",
        default=["styles/common.mplstyle", "styles/paper.mplstyle"],
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_iteration_metrics(args.metrics_path)
    if df.empty:
        raise RuntimeError(f"{args.metrics_path} に反復回数ごとの評価結果がありません")

    plt.style.use(args.style)
    grid = sns.relplot(
        data=df,
        x="iteration",
        y="sdr",
        hue="source",
        kind="line",
        marker="o",
        errorbar=None,
        height=3.0,
        aspect=1.1,
    )
    plt.style.use(args.style)
    grid.figure.set_size_inches(plt.rcParams["figure.figsize"])
    grid.set_axis_labels("Number of iterations", "SDR (dB)")
    for ax in grid.axes.flat:
        ax.set_xlim(0, 50)
        ax.set_xticks(range(0, 51, 10))
        apply_axis_style(ax)
    grid.figure.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.86, wspace=0.45)
    grid.figure.savefig(output_path)
    plt.close(grid.figure)


if __name__ == "__main__":
    main()
