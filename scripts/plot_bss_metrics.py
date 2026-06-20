from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bss_handson.plot_style import apply_axis_style


def load_metrics(
    results_dir: str | Path = "results",
    pattern: str = "bss_*/metrics.json",
) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(results_dir).glob(pattern)):
        metrics = json.loads(path.read_text())
        experiment = path.parent.name
        for source_index, value in enumerate(metrics["sdr"]):
            rows.append(
                {
                    "experiment": experiment,
                    "source": str(source_index),
                    "sdr": value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--pattern", default="bss_*/metrics.json")
    parser.add_argument("--output", default="results/metrics_summary.png")
    parser.add_argument(
        "--style",
        nargs="+",
        default=["styles/common.mplstyle", "styles/paper.mplstyle"],
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_metrics(args.results_dir, args.pattern)
    if df.empty:
        raise RuntimeError(f"{args.results_dir}/{args.pattern} が見つかりません")

    plt.style.use(args.style)
    grid = sns.catplot(
        data=df,
        x="experiment",
        y="sdr",
        hue="source",
        kind="bar",
        errorbar=None,
        height=3.0,
        aspect=1.1,
    )
    plt.style.use(args.style)
    grid.figure.set_size_inches(plt.rcParams["figure.figsize"])
    grid.set_axis_labels("Experiment", "SDR (dB)")
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=30)
        apply_axis_style(ax)
    grid.figure.subplots_adjust(left=0.12, right=0.98, bottom=0.25, top=0.86, wspace=0.45)
    grid.figure.savefig(output_path)
    plt.close(grid.figure)


if __name__ == "__main__":
    main()
