from pathlib import Path
import argparse
import csv
import json

import numpy as np
from omegaconf import OmegaConf

from bss_handson.io import save_json
from bss_handson.pipeline import run_separation_pipeline


def sample_source_positions(
    rng: np.random.Generator,
    room_size: list[float],
    n_sources: int,
    margin: float,
    min_distance: float,
) -> list[list[float]]:
    positions: list[list[float]] = []
    room_width, room_depth = room_size
    while len(positions) < n_sources:
        candidate = np.array(
            [
                rng.uniform(margin, room_width - margin),
                rng.uniform(margin, room_depth - margin),
            ]
        )
        if all(
            np.linalg.norm(candidate - np.array(position)) >= min_distance
            for position in positions
        ):
            positions.append(candidate.tolist())
    return positions


def run_trial(config: dict, rng: np.random.Generator, max_utterance_index: int) -> dict:
    dataset_config = config["dataset"]
    room_config = config["room"]
    n_sources = len(dataset_config["speakers"])
    utterance_indices = rng.integers(
        0,
        max_utterance_index + 1,
        size=n_sources,
    ).tolist()
    source_positions = sample_source_positions(
        rng=rng,
        room_size=room_config["size"],
        n_sources=n_sources,
        margin=config["random_eval"]["position_margin"],
        min_distance=config["random_eval"]["min_source_distance"],
    )

    trial_config = {
        **config,
        "dataset": {**dataset_config, "utterance_indices": utterance_indices},
        "room": {**room_config, "source_positions": source_positions},
    }
    result = run_separation_pipeline(trial_config)
    return {
        "utterance_indices": utterance_indices,
        "source_positions": source_positions,
        "metrics": result["metrics"],
    }


def summarize_trials(trials: list[dict]) -> dict:
    summary = {}
    for metric_name in ["sdr", "sir", "sar"]:
        values = np.asarray(
            [trial["metrics"][metric_name] for trial in trials], dtype=np.float64
        )
        summary[metric_name] = {
            "mean_per_source": values.mean(axis=0).tolist(),
            "std_per_source": values.std(axis=0).tolist(),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return summary


def save_trial_csv(trials: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial",
                "source",
                "utterance_index",
                "source_position",
                "sdr",
                "sir",
                "sar",
                "perm",
            ],
        )
        writer.writeheader()
        for trial_index, trial in enumerate(trials):
            metrics = trial["metrics"]
            for source_index, utterance_index in enumerate(trial["utterance_indices"]):
                writer.writerow(
                    {
                        "trial": trial_index,
                        "source": source_index,
                        "utterance_index": utterance_index,
                        "source_position": json.dumps(
                            trial["source_positions"][source_index]
                        ),
                        "sdr": metrics["sdr"][source_index],
                        "sir": metrics["sir"][source_index],
                        "sar": metrics["sar"][source_index],
                        "perm": metrics["perm"][source_index],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bss.yaml")
    parser.add_argument("--output-dir", default="results/random_eval")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-utterance-index", type=int, default=99)
    args = parser.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    config["random_eval"] = {
        "position_margin": 0.6,
        "min_source_distance": 1.0,
    }
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = [
        run_trial(
            config=config,
            rng=rng,
            max_utterance_index=args.max_utterance_index,
        )
        for _ in range(args.n_trials)
    ]
    result = {
        "n_trials": args.n_trials,
        "seed": args.seed,
        "config": config,
        "summary": summarize_trials(trials),
        "trials": trials,
    }

    save_json(result, output_dir / "random_average_metrics.json")
    save_trial_csv(trials, output_dir / "random_average_metrics.csv")

    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
