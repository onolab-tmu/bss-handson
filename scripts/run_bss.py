import argparse
import json
from pathlib import Path

import soundfile as sf

from bss_handson.auxiva import demix, project_back, separate
from bss_handson.config import (
    get_auxiva_config,
    get_dataset_config,
    get_output_dir,
    get_plot_config,
    get_room_config,
    get_stft_config,
    load_config,
    save_config,
)
from bss_handson.data import load_cmu_arctic_sources
from bss_handson.evaluation import evaluate_separation
from bss_handson.plot import save_spectrograms
from bss_handson.simulation import simulate_room
from bss_handson.stft import istft_sources, stft_channels


def save_json(data, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_indexed_audio(output_dir: Path, prefix: str, signals, fs: int) -> None:
    for index, signal in enumerate(signals):
        sf.write(output_dir / f"{prefix}_{index}.wav", signal, fs)


def evaluate_demixing_matrix(
    w,
    x,
    sources,
    mixture,
    fs: int,
    stft_config: dict,
    auxiva_config: dict,
) -> dict:
    y = demix(x, w)
    y = project_back(
        y,
        x,
        reference_mic=auxiva_config["reference_mic"],
        eps=auxiva_config["eps"],
    )
    estimates = istft_sources(
        y,
        fs=fs,
        n_samples=mixture.shape[1],
        **stft_config,
    )
    return evaluate_separation(sources, estimates)


def run_separation_pipeline(
    config: dict,
    evaluation_iterations: set[int] | None = None,
) -> dict:
    dataset_config = get_dataset_config(config)
    room_config = get_room_config(config)
    stft_config = get_stft_config(config)
    auxiva_config = get_auxiva_config(config)

    sources, source_fs = load_cmu_arctic_sources(**dataset_config)
    fs = room_config["fs"]
    if source_fs != fs:
        raise ValueError(f"source fs and room fs must match: {source_fs} != {fs}")

    mixture = simulate_room(
        sources=sources,
        **room_config,
    )
    x, _, _ = stft_channels(
        mixture,
        fs=fs,
        **stft_config,
    )

    iteration_metrics = []

    def evaluate_iteration(iteration: int, w) -> None:
        if evaluation_iterations is None or iteration not in evaluation_iterations:
            return
        metrics_at_iteration = evaluate_demixing_matrix(
            w,
            x,
            sources,
            mixture,
            fs,
            stft_config,
            auxiva_config,
        )
        iteration_metrics.append(
            {
                "iteration": iteration,
                "metrics": metrics_at_iteration,
            }
        )

    y, w = separate(
        x,
        callback=evaluate_iteration if evaluation_iterations is not None else None,
        **auxiva_config,
    )
    estimates = istft_sources(
        y,
        fs=fs,
        n_samples=mixture.shape[1],
        **stft_config,
    )
    metrics = evaluate_separation(sources, estimates)

    return {
        "fs": fs,
        "sources": sources,
        "mixture": mixture,
        "spectra": x,
        "separated_spectra": y,
        "demixing_matrix": w,
        "estimates": estimates,
        "metrics": metrics,
        "iteration_metrics": iteration_metrics,
    }


def save_run_outputs(output_dir: Path, config: dict, result: dict) -> None:
    fs = result["fs"]
    save_indexed_audio(output_dir, "source", result["sources"], fs)
    save_indexed_audio(output_dir, "mixture", result["mixture"], fs)
    save_indexed_audio(output_dir, "estimated", result["estimates"], fs)

    spectrogram_config = {**get_stft_config(config), **get_plot_config(config)}
    save_spectrograms(
        result["mixture"],
        fs=fs,
        output_path=output_dir / "mixture_spectrogram.png",
        title="Mixture",
        **spectrogram_config,
    )
    save_spectrograms(
        result["estimates"],
        fs=fs,
        output_path=output_dir / "estimated_spectrogram.png",
        title="Estimated sources",
        **spectrogram_config,
    )

    save_config(config, output_dir / "config.yaml")
    save_json(result["metrics"], output_dir / "metrics.json")
    save_json(result["iteration_metrics"], output_dir / "iteration_metrics.json")


def run_bss(config_path: str | Path, overrides: list[str] | None = None) -> None:
    config = load_config(config_path, overrides)
    output_dir = get_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    auxiva_config = get_auxiva_config(config)
    evaluation_iterations = set(range(0, auxiva_config["n_iter"] + 1, 10))
    result = run_separation_pipeline(
        config,
        evaluation_iterations=evaluation_iterations,
    )
    save_run_outputs(output_dir, config, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run_bss(args.config, args.overrides)


if __name__ == "__main__":
    main()
