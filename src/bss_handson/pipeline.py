import numpy as np

from bss_handson.auxiva import demix, project_back, separate
from bss_handson.data import load_cmu_arctic_sources
from bss_handson.evaluation import evaluate_separation
from bss_handson.simulation import simulate_room
from bss_handson.stft import istft_sources, stft_channels


def reconstruct_sources(
    spectra: np.ndarray,
    mixture: np.ndarray,
    fs: int,
    stft_config: dict,
) -> np.ndarray:
    return istft_sources(
        spectra,
        fs=fs,
        n_samples=mixture.shape[1],
        **stft_config,
    )


def estimate_sources_from_demixing(
    w,
    x,
    mixture,
    fs: int,
    stft_config: dict,
    auxiva_config: dict,
) -> np.ndarray:
    y = demix(x, w)
    y = project_back(
        y,
        x,
        reference_mic=auxiva_config["reference_mic"],
        eps=auxiva_config["eps"],
    )
    return reconstruct_sources(y, mixture, fs, stft_config)


def evaluate_demixed_sources(
    w,
    x,
    sources,
    mixture,
    fs: int,
    stft_config: dict,
    auxiva_config: dict,
) -> dict:
    estimates = estimate_sources_from_demixing(
        w,
        x,
        mixture,
        fs,
        stft_config,
        auxiva_config,
    )
    return evaluate_separation(sources, estimates)


def run_separation_pipeline(
    config: dict,
    evaluation_iterations: set[int] | None = None,
) -> dict:
    dataset_config = config["dataset"]
    room_config = config["room"]
    stft_config = config["stft"]
    auxiva_config = config["auxiva"]

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
        metrics_at_iteration = evaluate_demixed_sources(
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
    estimates = reconstruct_sources(y, mixture, fs, stft_config)
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
