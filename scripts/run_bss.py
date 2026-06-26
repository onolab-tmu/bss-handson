import argparse
from pathlib import Path

from bss_handson.config import load_config, save_config
from bss_handson.io import save_json, save_numbered_wavs
from bss_handson.pipeline import run_separation_pipeline
from bss_handson.plot import save_spectrograms


def save_audio_outputs(output_dir: Path, result: dict) -> None:
    fs = result["fs"]
    save_numbered_wavs(output_dir, "source", result["sources"], fs)
    save_numbered_wavs(output_dir, "mixture", result["mixture"], fs)
    save_numbered_wavs(output_dir, "estimated", result["estimates"], fs)


def save_spectrogram_outputs(output_dir: Path, config: dict, result: dict) -> None:
    fs = result["fs"]
    spectrogram_config = {**config["stft"], **config["plot"]}
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


def save_metadata_outputs(output_dir: Path, config: dict, result: dict) -> None:
    save_config(config, output_dir / "config.yaml")
    save_json(result["metrics"], output_dir / "metrics.json")
    save_json(result["iteration_metrics"], output_dir / "iteration_metrics.json")


def save_run_outputs(output_dir: Path, config: dict, result: dict) -> None:
    save_audio_outputs(output_dir, result)
    save_spectrogram_outputs(output_dir, config, result)
    save_metadata_outputs(output_dir, config, result)


def run_bss(config_path: str | Path, overrides: list[str] | None = None) -> None:
    config = load_config(config_path, overrides)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    auxiva_config = config["auxiva"]
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
