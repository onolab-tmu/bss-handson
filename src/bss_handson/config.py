from pathlib import Path

from omegaconf import OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    base_config = OmegaConf.load(path)
    override_config = OmegaConf.from_dotlist(overrides or [])
    config = OmegaConf.merge(base_config, override_config)
    return OmegaConf.to_container(config, resolve=True)


def save_config(config: dict, path: str | Path) -> None:
    OmegaConf.save(config=OmegaConf.create(config), f=path)


def get_output_dir(config: dict) -> Path:
    return Path(config["output_dir"])


def get_dataset_config(config: dict) -> dict:
    return dict(config["dataset"])


def get_room_config(config: dict) -> dict:
    return dict(config["room"])


def get_stft_config(config: dict) -> dict:
    return dict(config["stft"])


def get_auxiva_config(config: dict) -> dict:
    return dict(config["auxiva"])


def get_plot_config(config: dict) -> dict:
    return dict(config["plot"])
