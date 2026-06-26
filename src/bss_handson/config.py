from pathlib import Path

from omegaconf import OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    base_config = OmegaConf.load(path)
    override_config = OmegaConf.from_dotlist(overrides or [])
    config = OmegaConf.merge(base_config, override_config)
    return OmegaConf.to_container(config, resolve=True)


def save_config(config: dict, path: str | Path) -> None:
    OmegaConf.save(config=OmegaConf.create(config), f=path)
