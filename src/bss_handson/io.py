import json
from pathlib import Path

import soundfile as sf


def save_json(data, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_numbered_wavs(output_dir: Path, prefix: str, signals, fs: int) -> None:
    for index, signal in enumerate(signals):
        sf.write(output_dir / f"{prefix}_{index}.wav", signal, fs)
