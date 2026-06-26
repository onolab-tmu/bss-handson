import argparse
import json
from pathlib import Path

import soundfile as sf

from bss_handson import separate
from bss_handson.config import load_config, save_config
from bss_handson.data import load_cmu_arctic_sources
from bss_handson.evaluation import evaluate_separation
from bss_handson.plot import save_spectrograms
from bss_handson.simulation import simulate_room
from bss_handson.stft import istft_sources, stft_channels

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/bss.yaml")
parser.add_argument("overrides", nargs="*")
args = parser.parse_args()

config = load_config(args.config, args.overrides)
output_dir = Path(config["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

sources, source_fs = load_cmu_arctic_sources(**config["dataset"])
fs = config["room"]["fs"]
if source_fs != fs:
    raise ValueError(f"source fs and room fs must match: {source_fs} != {fs}")

mixture = simulate_room(sources=sources, **config["room"])
x, _, _ = stft_channels(mixture, fs=fs, **config["stft"])
y, _ = separate(x, **config["auxiva"])
estimates = istft_sources(y, fs=fs, n_samples=mixture.shape[1], **config["stft"])
metrics = evaluate_separation(sources, estimates)

for prefix, signals_to_save in [
    ("source", sources),
    ("mixture", mixture),
    ("estimated", estimates),
]:
    for index, signal_data in enumerate(signals_to_save):
        sf.write(output_dir / f"{prefix}_{index}.wav", signal_data, fs)

spectrogram_config = {**config["stft"], **config["plot"]}
save_spectrograms(
    mixture,
    fs=fs,
    output_path=output_dir / "mixture_spectrogram.png",
    title="Mixture",
    **spectrogram_config,
)
save_spectrograms(
    estimates,
    fs=fs,
    output_path=output_dir / "estimated_spectrogram.png",
    title="Estimated sources",
    **spectrogram_config,
)

save_config(config, output_dir / "config.yaml")
with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
