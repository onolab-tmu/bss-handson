import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from fast_bss_eval import bss_eval_sources

from bss_handson import separate
from bss_handson.config import load_config, save_config
from bss_handson.data import load_cmu_arctic_sources
from bss_handson.simulation import simulate_room
from bss_handson.stft import create_stft, istft_sources, stft_channels

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

eval_length = min(sources.shape[1], estimates.shape[1])
sdr, sir, sar, perm = bss_eval_sources(
    sources[:, :eval_length],
    estimates[:, :eval_length],
    compute_permutation=True,
)
metrics = {
    "sdr": np.asarray(sdr).tolist(),
    "sir": np.asarray(sir).tolist(),
    "sar": np.asarray(sar).tolist(),
    "perm": np.asarray(perm).tolist(),
}

for prefix, signals_to_save in [
    ("source", sources),
    ("mixture", mixture),
    ("estimated", estimates),
]:
    for index, signal_data in enumerate(signals_to_save):
        sf.write(output_dir / f"{prefix}_{index}.wav", signal_data, fs)

stft = create_stft(fs=fs, **config["stft"])
for prefix, signals_to_plot, title in [
    ("mixture_spectrogram", mixture, "Mixture"),
    ("estimated_spectrogram", estimates, "Estimated sources"),
]:
    plot_spectra = stft.stft(signals_to_plot, axis=-1)
    freqs = stft.f
    times = stft.t(signals_to_plot.shape[1])
    for index, spectrum in enumerate(plot_spectra):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        power_db = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1.0e-10))
        image = ax.imshow(
            power_db,
            origin="lower",
            aspect="auto",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            vmin=config["plot"]["vmin_db"],
            vmax=config["plot"]["vmax_db"],
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{title} {index}")
        fig.colorbar(image, ax=ax, label="Magnitude (dB)")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_{index}.png")
        plt.close(fig)

save_config(config, output_dir / "config.yaml")
with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
