import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from fast_bss_eval import bss_eval_sources
from scipy import signal

from bss_handson.config import load_config, save_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/bss.yaml")
parser.add_argument("overrides", nargs="*")
args = parser.parse_args()

config = load_config(args.config, args.overrides)

output_dir = Path(config["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

dataset_config = config["dataset"]
dataset_basedir = Path(dataset_config["basedir"])
speakers = dataset_config["speakers"]
utterance_indices = dataset_config["utterance_indices"]

room_config = config["room"]
fs = room_config["fs"]
room_size = room_config["size"]
rt60 = room_config["rt60"]
source_positions = room_config["source_positions"]
mic_positions = room_config["mic_positions"]

stft_config = config["stft"]
window = stft_config["window"]
win_length = stft_config["win_length"]
hop = stft_config["hop"]

auxiva_config = config["auxiva"]
n_iter = auxiva_config["n_iter"]
model = auxiva_config["model"]
update_method = auxiva_config["update_method"]
eps = auxiva_config["eps"]
reference_mic = auxiva_config["reference_mic"]

plot_config = config["plot"]
vmin_db = plot_config["vmin_db"]
vmax_db = plot_config["vmax_db"]

if update_method != "ip":
    raise ValueError(f"this version only supports update_method='ip': {update_method}")

signals = []
sample_rates = []
dataset_basedir.mkdir(parents=True, exist_ok=True)

for speaker, index in zip(speakers, utterance_indices, strict=True):
    corpus = pra.datasets.CMUArcticCorpus(
        basedir=str(dataset_basedir),
        download=True,
        speaker=[speaker],
    )
    sentence = corpus[index]
    signal_data = np.asarray(sentence.data, dtype=np.float64)
    signal_data = signal_data / max(np.max(np.abs(signal_data)), 1.0e-12)
    signals.append(signal_data)
    sample_rates.append(sentence.fs)

if len(set(sample_rates)) != 1:
    raise ValueError(f"sample rates must be identical: {sample_rates}")

source_fs = sample_rates[0]
if source_fs != fs:
    raise ValueError(f"source fs and room fs must match: {source_fs} != {fs}")

length = min(len(signal_data) for signal_data in signals)
sources = np.stack([signal_data[:length] for signal_data in signals], axis=0)

if len(source_positions) != sources.shape[0]:
    raise ValueError(
        f"source_positions must have one position per source: "
        f"{len(source_positions)} != {sources.shape[0]}"
    )
if len(mic_positions) != sources.shape[0]:
    raise ValueError(
        f"number of microphones must match number of sources: "
        f"{len(mic_positions)} != {sources.shape[0]}"
    )

absorption, max_order = pra.inverse_sabine(rt60, room_size)
room = pra.ShoeBox(
    room_size,
    fs=fs,
    materials=pra.Material(absorption),
    max_order=max_order,
)

for source, position in zip(sources, source_positions, strict=True):
    room.add_source(position, signal=source)

mic_array = np.asarray(mic_positions, dtype=np.float64).T
room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))
room.simulate()

mixture = np.asarray(room.mic_array.signals, dtype=np.float64)
mixture = mixture / max(np.max(np.abs(mixture)), 1.0e-12)

win = signal.get_window(window, win_length)
stft = signal.ShortTimeFFT(win, hop=hop, fs=fs)
mixture_spectra = stft.stft(mixture, axis=-1)
x = mixture_spectra.transpose(2, 1, 0)

n_frames, n_freq, n_channels = x.shape
w = np.tile(np.eye(n_channels, dtype=np.complex128), (n_freq, 1, 1))

for _ in range(n_iter):
    y = np.empty((n_frames, n_freq, n_channels), dtype=np.complex128)
    for t in range(n_frames):
        for f in range(n_freq):
            y[t, f] = w[f] @ x[t, f]

    power_sum = np.sum(np.abs(y) ** 2, axis=1)
    r = np.sqrt(np.maximum(power_sum, eps))
    if model == "laplace":
        weights = 1.0 / (2.0 * r)
    elif model == "gauss":
        weights = n_freq / (r**2)
    else:
        raise ValueError(f"model must be 'laplace' or 'gauss': {model}")

    v = np.empty((n_channels, n_freq, n_channels, n_channels), dtype=np.complex128)
    for k in range(n_channels):
        for f in range(n_freq):
            v_kf = np.zeros((n_channels, n_channels), dtype=np.complex128)
            for t in range(n_frames):
                x_tf = x[t, f, :, None]
                v_kf += weights[t, k] * (x_tf @ x_tf.conj().T)
            v[k, f] = v_kf / n_frames

    w_new = w.copy()
    for k in range(n_channels):
        for f in range(n_freq):
            eye_k = np.zeros(n_channels, dtype=np.complex128)
            eye_k[k] = 1.0
            wk = np.linalg.solve(w_new[f] @ v[k, f], eye_k)
            denom_sq = max(float(np.real(wk.conj() @ v[k, f] @ wk)), eps)
            w_new[f, k, :] = (wk / np.sqrt(denom_sq)).conj()
    w = w_new

y = np.empty((n_frames, n_freq, n_channels), dtype=np.complex128)
for t in range(n_frames):
    for f in range(n_freq):
        y[t, f] = w[f] @ x[t, f]

y_scaled = np.empty_like(y)
for f in range(n_freq):
    reference = x[:, f, reference_mic]
    for k in range(n_channels):
        separated = y[:, f, k]
        numerator = np.sum(reference * separated.conj())
        denominator = np.sum(np.abs(separated) ** 2)
        scale = numerator / max(float(denominator), eps)
        y_scaled[:, f, k] = scale * separated

spectra_sources_first = y_scaled.transpose(2, 1, 0)
estimates = stft.istft(
    spectra_sources_first,
    k1=mixture.shape[1],
    f_axis=-2,
    t_axis=-1,
)

eval_length = min(sources.shape[1], estimates.shape[1])
references_eval = sources[:, :eval_length]
estimates_eval = estimates[:, :eval_length]
sdr, sir, sar, perm = bss_eval_sources(
    references_eval,
    estimates_eval,
    compute_permutation=True,
)
metrics = {
    "sdr": np.asarray(sdr).tolist(),
    "sir": np.asarray(sir).tolist(),
    "sar": np.asarray(sar).tolist(),
    "perm": np.asarray(perm).tolist(),
}

for index, signal_data in enumerate(sources):
    sf.write(output_dir / f"source_{index}.wav", signal_data, fs)
for index, signal_data in enumerate(mixture):
    sf.write(output_dir / f"mixture_{index}.wav", signal_data, fs)
for index, signal_data in enumerate(estimates):
    sf.write(output_dir / f"estimated_{index}.wav", signal_data, fs)

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
            vmin=vmin_db,
            vmax=vmax_db,
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
