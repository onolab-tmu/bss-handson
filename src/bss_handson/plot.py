from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bss_handson.stft import create_stft


def _indexed_path(path: str | Path, index: int) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_{index}{path.suffix}")


def save_spectrograms(
    signals: np.ndarray,
    fs: int,
    output_path: str | Path,
    title: str,
    style: str | list[str],
    window: str,
    win_length: int,
    hop: int,
    vmin_db: float,
    vmax_db: float,
) -> None:
    plt.style.use(style)
    stft = create_stft(fs=fs, window=window, win_length=win_length, hop=hop)
    freqs = stft.f
    times = stft.t(signals.shape[1])
    spectra = stft.stft(signals, axis=-1)

    for index, spectrum in enumerate(spectra):
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
        fig.savefig(_indexed_path(output_path, index))
        plt.close(fig)
