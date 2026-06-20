import numpy as np
from scipy import signal


def create_stft(
    fs: int,
    window: str,
    win_length: int,
    hop: int,
) -> signal.ShortTimeFFT:
    win = signal.get_window(window, win_length)
    return signal.ShortTimeFFT(
        win,
        hop=hop,
        fs=fs,
    )


def stft_channels(
    signals: np.ndarray,
    fs: int,
    window: str,
    win_length: int,
    hop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stft = create_stft(fs, window, win_length, hop)
    spectra = stft.stft(signals, axis=-1)
    x = spectra.transpose(2, 1, 0)
    return x, stft.f, stft.t(signals.shape[1])


def istft_sources(
    spectra: np.ndarray,
    fs: int,
    window: str,
    win_length: int,
    hop: int,
    n_samples: int,
) -> np.ndarray:
    stft = create_stft(fs, window, win_length, hop)
    spectra_sources_first = spectra.transpose(2, 1, 0)
    return stft.istft(
        spectra_sources_first,
        k1=n_samples,
        f_axis=-2,
        t_axis=-1,
    )
