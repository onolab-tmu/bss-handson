from pathlib import Path

import numpy as np
import pyroomacoustics as pra


def normalize_signal(signal: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    return signal / max(np.max(np.abs(signal)), eps)


def load_cmu_arctic_utterance(
    basedir: str | Path,
    speaker: str,
    utterance_index: int,
) -> tuple[np.ndarray, int]:
    corpus = pra.datasets.CMUArcticCorpus(
        basedir=str(basedir),
        download=True,
        speaker=[speaker],
    )
    sentence = corpus[utterance_index]
    signal = np.asarray(sentence.data, dtype=np.float64)
    return normalize_signal(signal), sentence.fs


def trim_to_shortest(signals: list[np.ndarray]) -> np.ndarray:
    length = min(len(signal) for signal in signals)
    return np.stack([signal[:length] for signal in signals], axis=0)


def load_cmu_arctic_sources(
    basedir: str | Path,
    speakers: list[str],
    utterance_indices: list[int],
) -> tuple[np.ndarray, int]:
    basedir = Path(basedir)
    basedir.mkdir(parents=True, exist_ok=True)
    signals = []
    sample_rates = []

    for speaker, index in zip(speakers, utterance_indices, strict=True):
        signal, sample_rate = load_cmu_arctic_utterance(basedir, speaker, index)
        signals.append(signal)
        sample_rates.append(sample_rate)

    if len(set(sample_rates)) != 1:
        raise ValueError(f"sample rates must be identical: {sample_rates}")

    return trim_to_shortest(signals), sample_rates[0]
