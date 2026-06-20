from pathlib import Path

import numpy as np
import pyroomacoustics as pra


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
        corpus = pra.datasets.CMUArcticCorpus(
            basedir=str(basedir),
            download=True,
            speaker=[speaker],
        )
        sentence = corpus[index]
        signal = np.asarray(sentence.data, dtype=np.float64)
        signal = signal / max(np.max(np.abs(signal)), 1.0e-12)
        signals.append(signal)
        sample_rates.append(sentence.fs)

    if len(set(sample_rates)) != 1:
        raise ValueError(f"sample rates must be identical: {sample_rates}")

    length = min(len(signal) for signal in signals)
    sources = np.stack([signal[:length] for signal in signals], axis=0)
    return sources, sample_rates[0]
