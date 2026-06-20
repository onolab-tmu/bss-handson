import numpy as np
import pyroomacoustics as pra


def simulate_room(
    sources: np.ndarray,
    fs: int,
    size: list[float],
    rt60: float,
    source_positions: list[list[float]],
    mic_positions: list[list[float]],
) -> np.ndarray:
    if len(source_positions) != sources.shape[0]:
        raise ValueError(
            f"source_positions must have one position per source: "
            f"{len(source_positions)} != {sources.shape[0]}"
        )
    if len(mic_positions) != sources.shape[0]:
        raise ValueError(
            f"this handson assumes determined BSS, so the number of microphones "
            f"must match the number of sources: {len(mic_positions)} != {sources.shape[0]}"
        )

    absorption, max_order = pra.inverse_sabine(rt60, size)
    room = pra.ShoeBox(
        size,
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
    return mixture
