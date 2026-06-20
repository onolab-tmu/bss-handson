import numpy as np
from fast_bss_eval import bss_eval_sources


def evaluate_separation(
    references: np.ndarray,
    estimates: np.ndarray,
) -> dict[str, list[float]]:
    length = min(references.shape[1], estimates.shape[1])
    references = references[:, :length]
    estimates = estimates[:, :length]

    sdr, sir, sar, perm = bss_eval_sources(
        references,
        estimates,
        compute_permutation=True,
    )
    return {
        "sdr": np.asarray(sdr).tolist(),
        "sir": np.asarray(sir).tolist(),
        "sar": np.asarray(sar).tolist(),
        "perm": np.asarray(perm).tolist(),
    }
