from collections.abc import Callable

import numpy as np


def demix(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    n_frames, n_freq, n_channels = x.shape
    y = np.empty((n_frames, n_freq, n_channels), dtype=np.complex128)

    for t in range(n_frames):
        for f in range(n_freq):
            y[t, f] = w[f] @ x[t, f]

    return y


def source_weights(
    y: np.ndarray,
    model: str = "laplace",
    eps: float = 1.0e-10,
) -> np.ndarray:
    power_sum = np.sum(np.abs(y) ** 2, axis=1)
    r = np.sqrt(np.maximum(power_sum, eps))

    if model == "laplace":
        varphi = 1.0 / (2.0 * r)
    elif model == "gauss":
        varphi = y.shape[1] / (r**2)
    else:
        raise ValueError(f"model must be 'laplace' or 'gauss': {model}")

    return varphi


def weighted_covariance(
    x: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    n_frames, n_freq, n_channels = x.shape

    v = np.empty((n_channels, n_freq, n_channels, n_channels), dtype=np.complex128)
    for k in range(n_channels):
        for f in range(n_freq):
            v_kf = np.zeros((n_channels, n_channels), dtype=np.complex128)
            for t in range(n_frames):
                x_tf = x[t, f, :, None]
                v_kf += weights[t, k] * (x_tf @ x_tf.conj().T)
            v[k, f] = v_kf / n_frames

    return v


def objective(w: np.ndarray, v: np.ndarray) -> float:
    n_channels, n_freq = v.shape[:2]
    quadratic = 0.0

    for k in range(n_channels):
        for f in range(n_freq):
            wk = w[f, k].conj()
            quadratic += float(np.real(wk.conj() @ v[k, f] @ wk))

    logdet = 0.0
    for w_f in w:
        sign, logabsdet = np.linalg.slogdet(w_f)
        if sign == 0:
            return float("inf")
        logdet += logabsdet

    return float(quadratic - 2.0 * logdet)


def ip_update(
    x: np.ndarray,
    w: np.ndarray,
    model: str = "laplace",
    eps: float = 1.0e-10,
) -> np.ndarray:
    _, n_freq, n_channels = x.shape
    y = demix(x, w)
    weights = source_weights(y, model=model, eps=eps)
    v = weighted_covariance(x, weights)

    w_new = w.copy()

    for k in range(n_channels):
        for f in range(n_freq):
            eye_k = np.zeros(n_channels, dtype=np.complex128)
            eye_k[k] = 1.0
            wk = np.linalg.solve(w_new[f] @ v[k, f], eye_k)
            denom_sq = max(float(np.real(wk.conj() @ v[k, f] @ wk)), eps)
            w_new[f, k, :] = (wk / np.sqrt(denom_sq)).conj()

    return w_new


def iss_update(
    x: np.ndarray,
    w: np.ndarray,
    model: str = "laplace",
    eps: float = 1.0e-10,
) -> np.ndarray:
    _, n_freq, n_channels = x.shape
    y = demix(x, w)
    weights = source_weights(y, model=model, eps=eps)

    w_new = w.copy()
    y_new = y.copy()

    for k in range(n_channels):
        for f in range(n_freq):
            y_k = y_new[:, f, k].copy()
            w_k = w_new[f, k, :].copy()
            power_k = np.abs(y_k) ** 2

            for j in range(n_channels):
                denom = float(np.mean(weights[:, j] * power_k))
                denom = max(denom, eps)

                if j == k:
                    coeff = 1.0 - 1.0 / np.sqrt(denom)
                else:
                    numerator = np.mean(weights[:, j] * y_new[:, f, j] * y_k.conj())
                    coeff = numerator / denom

                y_new[:, f, j] -= coeff * y_k
                w_new[f, j, :] -= coeff * w_k

    return w_new


def project_back(
    y: np.ndarray,
    x: np.ndarray,
    reference_mic: int = 0,
    eps: float = 1.0e-10,
) -> np.ndarray:
    n_frames, n_freq, n_channels = y.shape
    y_scaled = np.empty_like(y)

    for f in range(n_freq):
        reference = x[:, f, reference_mic]
        for k in range(n_channels):
            separated = y[:, f, k]
            numerator = np.sum(reference * separated.conj())
            denominator = np.sum(np.abs(separated) ** 2)
            scale = numerator / max(float(denominator), eps)
            y_scaled[:, f, k] = scale * separated

    return y_scaled


def separate(
    x: np.ndarray,
    n_iter: int = 50,
    model: str = "laplace",
    update_method: str = "ip",
    eps: float = 1.0e-10,
    reference_mic: int = 0,
    callback: Callable[[int, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _, n_freq, n_channels = x.shape
    w = np.tile(np.eye(n_channels, dtype=np.complex128), (n_freq, 1, 1))
    update = {
        "ip": ip_update,
        "iss": iss_update,
    }.get(update_method)
    if update is None:
        raise ValueError(f"update_method must be 'ip' or 'iss': {update_method}")

    if callback is not None:
        callback(0, w.copy())

    for iteration in range(1, n_iter + 1):
        w = update(x, w, model=model, eps=eps)
        if callback is not None:
            callback(iteration, w.copy())

    y = demix(x, w)
    y = project_back(y, x, reference_mic=reference_mic, eps=eps)
    return y, w
