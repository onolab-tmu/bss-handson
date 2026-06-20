"""Blind source separation demo package."""

from bss_handson.auxiva import (
    separate,
    demix,
    project_back,
    source_weights,
    weighted_covariance,
    objective,
    ip_update,
    iss_update,
)

__all__ = [
    "separate",
    "demix",
    "project_back",
    "source_weights",
    "weighted_covariance",
    "objective",
    "ip_update",
    "iss_update",
]
