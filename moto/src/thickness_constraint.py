"""PDE-based maximum-thickness constraint for density-based MPM designs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ProjectionVolumeMode = Literal["reference", "current"]


@dataclass(frozen=True)
class MaximumThicknessParams:
  """Numerical parameters for the dimensionless thickness analysis."""

  diffusion: float = 1.0e-4
  h0: float = 0.1
  ramp_epsilon: float = 1.0e-3
  j_max: float = 0.01
  characteristic_width: float = 0.1
  reference_length: float = 0.09
  divergence_epsilon: float = 1.0e-8
  projection_volume_mode: ProjectionVolumeMode = "reference"

  def __post_init__(self) -> None:
    positive = (
      "diffusion",
      "h0",
      "ramp_epsilon",
      "characteristic_width",
      "reference_length",
      "divergence_epsilon",
    )
    for name in positive:
      if getattr(self, name) <= 0.0:
        raise ValueError(f"{name} must be positive")
    if self.j_max < 0.0:
      raise ValueError("j_max must be non-negative")
    if self.projection_volume_mode not in ("reference", "current"):
      raise ValueError(
        "projection_volume_mode must be 'reference' or 'current'"
      )


def smooth_characteristic(phi: np.ndarray, width: float = 0.1) -> np.ndarray:
  """Return the C2 Heaviside used by ``thickLSTO.ipynb``."""
  if width <= 0.0:
    raise ValueError("width must be positive")
  values = np.asarray(phi, dtype=float)
  result = np.empty_like(values)
  result[values <= -width] = 0.0
  result[values >= width] = 1.0
  middle = np.abs(values) < width
  x = values[middle] / width
  result[middle] = 0.5 + x * (
    15.0 / 16.0 - 5.0 / 8.0 * x**2 + 3.0 / 16.0 * x**4
  )
  return result


def smooth_characteristic_derivative(
  phi: np.ndarray, width: float = 0.1
) -> np.ndarray:
  """Return the derivative of :func:`smooth_characteristic`."""
  if width <= 0.0:
    raise ValueError("width must be positive")
  values = np.asarray(phi, dtype=float)
  result = np.zeros_like(values)
  middle = np.abs(values) < width
  x = values[middle] / width
  result[middle] = (
    15.0 / 16.0 - 15.0 / 8.0 * x**2 + 15.0 / 16.0 * x**4
  ) / width
  return result
