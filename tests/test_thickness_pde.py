import unittest

import numpy as np

from moto.src.thickness_constraint import (
  MaximumThicknessParams,
  analyze_maximum_thickness,
  precompute_rect4_mesh,
  smooth_characteristic,
)
from tests.test_thickness_constraint import structured_mesh


class MaximumThicknessPDETest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.precompute = precompute_rect4_mesh(
      structured_mesh(nx=16, ny=8)
    )
    cls.params = MaximumThicknessParams()

  def _horizontal_band(self, half_width):
    y = self.precompute.nodes[:, 1]
    phi = half_width - np.abs(y - 0.5)
    return smooth_characteristic(phi, width=0.05)

  def test_analysis_returns_finite_fields(self):
    result = analyze_maximum_thickness(
      self._horizontal_band(0.15), self.params, self.precompute
    )

    self.assertTrue(np.isfinite(result.constraint))
    self.assertTrue(np.all(np.isfinite(result.states)))
    self.assertTrue(np.all(np.isfinite(result.thickness)))
    self.assertTrue(np.all(np.isfinite(result.evaluation)))

  def test_thick_band_has_larger_constraint_than_thin_band(self):
    thin = analyze_maximum_thickness(
      self._horizontal_band(0.08), self.params, self.precompute
    )
    thick = analyze_maximum_thickness(
      self._horizontal_band(0.25), self.params, self.precompute
    )

    self.assertGreater(thick.constraint, thin.constraint)

  def test_void_constraint_is_minus_j_max(self):
    result = analyze_maximum_thickness(
      np.zeros(self.precompute.num_nodes), self.params, self.precompute
    )

    self.assertAlmostEqual(result.constraint, -self.params.j_max)


if __name__ == "__main__":
  unittest.main()
