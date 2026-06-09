import unittest

import numpy as np

from moto.src.thickness_constraint import (
  MaximumThicknessParams,
  smooth_characteristic,
  smooth_characteristic_derivative,
)


class MaximumThicknessParamsTest(unittest.TestCase):

  def test_defaults_match_specification(self):
    params = MaximumThicknessParams()

    self.assertEqual(params.h0, 0.1)
    self.assertEqual(params.j_max, 0.01)
    self.assertEqual(params.reference_length, 0.09)
    self.assertEqual(params.characteristic_width, 0.1)
    self.assertEqual(params.projection_volume_mode, "reference")

  def test_invalid_parameters_are_rejected(self):
    with self.assertRaises(ValueError):
      MaximumThicknessParams(h0=0.0)
    with self.assertRaises(ValueError):
      MaximumThicknessParams(j_max=-1.0)
    with self.assertRaises(ValueError):
      MaximumThicknessParams(projection_volume_mode="invalid")


class SmoothCharacteristicTest(unittest.TestCase):

  def test_values_at_transition_points(self):
    values = smooth_characteristic(np.array([-0.2, -0.1, 0.0, 0.1, 0.2]))

    np.testing.assert_allclose(values, [0.0, 0.0, 0.5, 1.0, 1.0])

  def test_derivative_matches_centered_difference(self):
    phi = np.array([-0.08, -0.03, 0.0, 0.04, 0.09])
    step = 1.0e-7
    finite_difference = (
      smooth_characteristic(phi + step)
      - smooth_characteristic(phi - step)
    ) / (2.0 * step)

    np.testing.assert_allclose(
      smooth_characteristic_derivative(phi),
      finite_difference,
      rtol=1.0e-6,
      atol=1.0e-7,
    )


if __name__ == "__main__":
  unittest.main()
