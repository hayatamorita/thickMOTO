import unittest
from types import SimpleNamespace

import numpy as np

from moto.src.thickness_constraint import (
  MaximumThicknessParams,
  precompute_rect4_mesh,
  smooth_characteristic,
  smooth_characteristic_derivative,
)


def structured_mesh(lx=0.18, ly=0.09, nx=2, ny=1):
  x = np.linspace(0.0, lx, nx + 1)
  y = np.linspace(0.0, ly, ny + 1)
  xx, yy = np.meshgrid(x, y)
  nodes = np.column_stack((xx.ravel(), yy.ravel()))
  elements = []
  for row in range(ny):
    for column in range(nx):
      lower_left = row * (nx + 1) + column
      elements.append(
        [
          lower_left,
          lower_left + 1,
          lower_left + nx + 2,
          lower_left + nx + 1,
        ]
      )
  return SimpleNamespace(
    num_dim=2,
    nodes=SimpleNamespace(coords=nodes),
    elem_nodes=np.asarray(elements),
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


class Rect4PrecomputeTest(unittest.TestCase):

  def test_coordinates_and_weights_are_dimensionless(self):
    precompute = precompute_rect4_mesh(structured_mesh())

    np.testing.assert_allclose(np.ptp(precompute.nodes, axis=0), [2.0, 1.0])
    self.assertAlmostEqual(float(np.sum(precompute.nodal_weights)), 2.0)
    self.assertEqual(precompute.shape_gradients.shape, (2, 4, 4, 2))

  def test_non_positive_reference_length_is_rejected(self):
    with self.assertRaises(ValueError):
      precompute_rect4_mesh(structured_mesh(), reference_length=0.0)


if __name__ == "__main__":
  unittest.main()
