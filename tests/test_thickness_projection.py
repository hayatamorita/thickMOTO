import unittest

import numpy as np

from moto.src.thickness_constraint import NormalizedGIMPProjection


class NormalizedGIMPProjectionTest(unittest.TestCase):

  def setUp(self):
    self.projection = NormalizedGIMPProjection.from_stencil(
      node_ids=np.array([[0, 1], [1, 2]]),
      shape_values=np.array([[0.75, 0.25], [0.25, 0.75]]),
      particle_volumes=np.array([2.0, 1.0]),
      num_nodes=4,
    )

  def test_uniform_particle_field_remains_uniform(self):
    projected = self.projection.apply(np.array([0.3, 0.3]))

    np.testing.assert_allclose(projected[:3], 0.3)
    self.assertEqual(projected[3], -1.0)

  def test_transpose_satisfies_inner_product_identity(self):
    particle_values = np.array([0.2, -0.7])
    nodal_values = np.array([0.4, 0.9, -0.1, 3.0])
    projected = self.projection.apply(particle_values, inactive_value=0.0)

    left = float(projected @ nodal_values)
    right = float(particle_values @ self.projection.transpose(nodal_values))

    self.assertAlmostEqual(left, right)

  def test_zero_denominator_node_has_zero_transpose_gradient(self):
    gradient = self.projection.transpose(np.array([0.0, 0.0, 0.0, 100.0]))

    np.testing.assert_allclose(gradient, 0.0)


if __name__ == "__main__":
  unittest.main()
