import unittest

import numpy as np

from moto.src.thickness_constraint import (
  MaximumThicknessParams,
  NormalizedGIMPProjection,
  analyze_maximum_thickness,
  evaluate_design_thickness,
  precompute_rect4_mesh,
)
from tests.test_thickness_constraint import structured_mesh


class MaximumThicknessGradientTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.precompute = precompute_rect4_mesh(
      structured_mesh(nx=6, ny=4)
    )
    cls.params = MaximumThicknessParams()

  def test_adjoint_gradient_matches_centered_difference(self):
    x = self.precompute.nodes[:, 0]
    y = self.precompute.nodes[:, 1]
    characteristic = 0.5 + 0.15 * np.sin(np.pi * x / 2.0) * np.sin(np.pi * y)
    result = analyze_maximum_thickness(
      characteristic, self.params, self.precompute
    )
    step = 1.0e-6
    sample_nodes = [0, 9, 17, self.precompute.num_nodes - 1]

    for node in sample_nodes:
      direction = np.zeros(self.precompute.num_nodes)
      direction[node] = step
      plus = analyze_maximum_thickness(
        characteristic + direction, self.params, self.precompute
      ).constraint
      minus = analyze_maximum_thickness(
        characteristic - direction, self.params, self.precompute
      ).constraint
      finite_difference = (plus - minus) / (2.0 * step)

      self.assertAlmostEqual(
        result.gradient_characteristic[node],
        finite_difference,
        delta=max(1.0e-7, 2.0e-4 * abs(finite_difference)),
      )

  def test_gradient_is_finite_and_has_one_value_per_node(self):
    result = analyze_maximum_thickness(
      np.full(self.precompute.num_nodes, 0.5),
      self.params,
      self.precompute,
    )

    self.assertEqual(
      result.gradient_characteristic.shape,
      (self.precompute.num_nodes,),
    )
    self.assertTrue(np.all(np.isfinite(result.gradient_characteristic)))


class DesignThicknessGradientTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.precompute = precompute_rect4_mesh(
      structured_mesh(nx=4, ny=3)
    )
    num_nodes = cls.precompute.num_nodes
    cls.projection = NormalizedGIMPProjection.from_stencil(
      node_ids=np.arange(num_nodes)[:, None],
      shape_values=np.ones((num_nodes, 1)),
      particle_volumes=np.ones(num_nodes),
      num_nodes=num_nodes,
    )
    filter_matrix = np.eye(num_nodes) * 0.8
    filter_matrix += np.roll(np.eye(num_nodes), 1, axis=1) * 0.2
    cls.filter_matrix = filter_matrix
    cls.params = MaximumThicknessParams(characteristic_width=0.4)

  def test_end_to_end_gradient_matches_centered_difference(self):
    num_design = self.precompute.num_nodes
    design = 0.5 + 0.01 * np.sin(np.arange(num_design))
    result = evaluate_design_thickness(
      design,
      self.filter_matrix,
      threshold_beta=1.0,
      threshold_eta=0.5,
      projection=self.projection,
      params=self.params,
      precompute=self.precompute,
    )
    step = 1.0e-6

    for index in (0, 7, num_design - 1):
      direction = np.zeros(num_design)
      direction[index] = step
      plus = evaluate_design_thickness(
        design + direction,
        self.filter_matrix,
        1.0,
        0.5,
        self.projection,
        self.params,
        self.precompute,
      ).analysis.constraint
      minus = evaluate_design_thickness(
        design - direction,
        self.filter_matrix,
        1.0,
        0.5,
        self.projection,
        self.params,
        self.precompute,
      ).analysis.constraint
      finite_difference = (plus - minus) / (2.0 * step)

      self.assertAlmostEqual(
        result.gradient_design[index],
        finite_difference,
        delta=max(1.0e-7, 3.0e-4 * abs(finite_difference)),
      )


if __name__ == "__main__":
  unittest.main()
