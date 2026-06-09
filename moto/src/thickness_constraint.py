"""PDE-based maximum-thickness constraint for density-based MPM designs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


ProjectionVolumeMode = Literal["reference", "current"]


@dataclass(frozen=True)
class Rect4Precompute:
  """Dimensionless Rect4 quantities shared by all thickness solves."""

  nodes: np.ndarray
  elements: np.ndarray
  shape_values: np.ndarray
  shape_gradients: np.ndarray
  det_j: np.ndarray
  nodal_weights: np.ndarray
  scalar_rows: np.ndarray
  scalar_cols: np.ndarray
  boundary_nodes: np.ndarray

  @property
  def num_nodes(self) -> int:
    return int(self.nodes.shape[0])

  @property
  def num_elements(self) -> int:
    return int(self.elements.shape[0])


@dataclass(frozen=True)
class NormalizedGIMPProjection:
  """Normalized material-point to grid-node projection."""

  node_ids: np.ndarray
  weights: np.ndarray
  active_nodes: np.ndarray
  num_nodes: int

  @classmethod
  def from_stencil(
    cls,
    node_ids: np.ndarray,
    shape_values: np.ndarray,
    particle_volumes: np.ndarray,
    num_nodes: int,
  ) -> "NormalizedGIMPProjection":
    nodes = np.asarray(node_ids, dtype=int)
    shapes = np.asarray(shape_values, dtype=float)
    volumes = np.asarray(particle_volumes, dtype=float).reshape(-1)
    if nodes.shape != shapes.shape:
      raise ValueError("node_ids and shape_values must have the same shape")
    if nodes.shape[0] != volumes.size:
      raise ValueError("particle_volumes must have one value per particle")
    if np.any(volumes < 0.0):
      raise ValueError("particle_volumes must be non-negative")

    valid = nodes >= 0
    if np.any(nodes[valid] >= num_nodes):
      raise ValueError("node_ids contains an out-of-range grid node")
    safe_nodes = np.where(valid, nodes, 0)
    raw_weights = np.where(valid, shapes * volumes[:, None], 0.0)
    denominator = np.zeros(num_nodes)
    np.add.at(denominator, safe_nodes.ravel(), raw_weights.ravel())
    normalized = np.divide(
      raw_weights,
      denominator[safe_nodes],
      out=np.zeros_like(raw_weights),
      where=valid & (denominator[safe_nodes] > 0.0),
    )
    return cls(
      node_ids=nodes,
      weights=normalized,
      active_nodes=denominator > 0.0,
      num_nodes=int(num_nodes),
    )

  def apply(
    self, particle_values: np.ndarray, inactive_value: float = -1.0
  ) -> np.ndarray:
    """Project particle values to nodes, using a fixed inactive-node value."""
    values = np.asarray(particle_values, dtype=float).reshape(-1)
    if values.size != self.node_ids.shape[0]:
      raise ValueError("particle_values must have one value per particle")
    valid = self.node_ids >= 0
    safe_nodes = np.where(valid, self.node_ids, 0)
    result = np.zeros(self.num_nodes)
    np.add.at(
      result,
      safe_nodes.ravel(),
      (self.weights * values[:, None]).ravel(),
    )
    result[~self.active_nodes] = inactive_value
    return result

  def transpose(self, nodal_values: np.ndarray) -> np.ndarray:
    """Apply the exact transpose of the normalized projection."""
    values = np.asarray(nodal_values, dtype=float).reshape(-1)
    if values.size != self.num_nodes:
      raise ValueError("nodal_values must have one value per grid node")
    valid = self.node_ids >= 0
    safe_nodes = np.where(valid, self.node_ids, 0)
    gathered = np.where(valid, values[safe_nodes], 0.0)
    return np.sum(self.weights * gathered, axis=1)


@dataclass(frozen=True)
class MaximumThicknessResult:
  """State and diagnostic fields from one maximum-thickness analysis."""

  characteristic: np.ndarray
  states: np.ndarray
  thickness: np.ndarray
  evaluation: np.ndarray
  constraint: float
  adjoint: np.ndarray
  gradient_characteristic: np.ndarray


@dataclass(frozen=True)
class DesignThicknessResult:
  """Maximum-thickness result and gradient with respect to design density."""

  rho_tilde: np.ndarray
  rho_bar: np.ndarray
  phi_particles: np.ndarray
  phi_nodes: np.ndarray
  analysis: MaximumThicknessResult
  gradient_design: np.ndarray


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


def _rect4_shape_values(xi: float, eta: float) -> np.ndarray:
  return 0.25 * np.array(
    [
      (1.0 - xi) * (1.0 - eta),
      (1.0 + xi) * (1.0 - eta),
      (1.0 + xi) * (1.0 + eta),
      (1.0 - xi) * (1.0 + eta),
    ]
  )


def _rect4_shape_gradients_reference(xi: float, eta: float) -> np.ndarray:
  return 0.25 * np.array(
    [
      [-(1.0 - eta), -(1.0 - xi)],
      [1.0 - eta, -(1.0 + xi)],
      [1.0 + eta, 1.0 + xi],
      [-(1.0 + eta), 1.0 - xi],
    ]
  )


def precompute_rect4_mesh(
  mesh, reference_length: float = 0.09
) -> Rect4Precompute:
  """Precompute a 2D MPM grid in dimensionless coordinates."""
  if reference_length <= 0.0:
    raise ValueError("reference_length must be positive")
  if int(mesh.num_dim) != 2:
    raise ValueError("maximum-thickness analysis currently supports only 2D")

  nodes = np.asarray(mesh.nodes.coords, dtype=float) / reference_length
  elements = np.asarray(mesh.elem_nodes, dtype=int)
  if elements.ndim != 2 or elements.shape[1] != 4:
    raise ValueError("mesh.elem_nodes must have shape (num_elements, 4)")

  gauss = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
  points = [(xi, eta) for xi in gauss for eta in gauss]
  shape_values = np.array(
    [_rect4_shape_values(xi, eta) for xi, eta in points]
  )
  shape_gradients = np.empty((elements.shape[0], 4, 4, 2))
  det_j = np.empty((elements.shape[0], 4))
  nodal_weights = np.zeros(nodes.shape[0])

  for element_index, element in enumerate(elements):
    element_coords = nodes[element]
    for point_index, (xi, eta) in enumerate(points):
      gradients_reference = _rect4_shape_gradients_reference(xi, eta)
      jacobian = gradients_reference.T @ element_coords
      determinant = float(np.linalg.det(jacobian))
      if determinant <= 0.0:
        raise ValueError("Rect4 element has a non-positive Jacobian")
      shape_gradients[element_index, point_index] = (
        gradients_reference @ np.linalg.inv(jacobian)
      )
      det_j[element_index, point_index] = determinant
      nodal_weights[element] += shape_values[point_index] * determinant

  local_rows, local_cols = np.meshgrid(
    np.arange(4), np.arange(4), indexing="ij"
  )
  scalar_rows = elements[:, local_rows.ravel()]
  scalar_cols = elements[:, local_cols.ravel()]
  lower = np.min(nodes, axis=0)
  upper = np.max(nodes, axis=0)
  boundary_mask = np.zeros(nodes.shape[0], dtype=bool)
  for axis in range(2):
    boundary_mask |= np.isclose(nodes[:, axis], lower[axis])
    boundary_mask |= np.isclose(nodes[:, axis], upper[axis])

  return Rect4Precompute(
    nodes=nodes,
    elements=elements,
    shape_values=shape_values,
    shape_gradients=shape_gradients,
    det_j=det_j,
    nodal_weights=nodal_weights,
    scalar_rows=scalar_rows,
    scalar_cols=scalar_cols,
    boundary_nodes=np.flatnonzero(boundary_mask),
  )


def build_gimp_projection(
  mesh,
  reference_state,
  current_state=None,
  mode: ProjectionVolumeMode = "reference",
) -> NormalizedGIMPProjection:
  """Build a normalized GIMP projection for a reference or current state."""
  if mode not in ("reference", "current"):
    raise ValueError("mode must be 'reference' or 'current'")
  if mode == "reference":
    state = reference_state
    coordinates = state.coord
    domain_lengths = state.domain_length0
    volumes = state.volume0
  else:
    if current_state is None:
      raise ValueError("current_state is required when mode='current'")
    state = current_state
    coordinates = state.coord
    domain_lengths = state.domain_length
    volumes = state.volume

  from moto.src import mpm_elem_map

  _, grid_map = mpm_elem_map.update_grid_particle_map(
    mp_coord=coordinates,
    mp_domain_length=domain_lengths,
    mesh_node_coords=mesh.nodes.coords,
    mesh_elem_nodes=mesh.elem_nodes,
    mesh_elem_min=mesh.elem_coord_min,
    mesh_elem_max=mesh.elem_coord_max,
    mesh_elem_size=mesh.elem_size,
    mp_max_elems_per_point=state.max_elems_per_point,
    mp_max_nodes_per_point=state.max_nodes_per_point,
    mp_num_pts=state.num_pts,
  )
  return NormalizedGIMPProjection.from_stencil(
    np.asarray(grid_map.grid_nodes_of_mp),
    np.asarray(grid_map.shp_fn),
    np.asarray(volumes),
    mesh.num_nodes,
  )


def _validate_nodal_field(
  values: np.ndarray, precompute: Rect4Precompute, name: str
) -> np.ndarray:
  field = np.asarray(values, dtype=float).reshape(-1)
  if field.shape != (precompute.num_nodes,):
    raise ValueError(f"{name} must have shape ({precompute.num_nodes},)")
  if not np.all(np.isfinite(field)):
    raise ValueError(f"{name} must contain only finite values")
  return field


def _assemble_state_system(
  characteristic: np.ndarray,
  params: MaximumThicknessParams,
  precompute: Rect4Precompute,
) -> tuple[sparse.csr_matrix, np.ndarray]:
  local_matrices = np.zeros((precompute.num_elements, 4, 4))
  right_hand_sides = np.zeros((2, precompute.num_nodes))
  for element_index, element in enumerate(precompute.elements):
    chi_element = characteristic[element]
    for point_index, shape in enumerate(precompute.shape_values):
      gradients = precompute.shape_gradients[element_index, point_index]
      weight = precompute.det_j[element_index, point_index]
      chi_q = float(shape @ chi_element)
      local_matrices[element_index] += (
        params.diffusion * (gradients @ gradients.T)
        + (1.0 - chi_q) * np.outer(shape, shape)
      ) * weight
      for axis in range(2):
        right_hand_sides[axis, element] += (
          chi_q * gradients[:, axis] * weight
        )
  matrix = sparse.coo_matrix(
    (
      local_matrices.ravel(),
      (precompute.scalar_rows.ravel(), precompute.scalar_cols.ravel()),
    ),
    shape=(precompute.num_nodes, precompute.num_nodes),
  ).tocsr()
  return matrix, right_hand_sides


def _solve_zero_boundary_fields(
  matrix: sparse.csr_matrix,
  right_hand_sides: np.ndarray,
  precompute: Rect4Precompute,
) -> np.ndarray:
  free = np.setdiff1d(
    np.arange(precompute.num_nodes),
    precompute.boundary_nodes,
    assume_unique=True,
  )
  fields = np.zeros((right_hand_sides.shape[0], precompute.num_nodes))
  if free.size == 0:
    return fields
  try:
    solve_free = sparse_linalg.factorized(matrix[free][:, free].tocsc())
    for axis in range(right_hand_sides.shape[0]):
      fields[axis, free] = solve_free(right_hand_sides[axis, free])
  except RuntimeError as error:
    raise np.linalg.LinAlgError("maximum-thickness PDE solve failed") from error
  if not np.all(np.isfinite(fields)):
    raise np.linalg.LinAlgError("maximum-thickness PDE solve was non-finite")
  return fields


def _assemble_adjoint_right_hand_sides(
  characteristic: np.ndarray,
  states: np.ndarray,
  params: MaximumThicknessParams,
  precompute: Rect4Precompute,
) -> np.ndarray:
  right_hand_sides = np.zeros((2, precompute.num_nodes))
  beta = 0.5 * params.h0 * np.sqrt(params.diffusion)
  for element_index, element in enumerate(precompute.elements):
    chi_element = characteristic[element]
    state_element = states[:, element]
    for point_index, shape in enumerate(precompute.shape_values):
      gradients = precompute.shape_gradients[element_index, point_index]
      weight = precompute.det_j[element_index, point_index]
      chi_q = float(shape @ chi_element)
      divergence = sum(
        float(gradients[:, axis] @ state_element[axis]) for axis in range(2)
      )
      q = 1.0 - beta * divergence
      ramp_prime = 0.5 * (
        1.0 + q / np.sqrt(q**2 + params.ramp_epsilon)
      )
      for axis in range(2):
        right_hand_sides[axis, element] += (
          -beta * chi_q * ramp_prime * gradients[:, axis] * weight
        )
  return right_hand_sides


def _constraint_gradient_characteristic(
  characteristic: np.ndarray,
  states: np.ndarray,
  adjoint: np.ndarray,
  evaluation_q: np.ndarray,
  precompute: Rect4Precompute,
) -> np.ndarray:
  gradient = np.zeros(precompute.num_nodes)
  for element_index, element in enumerate(precompute.elements):
    state_element = states[:, element]
    adjoint_element = adjoint[:, element]
    for point_index, shape in enumerate(precompute.shape_values):
      gradients = precompute.shape_gradients[element_index, point_index]
      weight = precompute.det_j[element_index, point_index]
      divergence_adjoint = sum(
        float(gradients[:, axis] @ adjoint_element[axis])
        for axis in range(2)
      )
      state_adjoint = sum(
        float(shape @ state_element[axis])
        * float(shape @ adjoint_element[axis])
        for axis in range(2)
      )
      integrand = (
        evaluation_q[element_index, point_index]
        + divergence_adjoint
        + state_adjoint
      )
      gradient[element] += shape * integrand * weight
  return gradient


def _quadrature_fields(
  states: np.ndarray,
  params: MaximumThicknessParams,
  precompute: Rect4Precompute,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  divergence = np.zeros((precompute.num_elements, 4))
  for element_index, element in enumerate(precompute.elements):
    state_element = states[:, element]
    for point_index in range(4):
      gradients = precompute.shape_gradients[element_index, point_index]
      divergence[element_index, point_index] = sum(
        float(gradients[:, axis] @ state_element[axis]) for axis in range(2)
      )
  beta = 0.5 * params.h0 * np.sqrt(params.diffusion)
  q = 1.0 - beta * divergence
  evaluation = 0.5 * (q + np.sqrt(q**2 + params.ramp_epsilon))
  denominator = np.sqrt(params.diffusion) * np.maximum(
    divergence, params.divergence_epsilon
  )
  thickness = 2.0 / denominator
  return divergence, evaluation, thickness


def _project_quadrature_field(
  values: np.ndarray, precompute: Rect4Precompute
) -> np.ndarray:
  accumulated = np.zeros(precompute.num_nodes)
  for element_index, element in enumerate(precompute.elements):
    for point_index, shape in enumerate(precompute.shape_values):
      weight = precompute.det_j[element_index, point_index]
      accumulated[element] += (
        shape * values[element_index, point_index] * weight
      )
  return np.divide(
    accumulated,
    precompute.nodal_weights,
    out=np.zeros_like(accumulated),
    where=precompute.nodal_weights > 0.0,
  )


def analyze_maximum_thickness(
  characteristic: np.ndarray,
  params: MaximumThicknessParams,
  precompute: Rect4Precompute,
) -> MaximumThicknessResult:
  """Solve the state PDE and evaluate the dimensionless thickness constraint."""
  chi = _validate_nodal_field(characteristic, precompute, "characteristic")
  if np.any((chi < 0.0) | (chi > 1.0)):
    raise ValueError("characteristic values must be in [0, 1]")
  matrix, right_hand_sides = _assemble_state_system(chi, params, precompute)
  states = _solve_zero_boundary_fields(matrix, right_hand_sides, precompute)
  _, evaluation_q, thickness_q = _quadrature_fields(
    states, params, precompute
  )
  adjoint_right_hand_sides = _assemble_adjoint_right_hand_sides(
    chi, states, params, precompute
  )
  adjoint = _solve_zero_boundary_fields(
    matrix, adjoint_right_hand_sides, precompute
  )
  gradient_characteristic = _constraint_gradient_characteristic(
    chi, states, adjoint, evaluation_q, precompute
  )
  constraint_integral = 0.0
  for element_index, element in enumerate(precompute.elements):
    chi_element = chi[element]
    for point_index, shape in enumerate(precompute.shape_values):
      chi_q = float(shape @ chi_element)
      constraint_integral += (
        chi_q
        * evaluation_q[element_index, point_index]
        * precompute.det_j[element_index, point_index]
      )
  return MaximumThicknessResult(
    characteristic=chi,
    states=states,
    thickness=_project_quadrature_field(thickness_q, precompute),
    evaluation=_project_quadrature_field(evaluation_q, precompute),
    constraint=float(constraint_integral - params.j_max),
    adjoint=adjoint,
    gradient_characteristic=gradient_characteristic,
  )


def _apply_operator(operator, values: np.ndarray) -> np.ndarray:
  try:
    return np.asarray(operator @ values, dtype=float).reshape(-1)
  except TypeError:
    import jax.numpy as jnp

    return np.asarray(operator @ jnp.asarray(values), dtype=float).reshape(-1)


def evaluate_design_thickness(
  design: np.ndarray,
  density_filter,
  threshold_beta: float,
  threshold_eta: float,
  projection: NormalizedGIMPProjection,
  params: MaximumThicknessParams,
  precompute: Rect4Precompute,
) -> DesignThicknessResult:
  """Evaluate ``G_thick`` and its exact discrete gradient with respect to ``x``."""
  if threshold_beta <= 0.0:
    raise ValueError("threshold_beta must be positive")
  if not 0.0 < threshold_eta < 1.0:
    raise ValueError("threshold_eta must be in (0, 1)")
  design_values = np.asarray(design, dtype=float).reshape(-1)
  if design_values.size != projection.node_ids.shape[0]:
    raise ValueError("design must have one value per material point")

  rho_tilde = _apply_operator(density_filter, design_values)
  denominator = np.tanh(threshold_beta * threshold_eta) + np.tanh(
    threshold_beta * (1.0 - threshold_eta)
  )
  shifted = threshold_beta * (rho_tilde - threshold_eta)
  rho_bar = (
    np.tanh(threshold_beta * threshold_eta) + np.tanh(shifted)
  ) / denominator
  threshold_derivative = (
    threshold_beta * (1.0 - np.tanh(shifted) ** 2) / denominator
  )

  phi_particles = 2.0 * (rho_bar - 0.5)
  phi_nodes = projection.apply(phi_particles, inactive_value=-1.0)
  characteristic = smooth_characteristic(
    phi_nodes, width=params.characteristic_width
  )
  analysis = analyze_maximum_thickness(
    characteristic, params, precompute
  )

  gradient_phi_nodes = (
    analysis.gradient_characteristic
    * smooth_characteristic_derivative(
      phi_nodes, width=params.characteristic_width
    )
  )
  gradient_phi_particles = projection.transpose(gradient_phi_nodes)
  gradient_rho_tilde = (
    2.0 * gradient_phi_particles * threshold_derivative
  )
  gradient_design = _apply_operator(
    density_filter.T, gradient_rho_tilde
  )
  if not np.all(np.isfinite(gradient_design)):
    raise FloatingPointError("maximum-thickness design gradient is non-finite")

  return DesignThicknessResult(
    rho_tilde=rho_tilde,
    rho_bar=rho_bar,
    phi_particles=phi_particles,
    phi_nodes=phi_nodes,
    analysis=analysis,
    gradient_design=gradient_design,
  )
