"""Utility functions."""

import enum
from typing import Tuple

import chex
import jax
import jax.experimental.sparse as jax_sprs
import jax.numpy as jnp
import numpy as np
import scipy.special as spy_spl
from jax.typing import ArrayLike


class Direction(enum.Enum):
  """Euclidean Directions."""

  X = 0
  Y = 1
  Z = 2


@chex.dataclass
class Extent:
  """Extent of a variable."""

  min: ArrayLike
  max: ArrayLike

  @property
  def range(self) -> ArrayLike:
    return self.max - self.min

  @property
  def center(self) -> ArrayLike:
    return 0.5 * (self.min + self.max)

  def normalize_array(self, x: ArrayLike) -> ArrayLike:
    """Linearly normalize `x` using `extent` ranges."""
    return (x - self.min) / self.range

  def renormalize_array(self, x: ArrayLike) -> ArrayLike:
    """Recover array from linearly normalized `x` using `extent` ranges."""
    return x * self.range + self.min


def safe_power(x: jax.Array, exp: float) -> jax.Array:
  """Compute the power `x**exp` with a safe check for negative/zero values.

  This function ensures that the input `x` is positive before applying the power
    operation. If `x` is negative or zero, it returns zero. This ensures that the
  power operation does not result in undefined behavior or complex numbers.

  Args:
    x: Input array.
    exp: Exponent value.

  Returns: The result of `x**exp` if `x` is positive, otherwise zero.
  """
  z = jnp.where(x <= 0.0, 1.0, x)
  return jnp.where(x > 0.0, jnp.power(z, exp), 0.0)


def safe_log(x: jax.Array) -> jax.Array:
  """Compute the natural logarithm of `x` with a safe check for non-positive values.

  Args:
    x: Input array.
  Returns: The natural logarithm of `x` if `x` is positive, otherwise zero.
  """
  z = jnp.where(x <= 0.0, 1.0, x)
  return jnp.where(x > 0.0, jnp.log(z), 0.0)


def safe_sqrt(x: jax.Array) -> jax.Array:
  """Compute the square root of x with a safe check for negative values.

  This function ensures that the input `x` is non-negative before applying the
    square root operation. If `x` is negative, it returns zero. This ensures that
    the square root operation does not result in undefined behavior.

  Args:
    x: Input array.

  Returns: The square root of `x` if `x` is non-negative, otherwise zero.
  """
  z = jnp.where(x <= 0.0, 1.0, x)
  return jnp.where(x >= 0.0, jnp.sqrt(z), 0.0)


def safe_divide(x: jax.Array, y: jax.Array, eps: float = 1.0e-6) -> jax.Array:
  """Compute the division of x by y with a safe check for division by zero.

  This function ensures that the denominator `y` is non-zero before applying the
    division operation. If `y` is zero, it returns zero. This ensures that the
    division operation does not result in undefined behavior.

  Args:
    x: Numerator array.
    y: Denominator array.
    eps: Small value below which the absolute value of the denominator is
      treated as zero.

  Returns: The result of `x / y` if `y` is non-zero, otherwise zero.
  """
  z = jnp.where(jnp.abs(y) < eps, 1.0, y)
  return jnp.where(jnp.abs(y) < eps, 0.0, x / z)


def safe_pnorm(x: jax.Array, p: float, axis: int):
  """Compute the p-norm of x with a safe check for negative values.

  This function ensures that the input `x` is non-negative before applying the
    p-norm operation. If `x` is negative, it returns zero. This ensures that the
  p-norm operation does not result in undefined behavior.

  The p-norm is defined as:
               ||x||_p = (sum_i(|x_i|^p))^(1/p)

  The function is often used to compute a smooth approximation to the maximum (or minimum)
  of a set of values.

  Args:
    x: The input array.
    p: The Exponent value. The larger the value, the closer the p-norm is to the maximum.
      However, the problem becomes more nonlinear. A typical value is 6.0.
    axis: The axis along which the p-norm is computed.

  Returns: The p-norm of `x` computed in a safe manner along the specified axis.
  """
  sum_x = jnp.sum(safe_power(x, p), axis=axis)
  return safe_power(sum_x, 1.0 / p)


def inverse_sigmoid(y: jax.Array) -> jax.Array:
  """The inverse of the sigmoid function.

  The sigmoid function f:x->y is defined as:

           f(x) = 1 / (1 + exp(-x))

  The inverse sigmoid function g: y->x is defined as:

           g(y) = ln(y / (1 - y))

  For details see https://tinyurl.com/y7mr76hm
  """
  return jnp.log(y / (1.0 - y))


def gauss_integ_points_weights(
  order: int,
  dimension: int,
) -> tuple[jax.Array, jax.Array]:
  """
  Returns the Gauss integration points and weights for the given order and
    dimension. The number of gauss points is order^dimension.

  Args:
    order (int): The order of the Gauss quadrature.
    dimension (int): The dimension of the integration. Must be in range (1, 3).

  Returns: A tuple containing the integration points and weights.
    - points (numpy.ndarray): An array of shape (order^dimension, dimension)
       containing the integration points.
    - weights (numpy.ndarray): An array of shape (order^dimension,) containing the
        integration weights.

  Raises:
      ValueError: If dimension is not in (1, 3).
  """
  # Get 1D Gauss points and weights
  x, w = spy_spl.roots_legendre(order)

  if dimension == 1:
    points = x.reshape(-1, 1)
    weights = w

  elif dimension == 2:
    # Generate 2D points and weights as tensor products of 1D points and weights
    points = jnp.array([[x_i, x_j] for x_i in x for x_j in x])
    weights = jnp.array([w_i * w_j for w_i in w for w_j in w])

  elif dimension == 3:
    # Generate 3D points and weights as tensor products of 1D points and weights
    points = jnp.array([[x_i, x_j, x_k] for x_i in x for x_j in x for x_k in x])
    weights = jnp.array([w_i * w_j * w_k for w_i in w for w_j in w for w_k in w])

  else:
    raise ValueError("Dimension must be in (1, 3)")

  return points, weights


def threshold_filter(density: jax.Array, beta: float, eta: float = 0.5) -> jax.Array:
  """Threshold project the density, pushing the values towards 0/1.

  Args:
    density: Array of size (num_elems,) that are in [0,1] that contain the
      density of the elements.
    beta: Sharpness of projection (typically ~ 1-32). Larger value indicates
      a sharper projection.
    eta: Center value about which the values are projected.

  Returns: The thresholded density value array of size (num_elems,).
  """
  v1 = jnp.tanh(eta * beta)
  nm = v1 + jnp.tanh(beta * (density - eta))
  dnm = v1 + jnp.tanh(beta * (1.0 - eta))
  return nm / dnm


def _cdist(a: jax.Array, b: jax.Array) -> jax.Array:
  """Computes pairwise Euclidean distances between rows of a and b.

  Args:
    a: Array of shape (N, D) - N points, D dimensions.
    b: Array of shape (M, D) - M points, D dimensions.

  Returns:
    Array of shape (N, M) containing distances.
  """
  diff = a[:, jnp.newaxis, :] - b[jnp.newaxis, :, :]
  return jnp.linalg.norm(diff, axis=-1)


class Filters(enum.Enum):
  """Enum for different types of filters."""

  LINEAR = enum.auto()
  CIRCULAR = enum.auto()
  GAUSSIAN = enum.auto()


def create_density_filter(
  coords: jax.Array,
  cutoff_distance: float,
  filter_type: Filters = Filters.LINEAR,
  eps: float = 1e-12,
) -> jax_sprs.BCOO:
  """Creates a density filter to smoothen out the field.

  The density filter is to ensure that the obtained density fields do not have
  checkerboard patterns. This is common in density-based topology optimization problems.

  Args:
    coords: An array of shape (num_pts, num_dim) of the coordinates of the points.
    cutoff_distance: A float, the radius beyond which the filter has zero influence.
    filter_type: A string, one of 'linear', 'circular', or 'gaussian'.
    eps: A float, small value to avoid division by zero added to the entries.

  Returns: A BCOO sparse matrix of size (num_pts, num_pts) of the filter.
  """
  num_pts = coords.shape[0]

  distances = _cdist(coords, coords)

  row_indices, col_indices = jnp.where(distances <= cutoff_distance)
  relevant_distances = distances[row_indices, col_indices]

  if filter_type == Filters.LINEAR:
    filter_values = 1.0 - (relevant_distances / cutoff_distance)

  elif filter_type == Filters.CIRCULAR:
    filter_values = jnp.sqrt(1.0 - (relevant_distances / cutoff_distance) ** 2)

  elif filter_type == Filters.GAUSSIAN:
    sigma = cutoff_distance / 3.0
    filter_values = jnp.exp(-0.5 * (relevant_distances / sigma) ** 2)

  else:
    raise ValueError(f"Unsupported filter type: {filter_type.name}")

  row_sums = jnp.zeros(num_pts).at[row_indices].add(filter_values) + eps
  inv_row_sums = 1.0 / row_sums
  normalized_filter_values = filter_values * inv_row_sums[row_indices]

  return jax_sprs.BCOO(
    (normalized_filter_values, jnp.stack([row_indices, col_indices], axis=1)),
    shape=(num_pts, num_pts),
  )


def is_point_on_segment(
  start_pt: np.ndarray, end_pt: np.ndarray, pt: np.ndarray, tolerance: float = 1e-9
) -> bool:
  """Checks if a point lies on a line segment with a given tolerance.

  A point is on the segment if it is both collinear with the segment's
  endpoints and lies within the axis-aligned bounding box of the segment.

  Args:
    start_pt: Array of shape (n,) of the start point of the line segment.
    end_pt: Array of shape (n,) of the end point of the line segment.
    pt: Array of shape (n,) of the point to check.
    tolerance: A small value to account for floating-point inaccuracies.

  Returns: True if the point is on the line segment, False otherwise.
  """
  # Boundedness Check
  in_bounds = np.all(pt >= np.minimum(start_pt, end_pt) - tolerance) and np.all(
    pt <= np.maximum(start_pt, end_pt) + tolerance
  )
  if not in_bounds:
    return False

  # Collinearity Check
  cross_product = np.cross(end_pt - start_pt, pt - start_pt)
  return np.abs(cross_product) < tolerance


@jax.custom_vjp
def spd_log(matrix: jnp.ndarray) -> jnp.ndarray:
  """Matrix logarithm for a symmetric positive definite (SPD) tensor.

  For an SPD matrix a ∈ ℝ^{d×d}, let its eigen-decomposition be
    a = q diag(λ) qᵀ,                                                     (Eq. 1)

  where:
    - q ∈ ℝ^{d×d} is an orthonormal matrix whose columns are eigenvectors,
    - λ ∈ ℝ^{d} collects eigenvalues λ_i > 0,
    - diag(λ) is the diagonal matrix with entries λ_i,

  The matrix logarithm is defined by spectral mapping:
    log(a) = q diag(log(λ)) qᵀ.                                           (Eq. 2)

  Reference (spectral definition of matrix functions):
    Higham, Nicholas J. Functions of Matrices: Theory and Computation.
    Philadelphia: Society for Industrial and Applied Mathematics, 2008. Section 11.1.

  Args:
    matrix: A SPD tensor of shape (..., d, d) that is to be ... .

  Returns:
    log_matrix: Matrix logarithm, shape (..., d, d).
  """
  matrix = 0.5 * (matrix + matrix.transpose(-1, -2))

  eigvals, eigvecs = jnp.linalg.eigh(matrix)
  eigvals = jnp.maximum(eigvals, 1e-16)

  log_matrix = jnp.einsum("...ik,...k,...jk->...ij", eigvecs, jnp.log(eigvals), eigvecs)
  return 0.5 * (log_matrix + log_matrix.transpose(-1, -2))


def _spd_log_fwd(
  matrix: jnp.ndarray,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
  """Forward rule for the custom VJP of `spd_log`.

  In JAX `custom_vjp`, the forward rule must return:
      (primal_output, residuals)

  We cache (q, λ) from the eigen-decomposition so the backward rule can reuse
  them without recomputing an eigen-solve.

  Args:
    matrix: SPD tensor, shape (..., d, d).

  Returns:
    log_matrix: Matrix logarithm, shape (..., d, d).
    residuals: Tuple (eigvecs, eigvals) cached for the backward rule, shapes
      (..., d, d) and (..., d).
  """
  matrix = 0.5 * (matrix + matrix.transpose(-1, -2))

  eigvals, eigvecs = jnp.linalg.eigh(matrix)
  eigvals = jnp.maximum(eigvals, 1e-16)

  log_matrix = jnp.einsum("...ik,...k,...jk->...ij", eigvecs, jnp.log(eigvals), eigvecs)
  log_matrix = 0.5 * (log_matrix + log_matrix.transpose(-1, -2))
  return log_matrix, (eigvecs, eigvals)


def _spd_log_bwd(
  residuals: Tuple[jnp.ndarray, jnp.ndarray],
  grad_out: jnp.ndarray,
) -> Tuple[jnp.ndarray]:
  """Backward rule (VJP) via the Fréchet derivative of the matrix logarithm.

  Let a = q diag(λ) qᵀ be the eigen-decomposition from (Eq. 1). Define the
  Loewner (divided-difference) matrix for f(x)=log(x):
    l_ij = (log(λ_i) - log(λ_j)) / (λ_i - λ_j),    i ≠ j
    l_ii = 1 / λ_i.                                                        (Eq. 3)

  Using the Daleckii–Kreĭn formula, the Fréchet derivative satisfies:
    d log(a)[e] = q ( l ⊙ (qᵀ e q) ) qᵀ,                                    (Eq. 4)

  where:
    - e ∈ ℝ^{d×d} is the perturbation direction,
    - ⊙ denotes elementwise (Hadamard) product.

  The VJP applies the same kernel with the upstream gradient `grad_out` in
  place of e.

  Reference (Fréchet derivative / divided differences for spectral matrix functions):
    Higham, Nicholas J. *Functions of Matrices: Theory and Computation*.
    Philadelphia: SIAM, 2008. Chapter 3 (Fréchet derivative; Loewner matrix).

  Args:
    residuals: Cached eigen-decomposition (q, λ), shapes (..., d, d) and (..., d).
    grad_out: Upstream gradient w.r.t. log(a), shape (..., d, d).

  Returns:
    grad_in: Gradient w.r.t. input `matrix`, shape (..., d, d).
  """
  eigvecs, eigvals = residuals

  grad_out = 0.5 * (grad_out + grad_out.transpose(-1, -2))
  grad_spec = jnp.einsum("...mi,...mn,...nj->...ij", eigvecs, grad_out, eigvecs)

  lam_i = eigvals[..., :, None]
  lam_j = eigvals[..., None, :]
  diff = lam_i - lam_j

  loewner = (jnp.log(lam_i) - jnp.log(lam_j)) / diff
  loewner = jnp.where(jnp.abs(diff) < 1e-12, 1.0 / lam_i, loewner)

  grad_in = jnp.einsum(
    "...ik,...kl,...jl->...ij", eigvecs, loewner * grad_spec, eigvecs
  )
  return (0.5 * (grad_in + grad_in.transpose(-1, -2)),)


spd_log.defvjp(_spd_log_fwd, _spd_log_bwd)


def compute_hydrostatic_stress(cauchy_stress: jnp.ndarray) -> jnp.ndarray:
  """Computes the hydrostatic pressure from the Cauchy stress tensor.

    The hydrostatic pressure is given by:
      
      p = 1/num_dim * trace(cauchy_stress)

  Args:
    cauchy_stress: The Cauchy stress tensor of (num_dim, num_dim) in the matrix form.

  Returns: The hydrostatic stress (scalar).
  """
  num_dim = cauchy_stress.shape[0]
  return (1.0 / num_dim) * jnp.trace(cauchy_stress)


def compute_deviatoric_stress(cauchy_stress: jnp.ndarray) -> jnp.ndarray:
  """Computes the deviatoric stress from the Cauchy stress tensor.

      The deviatoric stress is given by:
          sigma_dev = cauchy_stress - hydrostatic_stress * I

  Args:
    cauchy_stress: The Cauchy stress tensor of (num_dim, num_dim) in the matrix form.

  Returns: The deviatoric stress of shape (num_dim, num_dim).
  """
  identity = jnp.eye(cauchy_stress.shape[0])
  return cauchy_stress - compute_hydrostatic_stress(cauchy_stress) * identity


def compute_von_mises_stress(cauchy_stress: jnp.ndarray) -> jnp.ndarray:
  """Computes the von Mises stress from the Cauchy stress tensor.

      The von Mises stress is given by:
          sigma_vm = sqrt(3/2 * trace(sigma_dev^2))

  Args:
    cauchy_stress: The Cauchy stress tensor of (num_dim, num_dim) in the matrix form.

  Returns: The von Mises stress (scalar).
  """
  deviatoric = compute_deviatoric_stress(cauchy_stress)
  return jnp.sqrt(1.5 * jnp.einsum("ij, ji -> ", deviatoric, deviatoric))
