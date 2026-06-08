"""Basis functions for the MPM method."""

from typing import Tuple

import jax
import jax.numpy as jnp


def mpm_basis(
  grid_coords: jnp.ndarray,
  grid_size: jnp.ndarray,
  mp_coord: jnp.ndarray,
  mp_dom_length: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Compute the MPM basis functions and their gradients.

  Args:
    grid_coords: Array of shape (num_grid_nodes, num_dim) containing the coordinates of
      grid nodes associated with the material point.
    grid_size: Array of shape (num_dim,) containing the size of the grid elements.
    mp_coord: Array of shape (num_dim,) containing coordinates of the material point.
    mp_dom_length: Array of shape (num_dim,) containing the domain lengths of the
      material point.

  Returns: A tuple containing:
    shp_fn: Array of shape (num_grid_nodes,) containing the value of the shape function
      at the grid nodes.
    d_shp_fn: Array of shape (num_grid_nodes, num_dim,) containing the gradient of the
      shape function at the grid nodes.
  """

  shp = _shp_fn_gimp(
    x_particle=mp_coord,
    x_nodes=grid_coords,
    grid_size=grid_size,
    dom_len=mp_dom_length,
  )
  d_shp = _grad_shp_fn_gimp(
    x_particle=mp_coord,
    x_nodes=grid_coords,
    grid_size=grid_size,
    dom_len=mp_dom_length,
  )
  return shp, d_shp


def _shp_fn_gimp(
  x_particle: jnp.ndarray,
  x_nodes: jnp.ndarray,
  grid_size: jnp.ndarray,
  dom_len: jnp.ndarray,
) -> jnp.ndarray:
  """Material point basis functions.

  Args:
    see `mpm_basis` function.

  Returns: Array of shape (num_grid_nodes,) containing the value of the shape function
    at the grid nodes.
  """
  # For each dimension d: N_d(xp_d, xnode_d) -> (num_grid_nodes,)
  shps = jax.vmap(_shp_fn_gimp_1d, in_axes=(0, 1, 0, 0))(
    x_particle, x_nodes, grid_size, dom_len
  )  # (num_dim, num_grid_nodes)

  # Tensor-product basis: N = Π_d N_d
  return jnp.prod(shps, axis=0)  # (num_grid_nodes,)


def _grad_shp_fn_gimp(
  x_particle: jnp.ndarray,
  x_nodes: jnp.ndarray,
  grid_size: jnp.ndarray,
  dom_len: jnp.ndarray,
) -> jnp.ndarray:
  """Gradient of material point basis functions.

  Args:
    see `mpm_basis` function.

  Returns: Array of shape (num_grid_nodes, num_dim) containing the gradient of the shape
    function at the grid nodes.
  """
  return jax.jacfwd(_shp_fn_gimp, argnums=0)(
    x_particle, x_nodes, grid_size, dom_len
  )  # (num_grid_nodes, num_dim)


def _shp_fn_gimp_1d(
  x_particle: jnp.ndarray,
  x_nodes: jnp.ndarray,
  grid_size: jnp.ndarray,
  dom_len: jnp.ndarray,
) -> jnp.ndarray:
  """Compute the MPM basis functions and their gradients.

  The formulation is based on eq (25) in:

    Coombs, William M., and Charles E. Augarde. "AMPLE: a material point learning
    environment." Advances in Engineering Software 139 (2020): 102748.

  Args:
    x_particle: Scalar array of the position of the particle.
    x_nodes: Array of (n,) of the position of the grid nodes.
    grid_size: The size of the grid element.
    dom_len: The length of the particle domain.

  Returns: Array of shape (n,) containing the shape function.
  """
  delta = x_particle - x_nodes
  h, lp = grid_size, dom_len

  cond_1 = jnp.logical_and(-h - lp < delta, delta <= -h + lp)
  cond_2 = jnp.logical_and(-h + lp < delta, delta <= -lp)
  cond_3 = jnp.logical_and(-lp < delta, delta <= lp)
  cond_4 = jnp.logical_and(lp < delta, delta <= h - lp)
  cond_5 = jnp.logical_and(h - lp < delta, delta <= h + lp)

  shp_1 = (h + lp + delta) ** 2 / (4 * h * lp)
  shp_2 = 1 + delta / h
  shp_3 = 1 - ((delta**2 + lp**2) / (2 * h * lp))
  shp_4 = 1 - delta / h
  shp_5 = (h + lp - delta) ** 2 / (4 * h * lp)

  return (
    (1.0 * cond_1 * shp_1)
    + (1.0 * cond_2 * shp_2)
    + (1.0 * cond_3 * shp_3)
    + (1.0 * cond_4 * shp_4)
    + (1.0 * cond_5 * shp_5)
  )
