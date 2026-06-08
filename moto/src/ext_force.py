"""Compute the external force on the grid nodes."""

import jax.numpy as jnp


def compute_external_force(
  mat_pt_mass: jnp.ndarray,
  mat_pt_shp_fn: jnp.ndarray,
  mat_pt_grid_dofs: jnp.ndarray,
  pcl_force: jnp.ndarray,
  num_grid_nodes: int,
  gravity: jnp.ndarray,
) -> jnp.ndarray:
  """Compute force on the grid nodes from body and point forces at material points.

  Args:
    mat_pt_mass: Array of shape (num_particles,) containing the masses of the
      material points.
    mat_pt_shp_fn: Array of shape (num_particles, num_grid_nodes_per_mp,) containing the
      shape function values for each material point at the grid nodes associated with it.
    mat_pt_grid_dofs: Array of shape (num_particles, num_grid_dofs_per_mp) with the
      grid degrees of freedom associated with each material point.
    pcl_force: Array of shape (num_particles, num_dim) containing the forces on the
      particles.
    num_grid_nodes: Total number of grid nodes in the grid mesh.
    gravity: Array of shape (num_dim,) containing the gravity vector.

  Returns: Array of shape (num_grid_dofs,) containing the external force vector. Where,
    num_grid_dofs = num_grid_nodes * num_dim. The forces are assumed to be
    (f1_x, f1_y, f2_x, f2_y, ...).
  """
  # num_(p)ts, (m)ax_nodes_per_mp , (d)imensions
  num_dim = gravity.shape[0]
  num_grid_dofs = num_grid_nodes * num_dim

  mg = jnp.einsum("p, d -> pd", mat_pt_mass, gravity)
  pcl_forces = mg + pcl_force  # (p, d)
  contribs = jnp.einsum("pm, pd -> pmd", mat_pt_shp_fn, pcl_forces)  # (p, m, d)

  external_force = jnp.zeros((num_grid_dofs,))
  external_force = external_force.at[mat_pt_grid_dofs.ravel()].add(contribs.ravel())
  return external_force
