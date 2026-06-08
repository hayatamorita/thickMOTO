"""Driver for the Updated-Lagrangian Material Point Method (UL-MPM) solver."""

import dataclasses
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

import moto.src.bc as _bc
import moto.src.hyperelastic_mpm as _det_mps_struct
import moto.src.ext_force as _ext_force
import moto.src.material_points as _mp
import moto.src.mesher as _mesh
import moto.src.mpm_elem_map as _mpm_elem_map
import moto.src.nl_solver as _nlsolv


def compute_inactive_grid_dofs(
  dirichlet_dofs: jnp.ndarray,
  active_dof_mask: jnp.ndarray,
  num_dofs: int,
) -> jnp.ndarray:
  """Compute the inactive grid degrees of freedom mask.

  The inactive grid degrees of freedom (dof) are the degrees of freedom that either have
  fixed boundary conditions or are dofs not influenced by the material points
  in the simulation.

  Args:
    dirichlet_dofs: Array of shape (num_dirichlet_dofs,) containing the indices of the
      Dirichlet dofs (fixed boundary conditions).
    active_dof_mask: Boolean array of shape (num_dofs,) which is True for the indices of
      the grid mesh whose dofs are active. These are the dofs that are influenced by the
      material points.
    num_dofs: Total number of degrees of freedom in the grid.

  Returns: A boolean array of shape (num_dofs,) where True indicates inactive dofs.
    These are the dofs that should be eliminated (fixed or not influenced by particles).
  """
  dirichlet_mask = jnp.zeros((num_dofs,), dtype=bool)
  dirichlet_mask = dirichlet_mask.at[dirichlet_dofs].set(True)

  final_active_mask = jnp.logical_and(active_dof_mask, jnp.logical_not(dirichlet_mask))
  inactive_mask = jnp.logical_not(final_active_mask)
  return inactive_mask


def newton_solve(
  mesh: _mesh.GridMesh,
  mp_state: _mp.MaterialPointConfig,
  bc: _bc.BoundaryCondition,
  du_guess: jnp.ndarray,
  load_steps: int,
  gravity: jnp.ndarray,
  mpm_problem: _det_mps_struct.HyperelasticMPM,
  lame_lambda: jnp.ndarray,
  lame_mu: jnp.ndarray,
  mp_point_force: Optional[jnp.ndarray] = None,
  post_cb: Optional[Callable[[Any], Any]] = None,
) -> tuple[_mp.MaterialPointConfig, jnp.ndarray]:
  """UL-MPM load stepping: map MPs→grid, Newton solve on grid, update MPs.

  This driver runs an Updated-Lagrangian MPM loop with GIMP particle domains.
  It solves for the *incremental* grid displacement du at each step
  by balancing the **Total Internal Force** against the **Total External Force**.

  Steps per load increment:
    1) Build the particle→grid map (padded stencil) and identify active grid dofs.
    2) Assemble the full external force vector (gravity + point forces).
    3) Scale the external force by the current `load_factor` (Total Load).
    4) Solve grid equilibrium with Newton–Raphson (Modified/Robust):
         r(du) = f_int(du; state_prev) - f_ext(step) = 0
    5) Update particle state using the converged incremental displacement.

  Args:
    mesh: Background grid mesh object.
    mp_state: Material point state (dataclass) storing particle fields.
    bc: Boundary condition object.
    du_guess: Initial guess for grid displacement increment (num_grid_dofs, ).
    load_steps: Number of load steps (integer ≥ 1).
    gravity: Gravity vector (num_dim,).
    mpm_problem: `HyperelasticMPM` instance providing:
      - residual/tangent assembly on the grid,
      - particle constitutive + kinematic update after convergence.
    lame_lambda: Array of (num_pts,) of the first Lamé parameter λ for each particle.
    lame_mu: Array of (num_pts,) of the second Lamé parameter μ for each particle.
    mp_point_force: Optional per-particle point forces override (num_pts, num_dim).
      If None, uses `mp_state.point_force`.
    post_cb: Optional callback invoked after each load step:
      post_cb(mesh, mp_state, u_grid_np, step_idx_1based).

  Returns:
    mp_state: Final material point state after all load steps.
    u_total: Accumulated grid displacement across load steps (num_grid_dofs, ).
  """
  num_pts = mp_state.coord.shape[0]

  if mp_point_force is None:
    mp_point_force = mp_state.point_force
  else:
    mp_state = dataclasses.replace(mp_state, point_force=mp_point_force)

  def load_step_body(
    step: int, carry_vals: tuple[_mp.MaterialPointConfig, jnp.ndarray]
  ) -> tuple[_mp.MaterialPointConfig, jnp.ndarray]:
    mp_state, u_total = carry_vals
    jax.debug.print("step {step}/{load_steps}", step=step + 1, load_steps=load_steps)

    load_factor = (step + 1.0) / (1.0 * load_steps)

    mp_volume_prev = mp_state.volume

    active_dof_mask, grid_map = _mpm_elem_map.update_grid_particle_map(
      mp_coord=mp_state.coord,
      mp_domain_length=mp_state.domain_length,
      mesh_node_coords=mesh.nodes.coords,
      mesh_elem_nodes=mesh.elem_nodes,
      mesh_elem_min=mesh.elem_coord_min,
      mesh_elem_max=mesh.elem_coord_max,
      mesh_elem_size=mesh.elem_size,
      mp_max_elems_per_point=mp_state.max_elems_per_point,
      mp_max_nodes_per_point=mp_state.max_nodes_per_point,
      mp_num_pts=num_pts,
    )

    f_ext_full = _ext_force.compute_external_force(
      mat_pt_mass=mp_state.mass,
      mat_pt_shp_fn=grid_map.shp_fn,
      mat_pt_grid_dofs=grid_map.grid_dofs_of_mp,
      pcl_force=mp_point_force,
      num_grid_nodes=mesh.num_nodes,
      gravity=gravity,
    ).reshape(-1)

    eliminate_mask = compute_inactive_grid_dofs(
      dirichlet_dofs=bc["fixed_dofs"],
      active_dof_mask=active_dof_mask.reshape(-1),
      num_dofs=mesh.num_dofs,
    )

    mp_state_prev = dataclasses.replace(
      mp_state,
      volume=mp_volume_prev,
    )

    du = _nlsolv.modified_newton_raphson_solve(
      mpm_problem,
      du_guess,
      lame_lambda,
      lame_mu,
      grid_map,
      mp_state_prev,
      load_factor * f_ext_full,
      eliminate_mask,
    )

    mp_state_updated = mpm_problem.get_updated_mp_state(
      lame_lambda=lame_lambda,
      lame_mu=lame_mu,
      grid_displacement=du,
      grid_mapping=grid_map,
      prev_mp_state=mp_state_prev,
    )

    u_total_updated = u_total + du

    if post_cb is not None:
      jax.debug.callback(post_cb, mesh, mp_state_updated, u_total_updated, step + 1)

    return (mp_state_updated, u_total_updated)

  load_step_body = jax.checkpoint(load_step_body)
  jitted_load_step_body = jax.jit(load_step_body)

  u_total_init = jnp.zeros_like(du_guess)
  init_carry = (mp_state, u_total_init)

  final_carry = jax.lax.fori_loop(0, load_steps, jitted_load_step_body, init_carry)
  mp_state, u_total = final_carry

  return mp_state, u_total.reshape(-1, 1)
