"""Hyperelastic Updated-Lagrangian Material Point Method (UL-MPM) modules.

This module implements the grid equilibrium and particle state update routines for an
Updated-Lagrangian Material Point Method (MPM) formulation with GIMP (Generalised
Interpolation MPM)-style particle domains. The implementation targets **2D plane-stress**
grid unknowns while retaining **tensor-form (typically 3×3) constitutive storage** for
particle state.

We solve for grid displacements in `num_dim = 2`:
  u = [u(x,y), v(x,y)]ᵀ,

but we store particle kinematics/constitutive quantities in `tensor_dim × tensor_dim`
tensors (typically `tensor_dim = 3`) to:
  (1) enforce plane stress by solving an out-of-plane stretch per particle (σ_zz = 0), and
  (2) reuse tensor-form constitutive routines (SPD matrix log for Hencky strain,
      tensor-form isotropic law). The grid tangent stiffness is obtained via automatic
      differentiation of the particle internal-force kernel.

Core workflow per load step:
  1) Particle→grid mapping (GIMP):
    - Build per-particle padded stencils: `grid_nodes_of_mp`, `grid_dofs_of_mp`.
    - Evaluate GIMP shape functions N_a(X_p) and reference gradients ∇_X N_a(X_p).
    - All per-particle data are padded to a fixed stencil size `Nmax` for JIT-friendly
      vectorization.

  2) Grid Newton solve (equilibrium on the grid):
    - Assemble internal forces from particles and solve for the grid displacement increment:
        r(u) = f_int(u) - f_ext = 0.                                        (Eq. 1)

  3) Particle update (Lagrangian phase):
    - Interpolate converged grid displacements back to particles to update particle
      coordinates and accumulated displacement.
    - Update constitutive state per particle (F, elastic Hencky strain, stress).

Kinematics (in-plane incremental deformation gradient):
  The local nodal displacement field around a particle is approximated as:
      u(X) = Σ_a N_a(X) u_a,

  yielding an in-plane incremental deformation gradient using reference gradients:
      ΔF_{d m} = δ_{d m} + Σ_a u_{a d} (∂N_a/∂X_m),                         (Eq. 2)
  where indices (d, m) span the grid / kinematic dimension `num_dim`.

Plane-stress closure (solve out-of-plane stretch):
  Constitutive quantities are stored in `tensor_dim` (typically 3). We embed the in-plane
  increment into `tensor_dim` and solve an out-of-plane incremental stretch α so that
  the out-of-plane stress vanishes:
      σ_zz = 0   (equivalently τ_zz = 0).                                   (Eq. 3)

  For the 3D Hencky-strain linear elastic law:
      τ = λ tr(ε_e) I + 2 μ ε_e,
  the plane-stress constraint τ_zz = 0 yields a closed-form out-of-plane Hencky strain:
      ε_zz = -(λ/(λ+2μ)) (ε_xx + ε_yy).                                     (Eq. 4)

  Let ε_zz^prev be obtained from the previous elastic left Cauchy–Green state (B_e_prev),
  and ε_zz^new computed from Eq. (4). Then:
      α = exp(ε_zz^new - ε_zz^prev).                                        (Eq. 5)

Total deformation gradient update:
  The total deformation gradient is updated multiplicatively using a block-diagonal
  increment with in-plane ΔF and out-of-plane α:
      F_dd = ΔF · F_dd_prev,    F_zz = α · F_zz_prev.                        (Eq. 6)

Elastic strain measure (Hencky / logarithmic strain):
  We store the elastic left Cauchy–Green tensor B_e in tensor form and compute the
  elastic logarithmic (Hencky) strain as:
      ε_e = 0.5 log(B_e).                                                  (Eq. 7)
  The matrix logarithm log(·) is evaluated for SPD tensors via `spd_log` (custom VJP).

Constitutive law + stress (tensor form):
  Kirchhoff stress:
      τ = λ tr(ε_e) I + 2 μ ε_e,                                            (Eq. 8)
  Cauchy stress:
      σ = τ / J,   J = det(F).                                             (Eq. 9)

Internal force assembly (per particle):
  Using reference gradients and the inverse in-plane incremental deformation gradient:
      f_int,a = w · (∇_X N_a) · (ΔF^{-1}) · σ_dd,                           (Eq. 10)
  where σ_dd = σ[:num_dim, :num_dim] and the weight uses the **full** incremental Jacobian:
      w = V_prev · det(ΔF_emb) = V_prev · det(ΔF) · α.                      (Eq. 11)

Padding conventions:
  Particle→grid maps are padded to a fixed stencil size `Nmax`:
    - invalid nodes:  `grid_nodes_of_mp = -1`
    - invalid dofs:   `grid_dofs_of_mp = 0`  (safe gather)
    - invalid shapes: `shp_fn = 0`, `d_shp_fn = 0` (safe sums)

  All kernels explicitly mask padded entries to ensure:
    - no contribution to forces/tangents,
    - stable JAX vectorization and JIT compilation.

Reference:
  - Coombs, William M., and Charles E. Augarde. "AMPLE: a material point learning
    environment." *Advances in Engineering Software* 139 (2020): 102748.
"""

import dataclasses
import enum

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

import moto.src.bc as _bc
import moto.src.material as _mat
import moto.src.material_points as _mp
import moto.src.mesher as _mesh
import moto.src.nl_solver as _nl
import moto.src.utils as _utils


class StructField(enum.Enum):
  """The displacement fields."""

  U = 0
  V = 1


def _particle_state(
  local_grid_disp: jnp.ndarray,
  grad_shape_fn_ref: jnp.ndarray,
  mp_def_grad_prev: jnp.ndarray,
  mp_elastic_cauchy_prev: jnp.ndarray,
  lam: jnp.ndarray,
  mu: jnp.ndarray,
  eps: float = 1e-14,
) -> tuple[
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
]:
  """Update particle kinematics and constitutive state for one material point.

  This kernel is used in two places:
    - `_particle_force`: during grid internal-force / tangent assembly, and
    - `HyperelasticMPM.get_updated_mp_state`: after the grid Newton solve converges.

  The grid solve lives in `num_dim` (=2 for current plane-stress runs), while constitutive
  quantities are stored in `tensor_dim × tensor_dim` tensors (typically 3×3). Plane stress
  is enforced by choosing an out-of-plane elastic Hencky strain so that the out-of-plane
  Kirchhoff stress vanishes:

      τ_zz = 0.                                                            (Eq. 1)

  Kinematics (in-plane incremental deformation gradient):
    The local displacement field around the particle is approximated as:
        u(X) = Σ_n N_n(X) u_n.

    Using reference gradients, the in-plane incremental deformation gradient is:
        ΔF_{d m} = δ_{d m} + Σ_n u_{n d} (∂N_n/∂X_m),                      (Eq. 2)

    Its inverse is stored with indices (m, d), matching the force-assembly contraction:
        (ΔF^{-1})_{m d} = (ΔF_{d m})^{-1}.                                (Eq. 3)

  Elastic strain measure (Hencky from elastic left Cauchy–Green):
    The particle stores the elastic left Cauchy–Green tensor from the previous converged
    configuration:
        B_e_prev.                                                         (Eq. 4)

    We push-forward only the in-plane block:
        B_e,dd^new = ΔF · B_e,dd^prev · ΔFᵀ,                               (Eq. 5)

    and compute in-plane Hencky strain:
        ε_e,dd^new = 0.5 log(B_e,dd^new).                                  (Eq. 6)

  Plane-stress closure (Hencky-linear isotropic law):
    Kirchhoff stress in tensor form:
        τ = λ tr(ε_e) I + 2 μ ε_e.                                         (Eq. 7)

    Enforcing τ_zz = 0 yields:
        ε_zz^new = -k_ps (ε_xx^new + ε_yy^new),                            (Eq. 8)
    with
        k_ps = λ/(λ + 2μ).                                                 (Eq. 9)

    Let ε_zz^prev = 0.5 log(B_e,zz^prev). Define the incremental thickness stretch:
        α = exp(ε_zz^new - ε_zz^prev).                                     (Eq. 10)

  Total deformation gradient update (block-diagonal increment):
        F_dd^new = ΔF · F_dd^prev,     F_zz^new = α · F_zz^prev.           (Eq. 11)

  Full incremental Jacobian for UL-MPM weighting:
        det(ΔF_emb) = det(ΔF) · α.                                         (Eq. 12)

  Args:
    local_grid_disp: Flattened local nodal displacement vector
      (u1, v1, u2, v2, ...), ordered consistently with the local dof list used
      to gather grid unknowns for this particle.
      Shape: (num_nodes * num_dim,).
    grad_shape_fn_ref: Shape-function gradients in reference coordinates at the
      particle location:
          ∂N_n / ∂X_m
      where n indexes local grid nodes and m indexes reference directions.
      Shape: (num_nodes, num_dim).
    mp_def_grad_prev: Previous converged total deformation gradient F_prev stored
      in constitutive tensor space (typically 3×3). For plane-stress runs, the
      in-plane block F_prev[:2,:2] evolves from grid kinematics, while the out-of-plane
      stretch F_prev[2,2] tracks the thickness update.
      Shape: (tensor_dim, tensor_dim).
    mp_elastic_cauchy_prev: Previous converged elastic left Cauchy–Green tensor
      B_e_prev (SPD). This is the internal elastic state used for the Hencky strain
      update. For plane stress, only the in-plane push-forward is performed, and the
      out-of-plane component is updated via the plane-stress closure.
      Shape: (tensor_dim, tensor_dim).
    lam: A scalar of the first Lamé parameter λ.
    mu: A scalar of the second (shear modulus) μ.
    eps: Small positive floor used to clamp B_e,zz^prev before taking log, to avoid
      log(0) / numerical issues when the stored out-of-plane elastic stretch becomes
      extremely small.
      Scalar.

  Returns:
    def_grad: Updated total deformation gradient F (tensor_dim, tensor_dim).
    elastic_log_strain: Updated elastic Hencky strain ε_e (tensor_dim, tensor_dim).
    cauchy_stress: Updated Cauchy stress σ (tensor_dim, tensor_dim), with σ_zz = 0.
    incr_def_grad_dm: In-plane incremental deformation gradient ΔF (num_dim, num_dim).
    inv_incr_def_grad_md: In-plane inverse (ΔF)^{-1} stored as (m,d) (num_dim, num_dim).
    stress_dd: In-plane Cauchy stress block σ[:num_dim,:num_dim].
    elastic_left_cauchy_green: Updated elastic left Cauchy–Green B_e
      of shape (tensor_dim, tensor_dim).
    incr_jac_full: Full incremental Jacobian det(ΔF_emb) = det(ΔF) * α.
  """
  # (n)odes_per_particle, (d)(m)=grid kinematics, (k)(l)=constitutive tensor indices

  num_nodes, num_dim = grad_shape_fn_ref.shape
  tensor_dim = mp_def_grad_prev.shape[0]
  oop = num_dim  # out-of-plane index (z)

  nodal_disp = local_grid_disp.reshape(num_nodes, num_dim)  # (n,d)
  iden_d = jnp.eye(num_dim, dtype=nodal_disp.dtype)

  # ΔF_{d m} = δ_{d m} + Σ_n u_{n d} (∂N_n/∂X_m)
  incr_def_grad_dm = iden_d + jnp.einsum("nd, nm -> dm", nodal_disp, grad_shape_fn_ref)

  inv_incr_def_grad_md = jnp.linalg.inv(incr_def_grad_dm)

  elastic_left_cauchy_green_prev = 0.5 * (
    mp_elastic_cauchy_prev + mp_elastic_cauchy_prev.T
  )

  # In-plane push-forward: B_dd^new = ΔF · B_dd^prev · ΔFᵀ
  elastic_left_cauchy_green_prev_dd = elastic_left_cauchy_green_prev[:num_dim, :num_dim]
  elastic_left_cauchy_green_dd = jnp.einsum(
    "dm, mn, en -> de",
    incr_def_grad_dm,
    elastic_left_cauchy_green_prev_dd,
    incr_def_grad_dm,
  )
  elastic_left_cauchy_green_dd = 0.5 * (
    elastic_left_cauchy_green_dd + elastic_left_cauchy_green_dd.T
  )

  elastic_log_strain_dd = 0.5 * _utils.spd_log(elastic_left_cauchy_green_dd)
  trace_elastic_log_strain_in = jnp.trace(elastic_log_strain_dd)

  # Plane-stress closure: ε_zz = -k_ps tr_in,  k_ps = λ/(λ+2μ)
  plane_stress_kappa = lam / (lam + 2.0 * mu)
  elastic_log_strain_zz_new = -plane_stress_kappa * trace_elastic_log_strain_in

  # Previous ε_zz from stored B_e,zz^prev
  elastic_left_cauchy_green_zz_prev = jnp.maximum(
    elastic_left_cauchy_green_prev[oop, oop], eps
  )
  elastic_log_strain_zz_prev = 0.5 * jnp.log(elastic_left_cauchy_green_zz_prev)

  # Incremental thickness stretch α = exp(ε_zz^new - ε_zz^prev)
  alpha = jnp.exp(elastic_log_strain_zz_new - elastic_log_strain_zz_prev)

  # Total deformation gradient update (block-diagonal)
  def_grad_prev_dd = mp_def_grad_prev[:num_dim, :num_dim]
  def_grad_prev_zz = mp_def_grad_prev[oop, oop]

  def_grad_dd = jnp.einsum("dm, mn -> dn", incr_def_grad_dm, def_grad_prev_dd)
  def_grad_zz = alpha * def_grad_prev_zz

  def_grad = jnp.eye(tensor_dim, dtype=nodal_disp.dtype)
  def_grad = def_grad.at[:num_dim, :num_dim].set(def_grad_dd)
  def_grad = def_grad.at[oop, oop].set(def_grad_zz)

  # Rebuild B_e (tensor_dim×tensor_dim) for storage
  elastic_left_cauchy_green = jnp.eye(
    tensor_dim, dtype=elastic_left_cauchy_green_prev.dtype
  )
  elastic_left_cauchy_green = elastic_left_cauchy_green.at[:num_dim, :num_dim].set(
    elastic_left_cauchy_green_dd
  )
  elastic_left_cauchy_green = elastic_left_cauchy_green.at[oop, oop].set(
    elastic_left_cauchy_green_zz_prev * (alpha * alpha)
  )

  # Rebuild ε_e (tensor_dim×tensor_dim) for storage
  elastic_log_strain = jnp.zeros(
    (tensor_dim, tensor_dim), dtype=elastic_left_cauchy_green.dtype
  )
  elastic_log_strain = elastic_log_strain.at[:num_dim, :num_dim].set(
    elastic_log_strain_dd
  )
  elastic_log_strain = elastic_log_strain.at[oop, oop].set(elastic_log_strain_zz_new)

  kirchhoff_stress = _mat.compute_hencky_kirchhoff_stress(
    elastic_log_strain=elastic_log_strain,
    lame_lambda=lam,
    lame_mu=mu,
  )

  jac_def_grad = jnp.linalg.det(def_grad)
  cauchy_stress = kirchhoff_stress / jac_def_grad

  stress_dd = cauchy_stress[:num_dim, :num_dim]

  # det(ΔF_emb) = det(ΔF) * α
  incr_jac_full = jnp.linalg.det(incr_def_grad_dm) * alpha

  return (
    def_grad,
    elastic_log_strain,
    cauchy_stress,
    incr_def_grad_dm,
    inv_incr_def_grad_md,
    stress_dd,
    elastic_left_cauchy_green,
    incr_jac_full,
  )


def _particle_force(
  grid_disp: jnp.ndarray,
  grad_shape_fn_ref: jnp.ndarray,
  mp_volume_prev: jnp.ndarray,
  mp_def_grad_prev3: jnp.ndarray,
  mp_elastic_cauchy_prev3: jnp.ndarray,
  lam: jnp.ndarray,
  mu: jnp.ndarray,
) -> jnp.ndarray:
  """Compute one material point's internal force contribution on the grid.

  All particle kinematics + constitutive quantities (ΔF, F, ε_e, σ) are computed
  in `_particle_state(...)`. This function only performs the nodal force mapping:

    Weight:
      w = V_prev * det(ΔF_emb) = V_prev * det(ΔF) * α

    Force assembly (per node n, force component i):
      f_{n i} = w * (∂N_n/∂X_m) * (ΔF^{-1})_{m d} * σ_{d i}.            (Eq. 1)

  Args:
    grid_disp: Flattened nodal displacement vector (u1, v1, u2, v2, ...),
      shape (num_dim * num_nodes,).
    grad_shape_fn_ref: Shape gradients ∂N/∂X in reference coordinates,
      shape (num_nodes, num_dim).
    mp_volume_prev: Particle volume at the previous converged configuration for this
      load step.
    mp_def_grad_prev3: Previous converged total deformation gradient F_prev stored in
      tensor-form constitutive space (typically 3×3). For plane stress, the in-plane
      block F_prev[:2,:2] evolves from grid kinematics, while F_prev[2,2] tracks the
      thickness stretch chosen to enforce σ_zz = 0.
      Shape: (tensor_dim, tensor_dim).
    mp_elastic_cauchy_prev3: Previously converged elastic left Cauchy–Green tensor
      B_e_prev (SPD), stored in tensor-form constitutive space.
      Shape: (tensor_dim, tensor_dim).
    lam: First Lamé parameter λ for isotropic elasticity.
    mu: Shear modulus μ for isotropic elasticity.

  Returns:
    local_internal_force: Flattened nodal internal force contribution,
      shape (num_dim * num_nodes,).
  """
  # (n)odes_per_particle, (d)(i)(m)
  (_, _, _, _, inv_incr_def_grad, stress_dd, _, incr_jac_full) = _particle_state(
    grid_disp,
    grad_shape_fn_ref,
    mp_def_grad_prev3,
    mp_elastic_cauchy_prev3,
    lam,
    mu,
  )

  # Plane stress: w = V_prev * det(ΔF_emb) = V_prev * det(ΔF) * α
  weight = mp_volume_prev * incr_jac_full

  return weight * jnp.einsum(
    "nm, md, di -> ni", grad_shape_fn_ref, inv_incr_def_grad, stress_dd
  ).reshape(-1)


_particle_force_jac = jax.jacrev(_particle_force, argnums=0)


class HyperelasticMPM(_nl.NonlinearProblem):
  """Hyperelastic MPM residual and tangent assembly (plane stress).

  We solve grid unknowns in 2D (u, v) but store particle constitutive state in 3×3
  tensor form. Plane stress is enforced at the particle level by solving an out-of-plane
  Hencky strain (equivalently thickness stretch) so that σ_zz = 0 (τ_zz = 0).
  The grid tangent stiffness is assembled via AD of the particle internal-force kernel.
  """

  def __init__(self, solver_settings: dict, mesh: _mesh.Mesh) -> None:
    """Initialize the MPM nonlinear problem.

    Args:
      solver_settings: Nonlinear solver settings passed to `NonlinearProblem`.
      mesh: Grid mesh object. Must provide `mesh.nodes.coords` and `mesh.num_dofs`.
    """
    super().__init__(solver_settings)
    self.mesh = mesh

    dof_ids = jnp.arange(self.mesh.num_dofs, dtype=jnp.int32)
    self._diag = jnp.stack([dof_ids, dof_ids], axis=1)

  def get_residual_and_tangent_stiffness(
    self,
    grid_disp: jnp.ndarray,
    lame_lambda: jnp.ndarray,
    lame_mu: jnp.ndarray,
    grid_map: _mp.ParticleGridMap,
    mp_state_prev: _mp.MaterialPointConfig,
    grid_force_ext: jnp.ndarray,
    eliminate_mask: jnp.ndarray,
  ) -> tuple[jnp.ndarray, BCOO]:
    """Assemble global residual and tangent stiffness for the grid solve.

    This function builds:
      - the internal force vector f_int by summing per-particle contributions, and
      - the tangent stiffness matrix K = ∂f_int/∂u using AD (particle-level jacobians).

    The residual is defined as:
        r(u) = f_int(u) - f_ext.                                            (Eq. 1)

    Notes on padding:
      `grid_map` stores per-particle node/dof lists padded to `max_nodes_per_point`.
      Padded entries have `grid_nodes_of_mp = -1`. We mask these to ensure they do not
      contribute to forces or stiffness or accidentally scatter into dof=0.

    Args:
      grid_disp: Global grid displacement increment vector u, shape (num_dofs,). Within
        the Updated-Lagrangian framework, this represents the nodal displacement field
        relative to the configuration at the onset of the current load step.
      lame_lambda: An array of (num_particles,) of the first Lamé parameter λ.
      lame_mu: An array of (num_particles,) of the second Lamé parameter μ.
      grid_map: Particle→grid coupling dataclass containing padded arrays:
        - grid_nodes_of_mp (num_pts, max_nodes_per_point) padded with -1,
        - grid_dofs_of_mp (num_pts, max_nodes_per_point, num_dim) padded with 0,
        - d_shp_fn (num_pts, max_nodes_per_point, num_dim) padded with 0,
        - k_indices (num_pts * (num_dim*max_nodes_per_point)^2, 2) COO indices for
          tangent assembly.
      mp_state_prev: Particle state at the previous converged configuration containing:
        - def_grad(num_pts, 3, 3),
        - elastic_left_cauchy_green (num_pts, 3, 3),
        - volume (num_pts,) previous-converged volume for this load step.
      grid_force_ext: Global external force vector f_ext, shape (num_dofs,).
      eliminate_mask: Boolean mask of dofs to eliminate (Dirichlet + inactive), shape
        (num_dofs,).

    Returns:
      residual: Global residual vector r(u), shape (num_dofs,).
      tangent: Sparse global tangent stiffness matrix K, shape (num_dofs, num_dofs).
    """
    # (p)articles, (n)odes_per_particle(max), (d)(i)(m)=2 for grid
    grid_dofs_of_mp = grid_map.grid_dofs_of_mp  # (num_pts, max_nodes_per_point, 2)
    grad_shape_fn = grid_map.d_shp_fn  # (num_pts, max_nodes_per_point, 2)
    has_node = grid_map.grid_nodes_of_mp >= 0  # (num_pts, max_nodes_per_point)

    # dof_mask: (p, 2*max_nodes_per_point) matching the flattened local dof layout.
    dof_mask = jnp.repeat(has_node, repeats=self.mesh.num_dim, axis=1)

    # Replace padded dofs with 0 so gather is always valid, then mask.
    grid_dofs_flat = jnp.where(has_node[..., None], grid_dofs_of_mp, 0).reshape(
      grid_dofs_of_mp.shape[0], -1
    )  # (p, 2*max_nodes_per_point)

    grid_disp_local = (
      grid_disp[grid_dofs_flat] * dof_mask
    )  # (num_pts, 2*max_nodes_per_point)

    # Ensure padded gradient entries contribute nothing.
    grad_shape_fn = (
      grad_shape_fn * has_node[..., None]
    )  # (num_pts, max_nodes_per_point, 2)

    res_args = (
      grid_disp_local,
      grad_shape_fn,
      mp_state_prev.volume,
      mp_state_prev.def_grad,
      mp_state_prev.elastic_left_cauchy_green,
      lame_lambda,
      lame_mu,
    )

    # Per-particle internal forces and particle-level tangents.
    force_local = jax.vmap(_particle_force)(*res_args)

    tangent_local = jax.vmap(_particle_force_jac)(*res_args)

    # Scatter-add local forces into the global internal force vector.
    internal_force = (
      jnp.zeros((self.mesh.num_dofs,), force_local.dtype)
      .at[grid_dofs_flat]
      .add(force_local * dof_mask)
    )

    # Residual r = f_int - f_ext, then enforce elimination mask.
    residual = jnp.where(
      eliminate_mask, 0.0, internal_force - jnp.asarray(grid_force_ext)
    )

    # Assemble tangent stiffness
    data = (tangent_local * dof_mask[:, :, None] * dof_mask[:, None, :]).reshape(-1)
    idx = grid_map.k_indices.reshape(-1, 2).astype(jnp.int32)

    tangent = BCOO((data, idx), shape=(self.mesh.num_dofs, self.mesh.num_dofs))
    tangent = _bc.eliminate_dofs_from_matrix(tangent, eliminate_mask, self._diag)

    return residual, tangent

  def get_updated_mp_state(
    self,
    lame_lambda: jnp.ndarray,
    lame_mu: jnp.ndarray,
    grid_displacement: jnp.ndarray,
    grid_mapping: _mp.ParticleGridMap,
    prev_mp_state: _mp.MaterialPointConfig,
  ) -> _mp.MaterialPointConfig:
    """Update material point state after a converged grid solve (Lagrangian phase).

    This routine performs the *particle update* step of the USL/UL-MPM algorithm.
    After the grid Newton solve converges for the current load step, it maps the
    converged grid displacement field back to the material points to update particle
    kinematics (position, displacement, domain) and constitutive state (F, Bᵉ, εᵉ, σ).

    The update proceeds in five stages:

    1. **Kinematic Interpolation (GIMP)**:
      Interpolate grid nodal displacements to particles using GIMP shape functions:
        u_p = Σ_a N_a(X_p) u_a.

    2. **Coordinate + Displacement Update**:
      Advect particles and update accumulated particle displacement:
        x_p ← x_p + u_p,
        u_p,total ← u_p,total + u_p.

    3. **Constitutive Update**:
      Update per-particle constitutive quantities via `_particle_state`:
        - total deformation gradient F,
        - elastic left Cauchy–Green tensor Bᵉ,
        - elastic Hencky strain εᵉ,
        - Cauchy stress σ.

    4. Domain Length Update (rotation-invariant stretch):
      Update the particle domain half-lengths for GIMP using a frame-indifferent stretch
      measure based on the polar decomposition of the in-plane deformation gradient block.
      Let F_dd = F[:num_dim, :num_dim], and define:
        C = F_ddᵀ F_dd,  U = sqrt(C).
      Then update domain half-lengths using the diagonal stretch only (no shear coupling):
        l_{p,i}^{n+1} = U_{ii} · l_{p,i}^0.
      This avoids artificial domain distortion under large rigid-body rotations.

    5. Volume Update:
      Update particle volume using the total deformation gradient determinant:
        V = det(F) · V0.

    Args:
      lame_lambda: An array of (num_particles,) of the first Lamé parameter λ.
      lame_mu: An array of (num_particles,) of the second Lamé parameter μ.
      grid_displacement: Array of (num_dofs,)  of the converged global grid displacement
        increment vector for the load step. Within the Updated-Lagrangian framework,
        this is the nodal displacement field relative to the configuration at the onset
        of the current load step.
      grid_mapping: Particle→grid coupling dataclass containing padded arrays:
        - grid_dofs_of_mp (num_pts, max_nodes_per_point, num_dim) padded with 0,
        - grid_nodes_of_mp (num_pts, max_nodes_per_point) padded with -1,
        - shp_fn (num_pts, max_nodes_per_point) padded with 0,
        - d_shp_fn (num_pts, max_nodes_per_point, num_dim) padded with 0.
      prev_mp_state: Previous converged particle state containing:
        - coord, displacement, domain_length(0),
        - def_grad, eps_e, elastic_left_cauchy_green, cauchy_stress,
        - volume0, mass, pseudo_density.

    Returns:
      MaterialPointConfig: Updated material point state containing:
        - coord, displacement,
        - volume, domain_length,
        - def_grad, eps_e, cauchy_stress, elastic_left_cauchy_green.
    """

    num_dim = self.mesh.num_dim

    # Create a mask for valid nodes (padding entries are -1)
    valid_node_mask = grid_mapping.grid_nodes_of_mp >= 0

    # Replace invalid indices with 0 to prevent Out-Of-Bounds access during gather.
    grid_dof_indices = jnp.where(
      valid_node_mask[..., None], grid_mapping.grid_dofs_of_mp, 0
    )
    grid_dof_indices_flat = grid_dof_indices.reshape(grid_dof_indices.shape[0], -1)

    # Create a value mask to zero out the gathered data from the dummy '0' index
    dof_value_mask = jnp.repeat(valid_node_mask, repeats=num_dim, axis=1)

    # Gather local displacements: shape (num_pts, max_nodes * num_dim)
    local_grid_displacement = grid_displacement[grid_dof_indices_flat] * dof_value_mask

    # Mask shape functions and gradients
    shape_functions = grid_mapping.shp_fn * valid_node_mask
    shape_function_gradients = grid_mapping.d_shp_fn * valid_node_mask[..., None]

    nodal_displacement = local_grid_displacement.reshape(
      local_grid_displacement.shape[0], grid_mapping.shp_fn.shape[1], num_dim
    )

    # u_p = sum(N_i * u_i)
    particle_displacement = jnp.einsum(
      "pn,pnd->pd", shape_functions, nodal_displacement
    )

    # x_p = x_p + u_p
    coord_new = prev_mp_state.coord + particle_displacement

    (
      def_grad_new,
      elastic_strain_new,
      stress_new,
      _,
      _,
      _,
      elastic_left_cauchy_green,
      _,
    ) = jax.vmap(_particle_state)(
      local_grid_displacement,
      shape_function_gradients,
      prev_mp_state.def_grad,
      prev_mp_state.elastic_left_cauchy_green,
      lame_lambda,
      lame_mu,
    )

    deformation_gradient_dd = def_grad_new[:, :num_dim, :num_dim]

    # Compute Right Cauchy-Green Deformation Tensor: C = F.T @ F
    cauchy_green_deformation = jnp.einsum(
      "pji, pjk -> pik", deformation_gradient_dd, deformation_gradient_dd
    )

    # Compute Matrix Square Root to get Stretch Tensor U: U = sqrt(C)
    eigenvalues, eigenvectors = jnp.linalg.eigh(cauchy_green_deformation)

    # Clip eigenvalues to avoid numerical issues (sqrt of negative)
    eigenvalues = jnp.maximum(eigenvalues, 1e-14)

    # Reconstruct U = V * sqrt(D) * V.T
    stretch_tensor = jnp.einsum(
      "pik, pk, plk -> pil", eigenvectors, jnp.sqrt(eigenvalues), eigenvectors
    )

    diag_stretch = jnp.diagonal(stretch_tensor, axis1=1, axis2=2)
    diag_stretch = jnp.maximum(diag_stretch, 1e-14)  # Safety floor

    domain_length_new = prev_mp_state.domain_length0 * diag_stretch

    # Update volume: V = J * V0
    volume_new = jnp.linalg.det(def_grad_new) * prev_mp_state.volume0

    return dataclasses.replace(
      prev_mp_state,
      coord=coord_new,
      displacement=prev_mp_state.displacement + particle_displacement,
      volume=volume_new,
      domain_length=domain_length_new,
      def_grad=def_grad_new,
      eps_e=elastic_strain_new,
      cauchy_stress=stress_new,
      elastic_left_cauchy_green=elastic_left_cauchy_green,
    )
