"""Material point state + particle→grid map containers.

This module defines two dataclasses used throughout the UL-MPM code:

1) `MaterialPointConfig`
   Stores state for all material points (MPs): coordinates, deformation measures,
   constitutive state, and per-particle loads.

2) `ParticleGridMap`
   Stores the padded particle→grid coupling (GIMP stencil data): which grid nodes
   influence each particle, shape function values/gradients, and precomputed
   indices for sparse tangent assembly.

Notes on dimensions:
  - `num_dim` / `grid_dim` refers to the kinematic grid dimension (2 for current plane-stress runs).
  - `tensor_dim` refers to the constitutive tensor storage dimension (typically 3 for tensor-form storage,
    with plane stress enforced by solving an out-of-plane stretch so that σ_zz = 0).

Padding conventions (fixed sizes for JIT + vmap):
  - `grid_nodes_of_mp`: uses -1 as a never-valid node-id sentinel (valid ids are
    in [0, num_nodes-1]); this supports simple masks via `grid_nodes_of_mp >= 0`.
  - `grid_dofs_of_mp`: uses 0 as a safe in-bounds dummy index for gathers, so padded
    DOFs must be masked explicitly.
  - `shp_fn`, `d_shp_fn`: padded entries are 0.0 so they contribute nothing in
    interpolation, force assembly, and stiffness assembly.
"""

import jax.numpy as jnp
import jax_dataclasses as jdc


@jdc.pytree_dataclass
class MaterialPointConfig:
  """
  Stores data for all material points (MPs) in the simulation.

  Notes:
    - This container separates **grid/kinematic dimension** (`num_dim`) from the
      **constitutive tensor storage dimension** (`tensor_dim`).

    - Kinematic quantities live in `num_dim`:
        coord, displacement, point_force, domain_length(0)
      These are the quantities you interpolate between particles and the grid.

    - Constitutive quantities live in `tensor_dim`:
        def_grad (F), eps_e (εᵉ), elastic_left_cauchy_green (Bᵉ), cauchy_stress (σ)
      These are stored in full tensor form to support robust tensor-based constitutive
      routines (e.g., SPD log/exp for Hencky strain, tensor push-forward/pull-back).

    - Typical plane-stress setup:
        num_dim = 2, tensor_dim = 3.
      The grid solve is 2D (u, v). The constitutive update is performed in 3×3 form,
      where the out-of-plane elastic Hencky strain (and thickness stretch) is chosen so
      that σ_zz = 0 (equivalently τ_zz = 0).

  Attributes:
    coord: Material point coordinates in grid/kinematic space. Shape: (num_pts, num_dim).

    displacement: Accumulated material point displacement in grid/kinematic space.
      This is the *total* displacement carried by the material point (not the grid
      increment for a single Newton iteration). Shape: (num_pts, num_dim).

    volume0: Initial (reference) material point volume V₀.
      Typically set from the initial particle spacing and thickness assumptions.
      Shape: (num_pts,).

    volume: Current material point volume V.
      Typically updated from the deformation gradient determinant:
        V = det(F) * V₀   (when F is the total deformation gradient). Shape: (num_pts,).

    mass: Material point mass m.
      Usually constant over time for standard solid mechanics (no mass sources/sinks).
      Shape: (num_pts,).

    domain_length0: Initial particle domain half-lengths for GIMP (support radii).
      These determine the particle support size used in basis evaluation and mapping.
      Shape: (num_pts, num_dim).

    domain_length: Current particle domain half-lengths for GIMP (support radii).
      Updated each step to reflect stretching/compression of the particle domain
      (often using diagonal stretch measures for rotation-invariant updates).
      Shape: (num_pts, num_dim).

    def_grad: Total deformation gradient F stored in constitutive tensor space.
      - Stored as a `tensor_dim × tensor_dim` tensor per particle.
      - For plane stress: the in-plane (x–y) block carries deformation from grid kinematics;
        the out-of-plane component F_zz is updated to enforce σ_zz = 0.
      Shape: (num_pts, tensor_dim, tensor_dim).

    eps_e: Elastic logarithmic (Hencky) strain tensor εᵉ stored in constitutive tensor space.
      Often computed from the elastic left Cauchy–Green tensor Bᵉ via:
        εᵉ = 0.5 * log(Bᵉ),
      where log(·) denotes the SPD matrix logarithm.
      Shape: (num_pts, tensor_dim, tensor_dim).

    elastic_left_cauchy_green: Elastic left Cauchy–Green tensor Bᵉ stored in constitutive
      tensor space.
      - Used as the primary elastic strain state for Hencky-type updates.
      - Typically updated by push-forward with the incremental deformation.
      Shape: (num_pts, tensor_dim, tensor_dim).

    cauchy_stress: Cauchy stress tensor σ stored in constitutive tensor space.
      - For grid force assembly in `num_dim`, the in-plane block σ[:num_dim, :num_dim]
        is commonly extracted.
      Shape: (num_pts, tensor_dim, tensor_dim).

    point_force: Optional per-particle point forces (added to gravity/body loads).
      These are forces defined at particle locations (e.g., concentrated loads) and
      later mapped to the grid during external force assembly.
      Shape: (num_pts, num_dim).

    pseudo_density: Pseudo-density field for topology optimization.
      Commonly used to scale stiffness/stress or material parameters per particle.
      Shape: (num_pts,).

    num_pts: Number of material points (static metadata).
    num_dim: Grid / kinematic dimension (e.g., 2 for plane-stress grid solves).
    tensor_dim: Constitutive tensor storage dimension.
      Typical plane-stress setup uses `num_dim=2, tensor_dim=3` (3×3 tensor-form storage with σ_zz = 0 enforced
      by an out-of-plane stretch per particle).

    max_nodes_per_point: Padded stencil size per point (static metadata).
    max_elems_per_point: Padded element support size per point (static metadata).
    dofs_per_pt: Degrees of freedom per point stencil (= num_dim * max_nodes_per_point).
      (static metadata).

  Properties:
    mass_density: Mass density ρ computed pointwise as:
        ρ = mass / volume
      Shape: (num_pts,).
  """

  coord: jnp.ndarray
  displacement: jnp.ndarray
  volume0: jnp.ndarray
  volume: jnp.ndarray
  mass: jnp.ndarray
  domain_length0: jnp.ndarray
  domain_length: jnp.ndarray
  def_grad: jnp.ndarray
  eps_e: jnp.ndarray
  elastic_left_cauchy_green: jnp.ndarray
  cauchy_stress: jnp.ndarray
  point_force: jnp.ndarray
  pseudo_density: jnp.ndarray

  num_pts: jdc.Static[int]
  num_dim: jdc.Static[int]
  tensor_dim: jdc.Static[int]
  max_nodes_per_point: jdc.Static[int]
  max_elems_per_point: jdc.Static[int]
  dofs_per_pt: jdc.Static[int]


@jdc.pytree_dataclass
class ParticleGridMap:
  """Stores per-particle padded particle→grid coupling data.

  Attributes:
    grid_nodes_of_mp: Grid node ids influencing each particle (padded with -1)
      (num_pts, max_nodes_per_point).
    grid_elems_of_mp: Grid element ids overlapping each particle (padded with -1)
      (num_pts, max_elems_per_point).
    grid_dofs_of_mp: Grid dof ids per particle per stencil node (padded with 0)
      (num_pts, max_nodes_per_point, num_dim).

    shp_fn: Shape function values N_a(X_p) per particle per stencil node
      (num_pts, max_nodes_per_point).
    d_shp_fn: Shape function gradients ∇_X N_a(X_p) in reference coordinates
      (num_pts, max_nodes_per_point, num_dim).

    k_indices: COO (row, col) pairs for assembling per-particle tangents into a global sparse matrix
      (num_pts * dofs_per_pt * dofs_per_pt, 2).
  """

  grid_nodes_of_mp: jnp.ndarray
  grid_elems_of_mp: jnp.ndarray
  grid_dofs_of_mp: jnp.ndarray
  shp_fn: jnp.ndarray
  d_shp_fn: jnp.ndarray
  k_indices: jnp.ndarray


def compute_grid_dofs_of_mp(grid_nodes_of_mp: jnp.ndarray, num_dim: int) -> jnp.ndarray:
  """Compute grid dof ids for each particle stencil node.

  Dof numbering:
    dof_id(node, dim) = node * num_dim + dim

  Padding behavior:
    - For padded nodes (node == -1), dof ids are set to 0 (safe gather).

  Args:
    grid_nodes_of_mp: Grid node ids influencing each particle (padded with -1)
      (num_pts, max_nodes_per_point).
    num_dim: Spatial dimension.

  Returns:
    grid_dofs_of_mp: Grid dof ids per particle/node/dimension (padded with 0)
      (num_pts, max_nodes_per_point, num_dim).
  """
  d = jnp.arange(num_dim, dtype=grid_nodes_of_mp.dtype)  # (num_dim,)
  dofs = (
    grid_nodes_of_mp[..., None] * num_dim + d
  )  # (num_pts, max_nodes_per_point, num_dim)
  return jnp.where(grid_nodes_of_mp[..., None] >= 0, dofs, 0)


def compute_stiff_indices(
  grid_dofs_of_mp: jnp.ndarray, num_pts: int, dofs_per_pt: int
) -> jnp.ndarray:
  """Compute COO indices for per-particle tangent block assembly.

  For each particle, we flatten its local dof list:
    grid_dofs_of_mp[p].reshape(dofs_per_pt,)

  Then we build all (row, col) pairs of that list (Cartesian product), and stack
  across particles.

  Padding behavior:
    - Padded dofs are 0 (safe), so padded values must be masked elsewhere to
      avoid accumulating stiffness into dof=0.

  Args:
    grid_dofs_of_mp: Grid dof ids per particle/node/dim (padded with 0)
      (num_pts, max_nodes_per_point, num_dim).
    num_pts: Number of material points.
    dofs_per_pt: Degrees of freedom per particle stencil (= num_dim * max_nodes_per_point).

  Returns:
    k_indices: COO index pairs for all particle tangents (num_pts * dofs_per_pt * dofs_per_pt, 2).
  """
  grid_dofs = grid_dofs_of_mp.reshape(num_pts, dofs_per_pt)  # (num_pts, dofs_per_pt)

  krow = jnp.broadcast_to(grid_dofs[:, :, None], (num_pts, dofs_per_pt, dofs_per_pt))
  kcol = jnp.broadcast_to(grid_dofs[:, None, :], (num_pts, dofs_per_pt, dofs_per_pt))

  return jnp.stack((krow.reshape(-1), kcol.reshape(-1)), axis=1)


def initialize_new_material_points(
  num_pts: int,
  num_dim: int,
  max_nodes_per_point: int,
  max_elems_per_point: int,
  tensor_dim: int = 3,
  dtype: jnp.dtype = jnp.float64,
) -> tuple[MaterialPointConfig, ParticleGridMap]:
  """Initialize material point state + an empty padded particle→grid map.

  This creates:
    1) a `MaterialPointConfig` with zero / identity defaults, and
    2) a `ParticleGridMap` filled with padding sentinels:
       - nodes/elements padded with -1
       - dofs padded with 0 (safe gather)
       - shape values/gradients padded with 0.0 (safe sums)

  It also precomputes `k_indices` (COO row/col pairs) for assembling per-particle
  tangents into a global sparse matrix. Because the map is initially empty (all
  nodes are -1), the initial `k_indices` will be all zeros; once you build a real
  map (via your mapping/update routine), you should recompute `grid_dofs_of_mp`
  and `k_indices` consistently for that map.

  Args:
    num_pts: Number of material points.
    num_dim: Grid / kinematic dimension (e.g., 2 for plane-strain grid solves).
    max_nodes_per_point: Padded stencil size per particle.
    max_elems_per_point: Padded element support size per particle.
    tensor_dim: Constitutive tensor storage dimension.
      Typical plane-strain embedding uses `num_dim=2, tensor_dim=3`.
    dtype: Floating dtype for real-valued fields.

  Returns:
    mp_state: Initialized material point state.
    grid_map: Initialized empty padded particle→grid map.
  """

  eye = jnp.eye(tensor_dim, dtype=dtype)
  dofs_per_pt = max_nodes_per_point * num_dim

  mp_state = MaterialPointConfig(
    coord=jnp.zeros((num_pts, num_dim), dtype=dtype),
    displacement=jnp.zeros((num_pts, num_dim), dtype=dtype),
    volume0=jnp.zeros((num_pts,), dtype=dtype),
    volume=jnp.zeros((num_pts,), dtype=dtype),
    mass=jnp.zeros((num_pts,), dtype=dtype),
    domain_length0=jnp.zeros((num_pts, num_dim), dtype=dtype),
    domain_length=jnp.zeros((num_pts, num_dim), dtype=dtype),
    def_grad=jnp.broadcast_to(eye, (num_pts, tensor_dim, tensor_dim)),
    eps_e=jnp.zeros((num_pts, tensor_dim, tensor_dim), dtype=dtype),
    elastic_left_cauchy_green=jnp.broadcast_to(eye, (num_pts, tensor_dim, tensor_dim)),
    cauchy_stress=jnp.zeros((num_pts, tensor_dim, tensor_dim), dtype=dtype),
    point_force=jnp.zeros((num_pts, num_dim), dtype=dtype),
    pseudo_density=jnp.ones((num_pts,), dtype=dtype),
    num_pts=num_pts,
    num_dim=num_dim,
    tensor_dim=tensor_dim,
    max_nodes_per_point=max_nodes_per_point,
    max_elems_per_point=max_elems_per_point,
    dofs_per_pt=dofs_per_pt,
  )

  grid_nodes = -jnp.ones((num_pts, max_nodes_per_point), dtype=jnp.int32)
  grid_elems = -jnp.ones((num_pts, max_elems_per_point), dtype=jnp.int32)

  grid_dofs = compute_grid_dofs_of_mp(grid_nodes, num_dim)
  k_indices = compute_stiff_indices(grid_dofs, num_pts, dofs_per_pt)

  grid_map = ParticleGridMap(
    grid_nodes_of_mp=grid_nodes,
    grid_elems_of_mp=grid_elems,
    grid_dofs_of_mp=grid_dofs,
    shp_fn=jnp.zeros((num_pts, max_nodes_per_point), dtype=dtype),
    d_shp_fn=jnp.zeros((num_pts, max_nodes_per_point, num_dim), dtype=dtype),
    k_indices=k_indices,
  )

  return mp_state, grid_map
