"""Build per-material-point element/node stencils and GIMP basis data for grid assembly."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike

import moto.src.material_points as _mp
import moto.src.mpm_basis as _basis


def compute_elems_of_matpts(
  elem_min: jnp.ndarray,
  elem_max: jnp.ndarray,
  mp_coords: jnp.ndarray,
  mp_dom_lengths: jnp.ndarray,
  tol: jnp.ndarray,
  max_elems_per_matpt: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Compute which grid elements overlap each material point support region.

  We represent:
    - each element by an axis-aligned box with corners (elem_min, elem_max),
    - each material point support by a box centered at mp_coords with half-widths mp_dom_lengths.

  An element overlaps a material point support if the two boxes intersect in *all* dimensions.

  Args:
    elem_min: Element min corner coordinates. Shape: (num_elems, num_dim).
    elem_max: Element max corner coordinates. Shape: (num_elems, num_dim).
    mp_coords: Material point coordinates. Shape: (num_pts, num_dim).
    mp_dom_lengths: Material point support half-widths. Shape: (num_pts, num_dim).
    tol: Per-point tolerance used to avoid boundary ambiguity when supports lie on element faces.
      Shape: (num_pts, num_dim).
    max_elems_per_matpt: Maximum overlapping elements to keep per material point.

  Returns:
    matpt_elems: Element ids per material point, padded with -1.
      Shape: (num_pts, max_elems_per_matpt).
    elem_mask: Overlap mask (True if element overlaps point support).
      Shape: (num_pts, num_elems).
  """
  # pt_min/pt_max: (num_pts, num_dim)
  pt_min = mp_coords - mp_dom_lengths + tol
  pt_max = mp_coords + mp_dom_lengths - tol

  # elem_mask: (num_pts, num_elems)
  elem_mask = jnp.all(
    (elem_min[None, :, :] < pt_max[:, None, :])
    & (elem_max[None, :, :] > pt_min[:, None, :]),
    axis=-1,
  )

  # Keep up to max_elems_per_matpt element ids per point (pad with -1).
  flags = elem_mask.astype(jnp.int32)  # (num_pts, num_elems)
  top_vals, top_idx = lax.top_k(
    flags, max_elems_per_matpt
  )  # (num_pts, max_elems_per_matpt)
  matpt_elems = jnp.where(top_vals > 0, top_idx, -1)  # (num_pts, max_elems_per_matpt)

  return matpt_elems, elem_mask


def compute_nodes_of_matpts(
  elem_nodes: jnp.ndarray,
  matpt_elems: jnp.ndarray,
  num_nodes: int,
  max_nodes_per_matpt: int,
) -> jnp.ndarray:
  """Compute the unique grid nodes touched by each material point's element stencil.

  For each material point, we:
    1) gather nodes of its overlapping elements,
    2) take the unique set,
    3) keep up to max_nodes_per_matpt nodes (pad with -1).

  Args:
    elem_nodes: Node ids per element. Shape: (num_elems, nodes_per_elem).
    matpt_elems: Element ids per material point, padded with -1.
      Shape: (num_pts, max_elems_per_matpt).
    num_nodes: Total number of grid nodes.
    max_nodes_per_matpt: Maximum nodes to keep per material point.

  Returns:
    matpt_nodes: Node ids per material point, padded with -1.
      Shape: (num_pts, max_nodes_per_matpt).
  """
  num_elems = elem_nodes.shape[0]
  nodes_per_elem = elem_nodes.shape[1]

  def nodes_for_one(elems_row: jnp.ndarray) -> jnp.ndarray:
    # elems_row: (max_elems_per_matpt,)
    valid_elem = elems_row >= 0
    safe_elems = jnp.clip(elems_row, 0, num_elems - 1)

    local_nodes = elem_nodes[safe_elems]  # (max_elems_per_matpt, nodes_per_elem)
    nodes_flat = local_nodes.reshape(-1)  # (max_elems_per_matpt * nodes_per_elem,)

    # Only count contributions from valid (non-padded) elements.
    per_node_mask = jnp.repeat(valid_elem.astype(jnp.int32), repeats=nodes_per_elem)

    node_counts = jnp.zeros((num_nodes,), dtype=jnp.int32)
    node_counts = node_counts.at[nodes_flat].add(per_node_mask)

    flags = (node_counts > 0).astype(jnp.int32)  # (num_nodes,)
    vals, idx = lax.top_k(flags, max_nodes_per_matpt)  # (max_nodes_per_matpt,)

    return jnp.where(vals > 0, idx, -1)  # (max_nodes_per_matpt,)

  return jax.vmap(nodes_for_one)(matpt_elems)  # (num_pts, max_nodes_per_matpt)


def compute_shape_functions(
  matpt_nodes: jnp.ndarray,
  mesh_node_coords: jnp.ndarray,
  elem_size: jnp.ndarray,
  mp_coord: jnp.ndarray,
  mp_domain_length: jnp.ndarray,
  eps: float = 1e-14,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Compute GIMP basis values and reference gradients for each material point stencil.

  This computes raw GIMP basis values/gradients on each material point stencil and
  then enforces, on the valid (non-padded) stencil entries:

    1) Partition-of-unity:      Σ_m N_{p m} = 1
    2) Zero-sum gradients:      Σ_m ∇_X N_{p m} = 0

  Padded stencils (using -1) and floating point accumulation can otherwise violate
  these identities slightly, which can feed spurious modes into the UL-MPM solve.

  Args:
    matpt_nodes: Node ids per material point, padded with -1 of shape (num_pts,
      mp_max_nodes_per_point).
    mesh_node_coords: Grid node coordinates of shape (num_grid_nodes, num_dim).
    elem_size: Element size vector of shape: (num_dim,).
    mp_coord: Material point coordinates of shape: (num_pts, num_dim).
    mp_domain_length: Material point support half-widths of shape: (num_pts, num_dim).
    eps: Small floor to avoid division by zero in normalization.

  Returns:
    shp_fn: Shape/basis values N at each material point stencil (0 on padded nodes).
      Shape: (num_pts, mp_max_nodes_per_point).
    d_shp_fn: Reference gradients ∇_X N at each stencil (0 on padded nodes).
      Shape: (num_pts, mp_max_nodes_per_point, num_dim).
  """
  # (p) num_pts, (m) mp_max_nodes_per_point, (d) num_dim, (q) keepdims singleton (=1)

  valid = matpt_nodes >= 0  # (p, m)
  has_node = valid[..., None]  # (p, m, 1)

  safe_idx = jnp.where(valid, matpt_nodes, 0)  # (p, m)

  node_coords = mesh_node_coords[safe_idx]  # (p, m, d)
  node_coords = jnp.where(has_node, node_coords, 0.0)  # (p, m, d)

  shp_fn, d_shp_fn = jax.vmap(
    lambda coords, c0, dl: _basis.mpm_basis(coords, elem_size, c0, dl),
    in_axes=(0, 0, 0),
  )(node_coords, mp_coord, mp_domain_length)  # (p, m), (p, m, d)

  shp_fn = jnp.where(valid, shp_fn, 0.0)  # (p, m)
  d_shp_fn = jnp.where(has_node, d_shp_fn, 0.0)  # (p, m, d)

  # Enforce partition-of-unity and zero-sum gradients on the actual stencil
  shp_fn_sum = jnp.sum(shp_fn, axis=1, keepdims=True)  # (p, q)
  d_shp_fn_sum = jnp.sum(d_shp_fn, axis=1, keepdims=True)  # (p, q, d)

  shp_fn_sum = jnp.maximum(shp_fn_sum, eps)  # (p, q)

  shp_fn_norm = shp_fn / shp_fn_sum  # (p, m)
  # Quotient rule: d(N/s) = (s*dN - N*ds) / s^2, where s = Σ_m N_m and ds = Σ_m dN_m
  d_shp_fn_s = jnp.einsum("pmd,pq->pmd", d_shp_fn, shp_fn_sum)  # (p, m, d)

  shp_fn_g = jnp.einsum("pm,pqd->pmd", shp_fn, d_shp_fn_sum)  # (p, m, d)
  d_shp_fn_norm = (d_shp_fn_s - shp_fn_g) / (shp_fn_sum[..., None] ** 2)  # (p, m, d)

  # Keep padded nodes exactly zero for robustness
  shp_fn = jnp.where(valid, shp_fn_norm, 0.0)  # (p, m)
  d_shp_fn = jnp.where(has_node, d_shp_fn_norm, 0.0)  # (p, m, d)

  return shp_fn, d_shp_fn


def update_grid_particle_map(
  mp_coord: ArrayLike,
  mp_domain_length: ArrayLike,
  mesh_node_coords: ArrayLike,
  mesh_elem_nodes: ArrayLike,
  mesh_elem_min: ArrayLike,
  mesh_elem_max: ArrayLike,
  mesh_elem_size: ArrayLike,
  mp_max_elems_per_point: int,
  mp_max_nodes_per_point: int,
  mp_num_pts: int,
) -> tuple[jnp.ndarray, _mp.ParticleGridMap]:
  """Build the particle→grid map used by force and stiffness assembly.

  Steps:
    1) Find overlapping elements per material point (padded with -1).
    2) Convert those elements into a unique node stencil per material point (padded with -1).
    3) Evaluate GIMP basis values and reference gradients on that node stencil (padded with 0).
    4) Build:
       - active_dofs mask (which grid dofs are touched by any material point),
       - per-point grid dof ids and COO indices used for stiffness assembly.

  Args:
    mp_coord: Material point coordinates. Shape: (num_pts, num_dim).
    mp_domain_length: Material point support half-widths. Shape: (num_pts, num_dim).
    mesh_node_coords: Grid node coordinates. Shape: (num_nodes, num_dim).
    mesh_elem_nodes: Node ids per element. Shape: (num_elems, nodes_per_elem).
    mesh_elem_min: Element min corner coordinates. Shape: (num_elems, num_dim).
    mesh_elem_max: Element max corner coordinates. Shape: (num_elems, num_dim).
    mesh_elem_size: Element size vector. Shape: (num_dim,).
    mp_max_elems_per_point: Max overlapping elements per material point.
    mp_max_nodes_per_point: Max unique nodes per material point.
    mp_num_pts: Number of material points (= num_pts).

  Returns:
    active_dofs: Boolean mask of dofs touched by any material point. Shape: (num_grid_dofs,).
    mp_map: ParticleGridMap containing padded stencils, basis data, and stiffness COO indices.
  """
  mesh_node_coords = jnp.asarray(mesh_node_coords)
  num_nodes, num_dim = mesh_node_coords.shape

  tol = (
    jnp.sqrt(jnp.finfo(mp_domain_length.dtype).eps) * mp_domain_length
  )  # (num_pts, num_dim)

  matpt_elems, elem_mask = compute_elems_of_matpts(
    mesh_elem_min,
    mesh_elem_max,
    mp_coord,
    mp_domain_length,
    tol,
    max_elems_per_matpt=mp_max_elems_per_point,
  )  # (num_pts, mp_max_elems_per_point), (num_pts, num_elems)

  matpt_nodes = compute_nodes_of_matpts(
    jnp.asarray(mesh_elem_nodes),
    matpt_elems,
    num_nodes=num_nodes,
    max_nodes_per_matpt=mp_max_nodes_per_point,
  )  # (num_pts, mp_max_nodes_per_point)

  shp_fn, d_shp_fn = compute_shape_functions(
    matpt_nodes,
    mesh_node_coords,
    elem_size=mesh_elem_size,
    mp_coord=mp_coord,
    mp_domain_length=mp_domain_length,
  )  # (num_pts, mp_max_nodes_per_point), (num_pts, mp_max_nodes_per_point, num_dim)

  # Active dofs: nodes belonging to any element overlapped by any material point.
  active_elems = jnp.any(elem_mask, axis=0)  # (num_elems,)

  elem_nodes = jnp.asarray(mesh_elem_nodes)  # (num_elems, nodes_per_elem)
  nodes_per_elem = elem_nodes.shape[1]
  flat_node_ids = elem_nodes.reshape(-1)  # (num_elems * nodes_per_elem,)

  flat_elem_active = jnp.repeat(active_elems, repeats=nodes_per_elem).astype(jnp.int32)
  node_counts = (
    jnp.zeros((num_nodes,), dtype=jnp.int32).at[flat_node_ids].add(flat_elem_active)
  )
  active_nodes = node_counts > 0  # (num_nodes,)

  active_dofs = (
    jnp.repeat(active_nodes[:, None], repeats=num_dim, axis=1)
    .reshape(-1)
    .astype(jnp.bool_)
  )

  grid_dofs_of_mp = _mp.compute_grid_dofs_of_mp(
    matpt_nodes, num_dim
  )  # (num_pts, mp_max_nodes_per_point, num_dim)
  dofs_per_pt = int(mp_max_nodes_per_point * num_dim)
  k_indices = _mp.compute_stiff_indices(grid_dofs_of_mp, mp_num_pts, dofs_per_pt)

  mp_map = _mp.ParticleGridMap(
    grid_nodes_of_mp=matpt_nodes,
    grid_elems_of_mp=matpt_elems,
    grid_dofs_of_mp=grid_dofs_of_mp,
    shp_fn=shp_fn,
    d_shp_fn=d_shp_fn,
    k_indices=k_indices,
  )
  return active_dofs, mp_map
