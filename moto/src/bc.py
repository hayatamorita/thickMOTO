"""Classes of  boundary conditions."""

import abc
import enum
from typing import Callable, List, Optional, Tuple, TypedDict, Union

import jax.experimental.sparse as jax_sprs
import jax.numpy as jnp
import numpy as np

import moto.src.mesher as _mesh
import moto.src.utils as _utils

_Direc = _utils.Direction


def _is_face_on_edges(
  face_coords: np.ndarray, edges: List[np.ndarray], tol: float
) -> bool:
  """Check if the face coordinates are on the edges.

  Args:
    face_coords: Array of shape (n, num_dims) representing the coordinates of points
      on the face.
    edges: A list of arrays of shape (2, num_dims) representing the edge defined by two
      points (start and end).
    tol: Tolerance for checking if the point is on the segment. Usually set it to half
      the mesh size.
  Returns: True if the face coordinates are on any of the edges, False otherwise.
  """
  for edge in edges:
    if np.all(
      [_utils.is_point_on_segment(edge[0], edge[1], c, tol) for c in face_coords]
    ):
      return True
  return False


def identify_faces(
  mesh: _mesh.GridMesh,
  cond_fn: Optional[Callable[[np.ndarray], np.ndarray | bool]] = None,
  edges: List[np.ndarray] | None = None,
  tol: float = 1e-6,
) -> List[Tuple[int, int]]:
  """Identify the boundary faces of the mesh that satisfy a condition.

  Args:
    mesh: Mesh object.
    cond_fn: (Optional)Function that takes the coordinates of the nodes of a face and
      returns a boolean. If None, then edge must be provided.
    edges: (Optional) A list of arrays of shape (2, n) representing the edge to check
      against. If provided, cond_fn is ignored.

  Returns: List of tuples (elem, face) where elem is the element index and face
    is the face index.
  """
  face_nodes = mesh.elem_template.face_connectivity
  face_coords = mesh.elem_node_coords[:, face_nodes]

  elems, faces = np.nonzero(mesh.boundary_faces)
  coords = face_coords[elems, faces]

  if edges is not None:
    cond_fn = lambda c: _is_face_on_edges(c, edges, tol)  # noqa: E731

  keep = [cond_fn(c) for c in coords]
  elems = elems[keep]
  faces = faces[keep]
  return list(zip(elems.tolist(), faces.tolist()))


class BCType(enum.Enum):
  """Boundary condition types."""

  DIRICHLET = enum.auto()
  NEUMANN = enum.auto()


class BoundaryCondition(abc.ABC):
  """Base class for boundary conditions."""

  @abc.abstractmethod
  def __init__(self, type: BCType, name: str = ""):
    self.type = type
    self.name = name

  @abc.abstractmethod
  def process(self, mesh: _mesh.GridMesh):
    """Process the boundary condition to the mesh."""
    pass


class DirichletBC(BoundaryCondition):
  """Dirichlet boundary condition.

  The Dirichlet boundary condition is a type of boundary condition where the value of
  the solution is known at the boundary.

                             u = u_0 on Γ_D
  Where, u is the field variable, u_0 is the known value and Γ_D is the Dirichlet
  boundary.

  Attributes:
    elem_faces: A list of tuples of (element, face) for the boundary condition. The
      element is the element index in the mesh and the face is the face index of a face
      in the element. The face index is the local face index of the element.
    value: A list of tuples of (Enum, value) for the boundary condition. The
      enum consists of the dof number. For instance, in structural, the enum would be
      keys of (U,V,W) with values (0,1,2) respectively and so on. The value also
      consists of an array of (num_faces,) of the values for the boundary condition.
  """

  def __init__(
    self,
    elem_faces: List[tuple[int, int]],
    values: List[Tuple[enum.Enum, np.ndarray]],
    name: str = "",
  ):
    """Initialize Dirichlet boundary condition."""
    super().__init__(BCType.DIRICHLET, name)
    self.elem_faces = elem_faces
    self.values = values

  def process(self, mesh: _mesh.GridMesh) -> tuple[np.ndarray, np.ndarray]:
    """Process the Dirichlet boundary conditions.

    Args:
      mesh: The mesh object.

    Returns: Tuple of Arrays of (fixed_dofs, dirichlet_values) for the boundary
      condition.
    """
    face_conn = mesh.elem_template.face_connectivity
    nodes_per_face = face_conn.shape[1]
    dofs, values = [], []
    for i, (elem, face) in enumerate(self.elem_faces):
      face_nodes = mesh.elem_nodes[elem, face_conn[face]]
      face_dofs = _mesh.get_dofs_of_nodes(mesh, face_nodes)
      for dir, val in self.values:
        new_dofs = face_dofs[:, dir.value]
        new_vals = val[i] * np.ones((nodes_per_face,))
        for nd, nv in zip(new_dofs, new_vals):
          if nd not in dofs:
            dofs.append(nd)
            values.append(nv)

    values = np.array(values).reshape(-1)
    dofs = np.array(dofs).reshape(-1)
    return dofs, values


class NeumannBC(BoundaryCondition):
  """Neumann boundary condition. The Neumann boundary condition is a type of boundary
  condition where the value of the derivative of the solution is known at the
  boundary.

                n . ∇u = g on Γ_N
  Where, n is the outward normal to the boundary, ∇u is the gradient of the field
  variable and g is the known value of the derivative. Γ_N is the Neumann boundary.

  Attributes:
  elem_faces: A list of tuples of (element, face) for the boundary condition. The
    element is the element index in the mesh and the face is the face index of a face
    in the element. The face index is the local face index of the element.
  value: A list of tuples of (Enum, value) for the boundary condition. The
      enum consists of the dof number. For instance, in structural, the enum would be
      keys of (U,V,W) with values (0,1,2) respectively and so on. The value also
      consists of an array of (num_faces,) of the values for the boundary condition.
  """

  def __init__(
    self,
    elem_faces: List[tuple[int, int]],
    values: List[Tuple[_Direc, np.ndarray]],
    name: str = "",
  ):
    """Initialize Neumann boundary condition."""
    super().__init__(BCType.NEUMANN, name)
    self.elem_faces = elem_faces
    self.values = values

  def process(self, mesh: _mesh.GridMesh) -> List[tuple[int, np.ndarray]]:
    """Process the Neumann boundary conditions.

    Args:
      mesh: The mesh object.

    Returns: A list of elem_forces tuple where each tuple is (elem, force_at_elem_dofs).
       The elem is the element index and force_at_elem_dofs is an array of
       (num_dofs_per_elem,) of the forces at the element dofs.
       The size of the list is equal to the number of elements in `elem_faces`.
    """
    face_conn = mesh.elem_template.face_connectivity
    nodes_per_face = face_conn.shape[1]
    forces = []
    for i, (elem, face) in enumerate(self.elem_faces):
      face_dofs_local = _mesh.get_dofs_of_nodes(mesh, face_conn[face])
      face_force = jnp.zeros((mesh.num_dofs_per_elem,))
      for dofno, val in self.values:
        dofs = face_dofs_local[:, dofno.value]
        applied_force = val[i] * np.ones((nodes_per_face,)) / nodes_per_face
        face_force = face_force.at[dofs].set(applied_force)
      forces.append((elem, face_force))
    return forces


class BCDict(TypedDict):
  elem_forces: np.ndarray
  fixed_dofs: np.ndarray
  free_dofs: np.ndarray
  dirichlet_values: np.ndarray


def process_boundary_conditions(
  bcs: List[BoundaryCondition],
  mesh: _mesh.GridMesh,
) -> BCDict:
  """Process the boundary conditions.

  Args:
    bcs: List of boundary conditions.
    mesh: The mesh object.

  Returns: Tuple of (force, dirichlet_dofs, dirichlet_values)
  """
  dirichlet_dofs = []
  dirichlet_values = []
  elem_forces = jnp.zeros((mesh.num_elems, mesh.num_dofs_per_elem))

  for bc in bcs:
    if isinstance(bc, NeumannBC):
      forces = bc.process(mesh)
      for elem, elem_force in forces:
        elem_forces = elem_forces.at[elem].add(elem_force)

    elif isinstance(bc, DirichletBC):
      fixed_dofs, fixed_values = bc.process(mesh)
      dirichlet_dofs.extend(fixed_dofs)
      dirichlet_values.extend(fixed_values)

    else:
      raise ValueError("Unsupported boundary condition type")

  fixed_dofs = np.array(dirichlet_dofs).astype(int).reshape(-1)
  free_dofs = np.setdiff1d(np.arange(mesh.num_dofs), fixed_dofs)

  return BCDict(
    elem_forces=elem_forces,
    fixed_dofs=fixed_dofs,
    free_dofs=free_dofs,
    dirichlet_values=np.array(dirichlet_values).reshape(-1),
  )


def apply_dirichlet_bc(
  jacobian: jax_sprs.BCOO,
  fixed_dofs: jnp.ndarray,
) -> jax_sprs.BCOO:
  """Applies Dirichlet boundary conditions to a BCOO Jacobian matrix.

  This is a convenience wrapper that converts `fixed_dofs` into a boolean mask
  and then calls `eliminate_dofs_from_matrix(...)`.

  Args:
    jacobian: Jacobian matrix in BCOO format.
    fixed_dofs: Integer array (n_fixed,) with DOF indices to constrain.

  Returns:
    Modified Jacobian with couplings to constrained DOFs removed and identity
    diagonal contributions added on constrained DOFs.
  """
  fixed_dofs = jnp.asarray(fixed_dofs, dtype=jnp.int32)

  ndof = jacobian.shape[0]
  dof_mask = jnp.zeros((ndof,), dtype=jnp.bool_).at[fixed_dofs].set(True)

  ii = jnp.arange(ndof, dtype=jnp.int32)
  diag_pairs = jnp.stack([ii, ii], axis=1)

  return eliminate_dofs_from_matrix(jacobian, dof_mask, diag_pairs)


def eliminate_dofs_from_matrix(
  stiffness_mtrx: jax_sprs.BCOO,
  dof_mask: jnp.ndarray,
  diag_pairs: jnp.ndarray,
) -> jax_sprs.BCOO:
  """Eliminate selected DOFs from a BCOO matrix.

  Pure matrix operation:
    - zeros entries K[i,j] if i or j is masked,
    - adds a diagonal contribution `diag(mask)` so masked rows/cols behave like identity
      after duplicate indices are summed.

  Args:
    stiffness_mtrx: Sparse stiffness matrix in BCOO format.
    dof_mask: Boolean array (ndof,) marking DOFs to eliminate in the matrix.
    diag_pairs: Integer array (ndof,2) containing [[0,0], [1,1], ...].

  Returns:
    BCOO matrix with couplings to masked DOFs removed and an identity diagonal
    contribution added on masked DOFs.
  """
  idx = stiffness_mtrx.indices
  kill = dof_mask[idx[:, 0]] | dof_mask[idx[:, 1]]
  data = jnp.where(kill, 0.0, stiffness_mtrx.data)

  data = jnp.concatenate([data, dof_mask.astype(data.dtype)], axis=0)
  idx = jnp.concatenate([idx, diag_pairs], axis=0)
  return jax_sprs.BCOO((data, idx), shape=stiffness_mtrx.shape)


bcLike = Union[BoundaryCondition, DirichletBC, NeumannBC]
