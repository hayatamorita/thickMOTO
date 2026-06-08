"""Linear and Nonlinear solver for the system of equations integrated into JAX."""

import abc
import enum
import functools
import warnings
from typing import Dict

import jax
import jax.experimental.sparse as jax_sprs
import jax.numpy as jnp
import numpy as np
import pyamg
import scipy.sparse as spy_sprs
import scipy.sparse.linalg as spy_linalg
from jax.typing import ArrayLike

try:
  import petsc4py.PETSc as PETSc
except ImportError:
  PETSc = None
  warnings.warn("petsc4py library not found. PETSc solver is not available.")

try:
  import pypardiso  # type: ignore
except ImportError:
  warnings.warn("pypardiso library not found. Some solvers may not be available.")


class LinearSolvers(enum.Enum):
  """Enumeration of linear solvers."""

  LINALG_SOLVE = enum.auto()
  AMG_CG = enum.auto()
  AMG_BICGSTAB = enum.auto()
  SCIPY_SPARSE = enum.auto()
  PARDISO = enum.auto()
  PETSC = enum.auto()


def _jacobi_preconditioner(A: spy_sprs.coo_matrix) -> spy_sprs.coo_matrix:
  """
  Computes the Jacobi preconditioner for a sparse matrix A in COO format.
  """
  diag_data = A.diagonal()
  # avoid division by zero
  diag_data = np.where(diag_data == 0.0, 1.0, diag_data)
  diag_idxs = np.arange(A.shape[0])
  M_inv = spy_sprs.coo_matrix((1.0 / diag_data, (diag_idxs, diag_idxs)), shape=A.shape)
  return M_inv


def _petsc_solve(
  A: spy_sprs.csr_matrix, b: ArrayLike, solver_options: Dict
) -> np.ndarray:
  """Solve for u = A^{-1}b using PETSc.

  Args:
    A: A sparse CSR matrix of shape (m,m).
    b: The rhs vector of shape (m,).
    solver_options: A dictionary containing PETSc solver options.

  Returns: The solution vector of shape (m,).
  """
  if PETSc is None:
    raise ImportError("petsc4py is required when using LinearSolvers.PETSC.")

  ksp_type = (
    solver_options["petsc_solver"]["ksp_type"]
    if "petsc_solver" in solver_options and "ksp_type" in solver_options["petsc_solver"]
    else "bcgsl"
  )
  pc_type = (
    solver_options["petsc_solver"]["pc_type"]
    if "petsc_solver" in solver_options and "pc_type" in solver_options["petsc_solver"]
    else "ilu"
  )

  A = PETSc.Mat().createAIJ(
    size=A.shape,
    csr=(
      A.indptr.astype(PETSc.IntType, copy=False),
      A.indices.astype(PETSc.IntType, copy=False),
      A.data,
    ),
  )

  rhs = PETSc.Vec().createSeq(len(b))
  rhs.setValues(range(len(b)), np.array(b))
  ksp = PETSc.KSP().create()
  ksp.setOperators(A)
  ksp.setFromOptions()
  ksp.setType(ksp_type)
  ksp.pc.setType(pc_type)

  if ksp_type == "tfqmr":
    ksp.pc.setFactorSolverType("mumps")

  x = PETSc.Vec().createSeq(len(b))
  ksp.solve(rhs, x)

  return x.getArray()


def _bcoo_to_scipy_csr(A: jax_sprs.BCOO) -> spy_sprs.csr_matrix:
  """
  Robustly convert a (possibly padded) JAX BCOO to a SciPy CSR.
  Filters out padded/invalid indices and explicit zeros.
  """
  # Pull to host & NumPy
  data = np.asarray(jax.device_get(jax.lax.stop_gradient(A.data)))
  idx = np.asarray(jax.device_get(A.indices))
  rows, cols = idx[:, 0], idx[:, 1]
  m, n = A.shape

  # Keep only valid, in-bounds, nonzero entries
  mask = (rows >= 0) & (rows < m) & (cols >= 0) & (cols < n) & (data != 0)
  if mask is not None:
    rows = rows[mask]
    cols = cols[mask]
    data = data[mask]

  if rows.size == 0:
    return spy_sprs.csr_matrix((m, n))

  A_coo = spy_sprs.coo_matrix((data, (rows, cols)), shape=(m, n))
  return A_coo.tocsr()


def solve(
  A: jax_sprs.BCOO,
  b: jnp.ndarray,
  params: Dict,
  u0: jnp.ndarray = None,
) -> jnp.ndarray:
  """Solve for u = A^{-1}b using custom solvers.

  This function uses `jax.lax.custom_linear_solve` to solve the linear system of
    equations defined by the sparse matrix A and the right-hand side vector b. For
    the list of available solvers, see `LinearSolvers` enum. The solvers themselves
    are not differentiable. By wrapping them in `jax.lax.custom_linear_solve`, we can
    use them in JAX computations while still allowing for differentiation through the
    linear solve operation.

  Args:
    A: A sparse BCOO matrix of shape (m,m)
    b: The rhs vector of shape (m,).
    params: Additional parameters for the solver. This should include:
      - "solver": The type of solver to use, as defined in `LinearSolvers`
      - other solver-specific parameters, e.g., "rtol" for relative tolerance.
    u0: The initial guess for the solution of shape (m,). Not required for all solvers,
      but some solvers may need it (e.g., iterative solvers like CG or BICGSTAB).

  Returns: The solution vector of shape (m,).
  """

  def solver_wrapper(A, b):
    # Convert to robust SciPy CSR first
    A_sp = _bcoo_to_scipy_csr(A)
    b_np = np.asarray(jax.device_get(jax.lax.stop_gradient(b)))

    solver = params.get("solver", LinearSolvers.SCIPY_SPARSE)

    if solver == LinearSolvers.AMG_CG:
      M = _jacobi_preconditioner(A_sp.tocoo())
      x, _ = pyamg.krylov.cg(A_sp, b_np, tol=params.get("rtol", 1e-10), x0=u0, M=M)

    elif solver == LinearSolvers.AMG_BICGSTAB:
      M = _jacobi_preconditioner(A_sp.tocoo())
      x, _ = pyamg.krylov.bicgstab(
        A_sp, b_np, tol=params.get("rtol", 1e-10), x0=u0, M=M
      )

    elif solver == LinearSolvers.SCIPY_SPARSE:
      x = spy_linalg.spsolve(A_sp, b_np)

    elif solver == LinearSolvers.LINALG_SOLVE:
      # Dense solve fallback
      x = np.linalg.solve(A_sp.toarray(), b_np)

    elif solver == LinearSolvers.PARDISO:
      x = pypardiso.spsolve(A_sp, b_np)

    elif solver == LinearSolvers.PETSC:
      x = _petsc_solve(A_sp, b_np, params)

    else:
      raise ValueError("Invalid solver type")

    return x.astype(b_np.dtype, copy=False).reshape(b_np.shape)

  result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)

  def cust_fwd_solver(mv, b):
    return jax.pure_callback(solver_wrapper, result_shape, A, b)

  def cust_bwd_solver(mv, b):
    # use A.T for adjoint; robust convert inside solver_wrapper as well
    AT = A.T
    return jax.pure_callback(solver_wrapper, result_shape, AT, b)

  # Linear solve inside custom_linear_solve
  def mv(u):
    return A @ u

  sol = jax.lax.custom_linear_solve(
    mv, b, solve=cust_fwd_solver, transpose_solve=cust_bwd_solver
  )
  return sol.reshape(-1)


class NonlinearProblem(abc.ABC):
  """Base class for the nonlinear problems.

  This class defines the interface for nonlinear problems that can be solved using
    Newton's method or modified Newton's method. The derived classes should implement
    the `get_residual_and_tangent_stiffness` method to compute the residual and
    tangent stiffness matrix of the system of equations.
  """

  def __init__(self, solver_settings: dict):
    """Initializes the nonlinear problem with solver settings.

    Args:
      solver_settings: Dictionary containing the solver settings.
    """
    self.solver_settings = solver_settings

  @abc.abstractmethod
  def get_residual_and_tangent_stiffness(
    self, x: jax.Array, *params
  ) -> tuple[jax.Array, jax.Array]:
    """Base class function for computing the residual and tangent stiffness matrix.

    The function takes as arguments (x0, *params) where x0 is the current guess
    of the solution and *params are the additional parameters.

    Returns:
      res: An array of (num_dofs,) containing the residual.
      K: A sparse matrix of size (num_dofs, num_dofs) containing the tangent
        stiffness matrix.
    """
    pass


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def newton_raphson_solve(
  problem: NonlinearProblem,
  x0: jnp.ndarray,
  *params,
) -> jnp.ndarray:
  """Nonlinear solver using Newton's method.

  This function implements the Newton-Raphson method to solve the nonlinear system of
    equations defined by the `problem` object. The method iteratively updates the
    solution until convergence is achieved based on the specified settings in
    `problem.solver_settings["nonlinear"]`.

    We then define a custom JVP rule for this function to enable differentiation
    through the nonlinear solve operation. In particular, we use the implicit function
    theorem to compute the JVP of the solution with respect to the parameters.
    This avoids the need to differentiate through the entire Newton-Raphson iteration
    chain, which can be computationally expensive and memory-intensive.

  Args:
    problem: The function to compute the residual.
    jacobian_fn: The function to compute the jacobian. The function takes the
      same arguments as residual_fn.
    x0: Array of (n,) of the initial guess.
    *params: Additional parameters for the residual and jacobian functions.

  Returns: Array of (n,) of the solution.
  """

  ctr = 0
  res, _ = problem.get_residual_and_tangent_stiffness(x0, *params)
  init_res_norm = jax.lax.stop_gradient(jnp.linalg.norm(res))
  state = (x0, init_res_norm, ctr)

  settings = problem.solver_settings["nonlinear"]

  def cond_fn(state):
    _, res_norm, ctr = state
    cond_iter = ctr < settings["max_iter"]
    cond_res = res_norm > settings["threshold"] * init_res_norm
    return cond_iter & cond_res

  def body_fn(state):
    x0, res_norm, ctr = state
    residual, jacobian = problem.get_residual_and_tangent_stiffness(x0, *params)
    x0 -= solve(jacobian, residual, params=problem.solver_settings["linear"])
    res_norm = jax.lax.stop_gradient(jnp.linalg.norm(residual))
    ctr += 1
    return (x0, res_norm, ctr)

  x0, res_norm, ctr = jax.lax.while_loop(cond_fn, body_fn, state)
  jax.debug.print("NR converged in {x} iters with res_norm = {y}", x=ctr, y=res_norm)
  return x0


@newton_raphson_solve.defjvp
def solve_jvp(
  problem: NonlinearProblem,
  primals: tuple[jnp.ndarray, ...],
  tangents: tuple[jnp.ndarray, ...],
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """JVP rule for the Newton-Raphson solver.

  This function implements the JVP rule for the Newton-Raphson solver using the
    implicit function theorem. Given the primal inputs (x0, *params) and their tangents,
    we first solve the nonlinear system to obtain the solution x. We then compute the
    product of the derivative of the residual with respect to the parameters times
    the perturbation of the parameters. Finally, we solve the linear system to
    obtain the JVP of the solution with respect to the parameters.

  Args:
    problem: The nonlinear problem object.
    primals: A tuple containing the primal inputs (x0, *params).
    tangents: A tuple containing the tangents of the primal inputs (dx0, *dparams).

  Returns: A tuple containing the solution x and the JVP of the solution with respect
    to the parameters.
  """
  x0, *params = primals
  _, *dparams = tangents

  x = newton_raphson_solve(problem, x0, *params)
  _, df_dp, jacobian = jax.jvp(
    problem.get_residual_and_tangent_stiffness,
    (x, *params),
    (jnp.zeros_like(x), *dparams),
    has_aux=True,
  )
  jinvv_dfdp = solve(jacobian, -df_dp, params=problem.solver_settings["linear"])

  return x, jinvv_dfdp


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def modified_newton_raphson_solve(
  problem,
  x0: jnp.ndarray,
  *params,
  eps: float = 1e-30,
) -> jnp.ndarray:
  """Solve a nonlinear system with Newton damping + Armijo backtracking.

    We solve the nonlinear residual equation
        r(x) = 0,
    using a Newton-type iteration. At iteration k we compute a Newton direction by solving
        J(x_k) Δx_k = r(x_k),
    and propose an update
        x_{k+1} = x_k - α_k Δx_k.

    Two robustness mechanisms are used:

    1) Modified-Newton damping predictor:
      A scalar step-length predictor `lam_pred ∈ [lam_min, 1]` is estimated using residual
      norms at a half-step and full-step:
          x_half = x - 0.5 Δx,
          x_full = x - 1.0 Δx,
      and then computing `lam_pred` from these three norms. This provides a cheap, problem-
      dependent initial guess for the step length.

    2) Backtracking (Armijo) line search:
     Starting from `α = lam_pred`, we shrink `α ← α * line_search_shrink` until the Armijo
     sufficient decrease condition is satisfied:
         ||r(x - αΔx)|| <= (1 - c α) ||r(x)||,
     or until α drops below `line_search_alpha_min`. If no acceptable α is found, the
     iteration rejects the step and keeps `x` unchanged for that iteration.

    Nonlinear settings (required) in `problem.solver_settings["nonlinear"]`:

      - max_iter (int, >= 1):Maximum Newton iterations per solve.

      - threshold (float, (0, 1)):
          Relative convergence tolerance on residual norm: stop when
            ||r|| <= threshold * ||r0||.
          Typical values 1e-8 to 1e-12. Smaller is stricter (more iterations).

      - lam_min (float, (0, 1]):
          Lower bound for the predicted step length (lam_pred) before line search.
          Purpose: avoid tiny predictor steps caused by noisy half/full norms.
          Typical values 0.01–0.2. If too small, Newton can stagnate; if too large, the
          line search will do more work (or reject more often).

      - line_search_max_iter (int, >= 0):
          Max backtracking reductions. Typical values 8–20.
          More iterations increases robustness at the cost of extra residual evals.

      - line_search_shrink (float, (0, 1)):
          Backtracking factor: alpha <- alpha * shrink each failed trial.
          Typical values 0.5 (standard), sometimes 0.8 (gentler) or 0.25 (more aggressive).
          If shrink is too close to 1, backtracking may take many reductions.

      - line_search_alpha_min (float, (0, 1]):
          Smallest step length allowed before declaring the Armijo search failed.
          Typical values 1e-6 to 1e-3.
          If too large: you may reject steps that could succeed with more damping.
          If too small: you may waste work taking nearly-zero steps (slow progress).

      - line_search_armijo_c (float, (0, 1)):
          Armijo sufficient-decrease constant in:
            ||r(x - alpha dx)|| <= (1 - c * alpha) ||r(x)||.
          Typical values 1e-4 (common default) to 1e-2.
          Larger c demands more decrease (harder to satisfy), causing more damping.

    Reference:
      Armijo, Larry. 1966. “Minimization of Functions Having Lipschitz Continuous First
      Partial Derivatives.” Pacific Journal of Mathematics 16 (1): 1–3.

    Args:
      problem: Nonlinear problem provides `get_residual_and_tangent_stiffness(x, *params)'
      x0: Initial guess, shape (num_dofs,).
      *params: Extra args forwarded to residual/tangent assembly.
      eps: Small number to avoid division by zero.
  s
    Returns:
      x: Converged (or last) iterate, shape (num_dofs,).
  """
  nl = problem.solver_settings["nonlinear"]

  def safe_norm(r: jnp.ndarray) -> jnp.ndarray:
    n = jnp.linalg.norm(r)
    return jnp.where(jnp.isfinite(n), n, jnp.inf)

  r0, _ = problem.get_residual_and_tangent_stiffness(x0, *params)
  r0_norm = jax.lax.stop_gradient(safe_norm(r0))

  state = (x0, r0_norm, jnp.asarray(0, dtype=jnp.int32))

  def cond_fn(state):
    _, r_norm, it = state
    return (it < nl["max_iter"]) & (r_norm > nl["threshold"] * r0_norm)

  def body_fn(state):
    x, _, it = state

    r, k = problem.get_residual_and_tangent_stiffness(x, *params)
    r_norm = jax.lax.stop_gradient(safe_norm(r))

    dx = solve(k, r, params=problem.solver_settings["linear"])  # J dx = r

    # Newton predictor alpha0 from half/full trial norms
    x_half = x - 0.5 * dx
    r_half, _ = problem.get_residual_and_tangent_stiffness(x_half, *params)
    r_half_norm = jax.lax.stop_gradient(safe_norm(r_half))

    x_full = x - dx
    r_full, _ = problem.get_residual_and_tangent_stiffness(x_full, *params)
    r_full_norm = jax.lax.stop_gradient(safe_norm(r_full))

    numer = 3.0 * r_norm + r_full_norm - 4.0 * r_half_norm
    denom = 4.0 * r_norm + 4.0 * r_full_norm - 8.0 * r_half_norm

    alpha0 = jnp.where(jnp.abs(denom) > eps, numer / denom, 1.0)
    alpha0 = jnp.where(jnp.isfinite(alpha0), alpha0, 1.0)
    alpha0 = jnp.clip(alpha0, nl["lam_min"], 1.0)

    # Armijo: ||r(x - a dx)|| <= (1 - c a) ||r(x)||
    def eval_trial(alpha):
      x_trial = x - alpha * dx
      r_trial, _ = problem.get_residual_and_tangent_stiffness(x_trial, *params)
      r_trial_norm = jax.lax.stop_gradient(safe_norm(r_trial))
      return x_trial, r_trial_norm

    def armijo_satisfied(alpha, r_trial_norm):
      rhs = (1.0 - nl["line_search_armijo_c"] * alpha) * r_norm
      return jnp.isfinite(r_trial_norm) & (r_trial_norm <= rhs)

    x_trial0, r_trial0_norm = eval_trial(alpha0)
    armijo0 = armijo_satisfied(alpha0, r_trial0_norm)

    # backtrack state:
    bt_state = (
      alpha0,
      x_trial0,
      r_trial0_norm,
      armijo0,
      jnp.asarray(0, dtype=jnp.int32),
      x_trial0,
      r_trial0_norm,
    )

    def backt_cond(st):
      alpha, _, _, armijo, bt_it, *_ = st
      return (
        (~armijo)
        & (bt_it < nl["line_search_max_iter"])
        & (alpha > nl["line_search_alpha_min"])
      )

    def backt_body(st):
      alpha, _, _, _, bt_it, best_x, best_norm = st

      alpha = alpha * nl["line_search_shrink"]
      x_trial, r_trial_norm = eval_trial(alpha)
      armijo = armijo_satisfied(alpha, r_trial_norm)

      improved = r_trial_norm < best_norm
      best_x = jnp.where(improved, x_trial, best_x)
      best_norm = jnp.where(improved, r_trial_norm, best_norm)

      return (alpha, x_trial, r_trial_norm, armijo, bt_it + 1, best_x, best_norm)

    _, x_trial, r_trial_norm, armijo, _, best_x, best_norm = jax.lax.while_loop(
      backt_cond, backt_body, bt_state
    )

    # Accept Armijo if satisfied; else accept best decreasing; else reject.
    best_decreases = jnp.isfinite(best_norm) & (best_norm < r_norm)
    use_best = (~armijo) & best_decreases

    x_new = jnp.where(armijo, x_trial, jnp.where(use_best, best_x, x))
    r_new = jnp.where(armijo, r_trial_norm, jnp.where(use_best, best_norm, r_norm))

    return (x_new, r_new, it + 1)

  x, r_norm, it = jax.lax.while_loop(cond_fn, body_fn, state)

  r0_safe = jnp.maximum(r0_norm, eps)
  ratio = jax.lax.stop_gradient(r_norm / r0_safe)
  converged = r_norm <= nl["threshold"] * r0_norm

  def _print_conv(_):
    jax.debug.print(
      "NR converged in {it} iters, res_norm/res_norm_0: {r}", it=it, r=ratio
    )
    return 0

  def _print_stop(_):
    jax.debug.print(
      "NR stopped in {it} iters, res_norm/res_norm_0: {r}", it=it, r=ratio
    )
    return 0

  _ = jax.lax.cond(converged, _print_conv, _print_stop, operand=0)
  return x


@modified_newton_raphson_solve.defjvp
def solve_jvp_mod(  # noqa: F811
  problem: NonlinearProblem,
  primals: tuple[jnp.ndarray, ...],
  tangents: tuple[jnp.ndarray, ...],
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """JVP rule for the modified Newton-Raphson solver.

  This function implements the JVP rule for the modified Newton-Raphson solver using the
    implicit function theorem. Given the primal inputs (x0, *params) and their tangents,
    we first solve the nonlinear system to obtain the solution x. We then compute the
    product of the derivative of the residual with respect to the parameters times
    the perturbation of the parameters. Finally, we solve the linear system to
    obtain the JVP of the solution with respect to the parameters.

  Args:
    problem: The nonlinear problem object.
    primals: A tuple containing the primal inputs (x0, *params).
    tangents: A tuple containing the tangents of the primal inputs (dx0, *dparams).

  Returns: A tuple containing the solution x and the JVP of the solution with respect
    to the parameters.
  """
  x0, *params = primals
  _, *dparams = tangents

  x = modified_newton_raphson_solve(problem, x0, *params)
  _, df_dp, jacobian = jax.jvp(
    problem.get_residual_and_tangent_stiffness,
    (x, *params),
    (jnp.zeros_like(x), *dparams),
    has_aux=True,
  )
  jinvv_dfdp = solve(jacobian, -df_dp, params=problem.solver_settings["linear"])
  return x, jinvv_dfdp
