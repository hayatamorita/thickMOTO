"""Material properties for structural analysis."""

from typing import Optional

import chex
import jax.numpy as jnp
from jax.typing import ArrayLike


@chex.dataclass
class StructuralMaterial:
  """Linear structural material constants.

  Attributes:
    youngs_modulus: The young's modulus of the material [Pa].
    poissons_ratio: The poisson's ratio of the material [-].
    mass_density: Mass density of material in [kg/m^3].
    yield_strength: Yield strength of the material [Pa].
  """

  youngs_modulus: Optional[float] = None
  poissons_ratio: Optional[float] = None
  mass_density: Optional[float] = None
  yield_strength: Optional[float] = None

  @property
  def shear_modulus(self) -> float:
    return self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))

  @property
  def bulk_modulus(self) -> float:
    return self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poissons_ratio))

  @property
  def lame_parameters(self) -> tuple[float, float]:
    """Get the Lame parameters for the material.

    Returns: The Lame parameters as a tuple (lambda, mu) for the material.
    """
    lam = (
      self.youngs_modulus
      * self.poissons_ratio
      / ((1.0 + self.poissons_ratio) * (1.0 - 2.0 * self.poissons_ratio))
    )
    mu = self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))
    return lam, mu


def get_lame_parameters_from_youngs_modulus_and_poissons_ratio(
  youngs_modulus: ArrayLike,
  poissons_ratio: ArrayLike,
) -> tuple[ArrayLike, ArrayLike]:
  """Get the Lame parameters from Young's modulus and Poisson's ratio.

  Args:
    youngs_modulus: The Young's modulus of the material [Pa].
    poissons_ratio: The Poisson's ratio of the material [-].

  Returns: The Lame parameters as a tuple (lambda, mu) for the material.
  """
  lam = (
    youngs_modulus
    * poissons_ratio
    / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
  )
  mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
  return lam, mu


def compute_hencky_kirchhoff_stress(
  elastic_log_strain: jnp.ndarray,
  lame_lambda: jnp.ndarray,
  lame_mu: jnp.ndarray,
) -> jnp.ndarray:
  """Kirchhoff stress for Hencky-linear isotropic law.

      τ = λ tr(ε_e) I + 2 μ ε_e
  Args:
    elastic_log_strain: Elastic logarithmic (Hencky) strain tensor ε_e stored in
      tensor-form constitutive space of shape (tensor_dim, tensor_dim).
    lame_lambda: A scalar of the First Lamé parameter λ.
    lame_mu: A scalar of the second Lame parameter (Shear modulus) μ.

  Returns:
    The Kirchhoff stress tensor τ of shape (tensor_dim, tensor_dim).
  """
  tensor_dim = elastic_log_strain.shape[0]
  iden_t = jnp.eye(tensor_dim, dtype=elastic_log_strain.dtype)
  tr_eps = jnp.trace(elastic_log_strain)
  return lame_lambda * tr_eps * iden_t + 2.0 * lame_mu * elastic_log_strain
