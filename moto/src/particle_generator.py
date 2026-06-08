"""Create particle positions from input geometry and other standard shapes."""

from typing import Optional, Tuple

import jax.numpy as jnp
import shapely
import shapely.geometry as shap_geom
from jax.typing import ArrayLike


def particles_from_brep(
  nelx: int,
  nely: int,
  brep: Optional[str] = None,
  shapely_obj: Optional[shap_geom.base.BaseGeometry] = None,
) -> Tuple[ArrayLike, ArrayLike]:
  """Create particles inside an object from a boundary representation file.

  NOTE: The number of elements `nelx` and `nely` are not strictly enforced. The
  function will create particles closest to the desired nelx * nely, but the actual
  number of particles may be less than that. This is due to the fact that the
  particles are created inside the object defined by the boundary representation
  file. The function will create a grid of particles in the x and y directions,
  respectively, and then filter out the particles that are outside the object.

  Args:
    brep: Either a string of the brep file representating a boundary (geojson) or .
    nelx: The number of elements in the x-direction.
    nely: The number of elements in the y-direction.
    shapely_obj: A shapely object representing the geometry. If this is provided, the
      `brep` argument is ignored.

  Returns: A tuple of two arrays:
    - particle_coords: Array of shape (num_particles, 2) containing the x and y
      coordinates of the particles.
    - particle_volumes: Array of shape (num_particles,) containing the volumes of the
      particles.

  """
  if shapely_obj is not None:
    geometry = shapely_obj
  elif brep is not None:
    with open(brep, "r") as f:
      geometry = shapely.from_geojson(f.read())

  min_x, min_y, max_x, max_y = geometry.bounds
  x_coords = jnp.linspace(min_x, max_x, nelx + 1)
  y_coords = jnp.linspace(min_y, max_y, nely + 1)
  dx, dy = jnp.abs(max_x - min_x) / (nelx + 1), jnp.abs(max_y - min_y) / (nely + 1)
  xx, yy = jnp.meshgrid(x_coords, y_coords)

  nodes = jnp.vstack([xx.ravel(), yy.ravel()]).T
  valid_nodes = []
  for node in nodes:
    if geometry.contains(shap_geom.Point(node)):
      valid_nodes.append(node)
  valid_nodes = jnp.array(valid_nodes)

  num_particles = valid_nodes.shape[0]
  particle_volumes = jnp.full((num_particles,), dx * dy)
  return valid_nodes, particle_volumes


def particles_inside_circle(
  center: ArrayLike,
  radius: float,
  nelx: int,
  nely: int,
) -> Tuple[ArrayLike, ArrayLike]:
  """Create particles inside a circle.

  Args:
    center: Array of (center_x, center_y) of the center of the circle.
    radius: The radius of the circle.
    nelx: The number of elements in the x-direction.
    nely: The number of elements in the y-direction.

  Returns:
    pos: Array of shape (num_particles, 2) containing the x and y coordinates of the
      particles.
    vol: Array of shape (num_particles,) containing the volumes of the particles.
  """
  center_point = shap_geom.Point(center[0], center[1])
  circle = center_point.buffer(radius)
  return particles_from_brep(
    nelx=nelx,
    nely=nely,
    shapely_obj=circle,
  )


def particles_inside_rectangle(
  min_x: float, min_y: float, max_x: float, max_y: float, nelx: int, nely: int
) -> Tuple[ArrayLike, ArrayLike]:
  """Create particles inside a rectangle.

  Args:
    min_x: The minimum x-coordinate of the rectangle.
    min_y: The minimum y-coordinate of the rectangle.
    max_x: The maximum x-coordinate of the rectangle.
    max_y: The maximum y-coordinate of the rectangle.
    nelx: The number of elements in the x-direction.
    nely: The number of elements in the y-direction.

  Returns:
    pos: Array of shape (num_particles, 2) containing the x and y coordinates of the
      particles.
    vol: Array of shape (num_particles,) containing the volumes of the particles.
  """
  rectangle = shap_geom.box(min_x, min_y, max_x, max_y)
  return particles_from_brep(
    nelx=nelx,
    nely=nely,
    shapely_obj=rectangle,
  )


def particles_inside_polygon(
  vertices: ArrayLike,
  nelx: int,
  nely: int,
) -> Tuple[ArrayLike, ArrayLike]:
  """Create particles inside a polygon.

  Args:
    vertices: Array of shape (num_vertices, 2) containing the x and y coordinates of the
      vertices of the polygon.
    nelx: The number of elements in the x-direction.
    nely: The number of elements in the y-direction.

  Returns:
    pos: Array of shape (num_particles, 2) containing the x and y coordinates of the
      particles.
    vol: Array of shape (num_particles,) containing the volumes of the particles.
  """
  polygon = shap_geom.Polygon(vertices)
  return particles_from_brep(
    nelx=nelx,
    nely=nely,
    shapely_obj=polygon,
  )


def particles_inside_sphere(
  center: ArrayLike,
  radius: float,
  num_particles_approx: int,
) -> Tuple[ArrayLike, ArrayLike]:
  """Create uniformly spaced particles inside a sphere.

  This method generates points on a uniform Cartesian grid within the
  sphere's bounding box. It then selects only those points that lie inside
  the sphere. The actual number of particles returned may differ slightly
  from `num_particles_approx` due to the grid structure.

  Args:
    center: Array of (center_x, center_y, center_z) of the center of the sphere.
    radius: The radius of the sphere.
    num_particles_approx: The number of particles approximately inside the sphere.

  Returns:
    pos: Array of shape (num_particles, 3) containing the x, y and z coordinates of the
      particles.
    vol: Array of shape (num_particles,) containing the volumes of the particles.
  """
  # Estimate the number of grid points needed along one dimension (k)
  # The ratio of sphere volume to bounding box volume is pi/6.
  # We want n_actual ~ num_particles_approx.
  # n_actual is roughly k^3 * (pi/6) for a grid in the bounding box.
  # So, k^3 ~ num_particles_approx * 6 / pi
  k = int(jnp.cbrt(float(num_particles_approx) * 6.0 / jnp.pi))

  box_min = center - radius
  box_max = center + radius

  # Generate grid coordinates using linspace for each dimension
  x_lin = jnp.linspace(box_min[0], box_max[0], k)
  y_lin = jnp.linspace(box_min[1], box_max[1], k)
  z_lin = jnp.linspace(box_min[2], box_max[2], k)

  x_coords, y_coords, z_coords = jnp.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
  grid_points = jnp.stack(
    [x_coords.ravel(), y_coords.ravel(), z_coords.ravel()], axis=-1
  )

  # Calculate squared distance from the sphere center for each grid point
  dist_sq = jnp.sum((grid_points - center) ** 2, axis=1)

  # Create a boolean mask for points inside or on the sphere boundary
  inside_mask = dist_sq <= (radius**2)

  # Select the grid points that fall inside the sphere
  pos = grid_points[inside_mask]
  num_particles = pos.shape[0]

  # Assign an equal volume fraction to each particle inside the sphere
  sphere_volume = (4.0 / 3.0) * jnp.pi * (radius**3)
  particle_volume = sphere_volume / float(num_particles)
  vol = jnp.full(num_particles, particle_volume)

  return pos, vol
