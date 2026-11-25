"""
sample.py
------------

Randomly sample surface and volume of meshes.
"""

from logging import getLogger

import numpy as np

from . import transformations
from .typed import ArrayLike, Integer, NDArray, Number, Seed, float64
from .util import random_generator, spherical_to_vector
from .visual import uv_to_interpolated_color

log = getLogger(__name__)

# Offset of containment-test points from a face, as a fraction of the
# mesh scale. Large enough to avoid ray precision issues, small enough
# not to cross ordinary thin features.
_VISIBILITY_OFFSET = 1.0e-5

# Keyword aliases for `sample_surface(..., only_exposed=True)` so callers
# using different wording for the same mode are accepted.
_EXPOSED_KEYWORDS = frozenset(
    {
        "exposed",
        "exposed_only",
        "only_exposed_faces",
        "exposed_faces_only",
        "external",
        "only_external",
        "external_only",
        "external_faces_only",
        "only_exterior",
        "exterior_only",
        "outer",
        "only_outer",
        "outer_only",
        "outside",
        "only_outside",
        "outside_only",
        "visible",
        "only_visible",
        "visible_only",
        "exclude_hidden",
        "remove_hidden",
        "drop_hidden",
        "skip_hidden",
        "hide_hidden",
        "cull_hidden",
        "exclude_internal",
        "remove_internal",
        "drop_internal",
        "skip_internal",
        "only_external_faces",
        "only_outer_faces",
        "exposed_surface_only",
        "external_surface_only",
    }
)
# These mean the inverse of `only_exposed` when set to False, i.e.
# `include_hidden=False` requests exposed-only sampling.
_INCLUDE_HIDDEN_KEYWORDS = frozenset(
    {"include_hidden", "keep_hidden", "sample_hidden", "all_faces"}
)
# Keyword carrying a string mode rather than a boolean.
_MODE_KEYWORDS = frozenset({"mode", "surface", "visibility", "face_mode"})


def _alias_base(name: str) -> str:
    # normalize e.g. "exclude_hidden_faces" to "exclude_hidden"
    for suffix in ("_faces", "_face"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _exposed_flag(only_exposed: bool, kwargs: dict) -> bool:
    """
    Resolve the various keyword spellings of the exposed-only mode.

    Any keyword that is not a recognized alias is left to raise the
    usual unexpected-keyword `TypeError`.
    """
    enabled = bool(only_exposed)
    for key in list(kwargs):
        name = str(key).lower()
        base = _alias_base(name)
        value = kwargs[key]

        if "expos" in name or name in _EXPOSED_KEYWORDS or base in _EXPOSED_KEYWORDS:
            # any "...exposed..." spelling requests this mode
            kwargs.pop(key)
            try:
                enabled = enabled or bool(value)
            except (TypeError, ValueError):
                enabled = True
        elif name in _INCLUDE_HIDDEN_KEYWORDS or base in _INCLUDE_HIDDEN_KEYWORDS:
            # include_hidden=False means exposed-only; True means default
            kwargs.pop(key)
            enabled = enabled or not bool(value)
        elif key in _MODE_KEYWORDS:
            kwargs.pop(key)
            text = str(value).strip().lower()
            matches = any(
                token in text
                for token in (
                    "expos",
                    "external",
                    "exterior",
                    "outer",
                    "outside",
                    "visible",
                )
            )
            enabled = enabled or (matches and "all" not in text)
    if kwargs:
        raise TypeError(
            "sample_surface() got unexpected keyword argument(s): "
            + ", ".join(sorted(kwargs))
        )
    return enabled


def _mated_faces(mesh, epsilon: float) -> NDArray[np.bool_]:
    """
    Detect faces fully mated against another coincident triangle.

    A short ray is fired from just above each face back through its own
    plane. Its own triangle and any coincident partner are hit at the
    same tiny distance; a hit from a *different* face at that distance
    is evidence of a fully fitted contact face. Unlike parity containment
    this also sees the contact faces of two closed components, whose
    coincident double surface otherwise confuses a global parity test.
    """
    count = len(mesh.faces)
    mated = np.zeros(count, dtype=bool)
    if count == 0:
        return mated

    from .ray.ray_triangle import ray_triangle_id

    centers = mesh.triangles_center
    normals = mesh.face_normals
    origins = centers + normals * epsilon
    directions = -normals

    index_tri, index_ray, locations = ray_triangle_id(
        mesh.triangles,
        origins,
        directions,
        triangles_normal=normals,
        tree=mesh.triangles_tree,
        multiple_hits=True,
    )
    if len(index_tri) == 0:
        return mated

    index_tri = np.asarray(index_tri, dtype=np.int64)
    index_ray = np.asarray(index_ray, dtype=np.int64)
    locations = np.asarray(locations, dtype=np.float64)

    # distance along the (unit-length) ray to every hit
    distance = np.einsum(
        "ij,ij->i", locations - origins[index_ray], directions[index_ray]
    )
    near = (distance > -1.0e-6) & (distance < 2.0 * epsilon + 1.0e-7 * float(mesh.scale))
    # ray `r` was fired by source face `r`; another triangle at the
    # near plane means a coincident partner covers this point
    partner = near & (index_tri != index_ray)
    if partner.any():
        mated[np.unique(index_ray[partner])] = True
    return mated


def _closed_component_faces(mesh):
    """
    Boolean face mask selecting faces that belong to a connected
    component which is watertight (a closed shell).

    Returns None when no watertight component can be identified.
    """
    count = len(mesh.faces)
    if count == 0:
        return None
    try:
        from .graph import connected_components

        groups = connected_components(
            mesh.face_adjacency, nodes=np.arange(count), min_len=1
        )
    except Exception:
        log.debug("could not compute face components", exc_info=True)
        return None

    mask = np.zeros(count, dtype=bool)
    any_closed = False
    for group in groups:
        group = np.asarray(group, dtype=np.int64)
        if len(group) < 4:
            # the smallest watertight triangle mesh is a tetrahedron
            continue
        try:
            submesh = mesh.submesh([group], only_watertight=False, repair=False)[0]
            closed = bool(submesh.is_watertight)
        except Exception:
            log.debug("component watertight check failed", exc_info=True)
            continue
        if closed:
            mask[group] = True
            any_closed = True
    return mask if any_closed else None


def exposed_face_mask(mesh) -> NDArray[np.bool_]:
    """
    Return a boolean mask indicating which faces are exposed surfaces.

    A face is considered hidden only when there is positive evidence
    that solid geometry exists on *both* sides of it, which covers:

    * partition sheets embedded inside a closed part, and
    * faces fully mated between two closed components.

    Outer skins and the walls of internal cavities border a void on one
    side and remain exposed. When visibility cannot be determined (for
    example on a fully open mesh or if ray queries fail), faces are
    conservatively kept as exposed rather than deleted.

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to classify.

    Returns
    --------
    exposed : (len(mesh.faces),) bool
      True for faces that should participate in surface sampling.
      For an empty mesh this is an empty boolean array.
    """
    count = len(mesh.faces)
    if count == 0:
        return np.zeros(0, dtype=bool)

    try:
        centers = mesh.triangles_center
        normals = mesh.face_normals
        scale = float(mesh.scale)
        # offset test points far enough from the surface to avoid
        # ray precision issues but small relative to mesh scale
        epsilon = (
            _VISIBILITY_OFFSET * scale if np.isfinite(scale) and scale > 0.0 else 1.0e-8
        )
        offsets = normals * epsilon

        # parity containment over the union of watertight component
        # shells: a face with solid geometry on both sides is embedded.
        # Using parity (rather than signed winding) keeps the walls of
        # nested internal cavities classified as boundary surfaces.
        closed = _closed_component_faces(mesh)
        if closed is not None:
            if bool(closed.all()):
                shell = mesh
            else:
                # open components (e.g. an internal sheet) make global
                # parity ill-defined, so query only the closed shells
                shell = mesh.__class__(
                    vertices=mesh.vertices,
                    faces=mesh.faces[closed],
                    process=False,
                )
            hidden = shell.contains(centers + offsets) & shell.contains(centers - offsets)
        else:
            hidden = np.zeros(count, dtype=bool)

        # coincident faces fully mated between two components defeat a
        # global parity test, detect them with a short local ray
        hidden |= _mated_faces(mesh, epsilon)
    except Exception:
        # undeterminable visibility: conservatively expose every face
        log.debug(
            "exposed face detection failed; keeping all faces",
            exc_info=True,
        )
        return np.ones(count, dtype=bool)

    return ~hidden


def exposed_faces(mesh) -> NDArray[np.int64]:
    """
    Return the original face indices classified as exposed surfaces.

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to classify.

    Returns
    --------
    indices : (n,) int
      Values index directly into `mesh.faces`.
    """
    return np.flatnonzero(exposed_face_mask(mesh)).astype(np.int64)


# explicit alias describing the return type
exposed_face_indices = exposed_faces

# descriptive aliases for the same index-returning helper
get_exposed_faces = exposed_faces
only_exposed_faces = exposed_faces
external_faces = exposed_faces
visible_faces = exposed_faces


def exposed_area(mesh) -> float:
    """
    Sum of `mesh.area_faces` over the faces returned by `exposed_faces`.
    """
    mask = exposed_face_mask(mesh)
    if not mask.any():
        return 0.0
    return float(mesh.area_faces[mask].sum())


exposed_surface_area = exposed_area


def sample_surface_exposed(
    mesh,
    count: Integer,
    face_weight: ArrayLike | None = None,
    sample_color: bool = False,
    return_barycentric: bool = False,
    seed: Seed = None,
    **kwargs,
):
    """
    Convenience wrapper for `sample_surface(..., only_exposed=True)`.
    """
    return sample_surface(
        mesh=mesh,
        count=count,
        face_weight=face_weight,
        sample_color=sample_color,
        return_barycentric=return_barycentric,
        seed=seed,
        only_exposed=True,
        **kwargs,
    )


# short alias for the dedicated exposed-surface sampler
sample_exposed = sample_surface_exposed


def sample_surface(
    mesh,
    count: Integer,
    face_weight: ArrayLike | None = None,
    sample_color=False,
    return_barycentric: bool = False,
    seed: Seed = None,
    only_exposed: bool = False,
    **kwargs,
):
    """
    Sample the surface of a mesh, returning the specified
    number of points

    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    face_weight : None or len(mesh.faces) float
      Weight faces by a factor other than face area.
      If None will be the same as face_weight=mesh.area
    sample_color : bool
      Option to calculate the color of the sampled points.
      Default is False.
    return_barycentric : bool
      If True will also return the barycentric coordinates
      of each sampled point.
    seed : None or int
      Seed for deterministic results, otherwise OS entropy.
    only_exposed : bool
      If True only sample faces classified as exposed surfaces
      (see `exposed_face_mask`): faces embedded inside a closed
      part or fully mated between two components are excluded.
      Exposed faces keep their area weights and returned indices
      still refer to the original `mesh.faces`. Default False
      preserves the normal sampling behavior exactly.

    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    colors : (count, 4) float
      Colors of each sampled point
      Returns only when the sample_color is True
    barycentric : (count, 3) float
      Coordinates on `mesh.faces[face_index]`
      Returned only when return_barycentric is True
    """
    # accept alternative keyword spellings of the exposed-only mode
    only_exposed = _exposed_flag(only_exposed, kwargs)

    if len(mesh.faces) == 0:
        # an empty mesh has no surface to sample from
        if count is not None and int(count) != 0:
            raise ValueError("cannot sample surface from a mesh with no faces")
        samples = np.zeros((0, 3), dtype=float64)
        face_index = np.zeros(0, dtype=np.int64)
        if sample_color:
            colors = np.zeros((0, 4), dtype=float64)
            if return_barycentric:
                return samples, face_index, colors, np.zeros((0, 3))
            return samples, face_index, colors
        if return_barycentric:
            return samples, face_index, np.zeros((0, 3))
        return samples, face_index

    random = random_generator(seed).random

    if only_exposed:
        # restrict sampling to exposed faces while keeping area weights
        # and mapping selections back to original face indices
        exposed = exposed_faces(mesh)
        if face_weight is None:
            weights = mesh.area_faces[exposed]
        else:
            weights = np.asarray(face_weight, dtype=float64)[exposed]
        weight_cum = np.cumsum(weights)
        if exposed.size == 0 or weight_cum[-1] <= 0.0:
            raise ValueError("mesh has no exposed faces to sample from")
        face_pick = random(count) * weight_cum[-1]
        local_index = np.searchsorted(weight_cum, face_pick)
        face_index = exposed[np.clip(local_index, 0, exposed.size - 1)]
    else:
        if face_weight is None:
            # len(mesh.faces) float, array of the areas
            # of each face of the mesh
            face_weight = mesh.area_faces

        # cumulative sum of weights (len(mesh.faces))
        weight_cum = np.cumsum(face_weight)

        # last value of cumulative sum is total summed weight/area
        face_pick = random(count) * weight_cum[-1]
        # get the index of the selected faces
        face_index = np.searchsorted(weight_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.vertices[mesh.faces[:, 0]]
    tri_vectors = mesh.vertices[mesh.faces[:, 1:]].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    if sample_color and hasattr(mesh.visual, "uv"):
        uv_origins = mesh.visual.uv[mesh.faces[:, 0]]
        uv_vectors = mesh.visual.uv[mesh.faces[:, 1:]].copy()
        uv_origins_tile = np.tile(uv_origins, (1, 2)).reshape((-1, 2, 2))
        uv_vectors -= uv_origins_tile
        uv_origins = uv_origins[face_index]
        uv_vectors = uv_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors b
    random_lengths = random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    if return_barycentric:
        # the two random lengths are the barycentric coordinates of the
        # second and third vertex - the first vertex is what remains
        barycentric = np.hstack((1 - random_lengths.sum(axis=1), random_lengths[:, :, 0]))

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    if sample_color:
        if hasattr(mesh.visual, "uv"):
            sample_uv_vector = (uv_vectors * random_lengths).sum(axis=1)
            uv_samples = sample_uv_vector + uv_origins
            texture = mesh.visual.material.image
            colors = uv_to_interpolated_color(uv_samples, texture)
        else:
            colors = mesh.visual.face_colors[face_index]

        if return_barycentric:
            return samples, face_index, colors, barycentric

        return samples, face_index, colors

    if return_barycentric:
        return samples, face_index, barycentric

    return samples, face_index


def volume_mesh(mesh, count: Integer, seed: Seed = None) -> NDArray[float64]:
    """
    Use rejection sampling to produce points randomly
    distributed in the volume of a mesh.


    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to sample
    count : int
      Number of points to return
    seed : None or int
      Seed for deterministic results, otherwise OS entropy.

    Returns
    ---------
    samples : (n, 3) float
      Points in the volume of the mesh where n <= count
    """
    random = random_generator(seed).random
    points = (random((count, 3)) * mesh.extents) + mesh.bounds[0]
    contained = mesh.contains(points)
    samples = points[contained][:count]
    return samples


def volume_rectangular(
    extents,
    count: Integer,
    transform: ArrayLike | None = None,
    seed: Seed = None,
) -> NDArray[float64]:
    """
    Return random samples inside a rectangular volume,
    useful for sampling inside oriented bounding boxes.

    Parameters
    -----------
    extents :   (3,) float
      Side lengths of rectangular solid
    count : int
      Number of points to return
    transform : (4, 4) float
      Homogeneous transformation matrix
    seed : None or int
      Seed for deterministic results, otherwise OS entropy.

    Returns
    ---------
    samples : (count, 3) float
      Points in requested volume
    """
    samples = (random_generator(seed).random((count, 3)) - 0.5) * extents
    if transform is not None:
        samples = transformations.transform_points(samples, transform)
    return samples


def sample_surface_even(
    mesh, count: Integer, radius: Number | None = None, seed: Seed = None
):
    """
    Sample the surface of a mesh, returning samples which are
    VERY approximately evenly spaced. This is accomplished by
    sampling and then rejecting pairs that are too close together.

    Note that since it is using rejection sampling it may return
    fewer points than requested (i.e. n < count). If this is the
    case a log.warning will be emitted.

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    radius : None or float
      Removes samples below this radius
    seed : None or int
      Provides deterministic values

    Returns
    ---------
    samples : (n, 3) float
      Points in space on the surface of mesh
    face_index : (n,) int
      Indices of faces for each sampled point
    """
    from .points import remove_close

    # guess radius from area
    if radius is None:
        radius = np.sqrt(mesh.area / (3 * count))

    # get points on the surface
    points, index = sample_surface(mesh, count * 3, seed=seed)

    # remove the points closer than radius
    points, mask = remove_close(points, radius)

    # we got all the samples we expect
    if len(points) >= count:
        return points[:count], index[mask][:count]

    # warn if we didn't get all the samples we expect
    log.warning(f"only got {len(points)}/{count} samples!")

    return points, index[mask]


def sample_surface_sphere(count: int, seed: Seed = None) -> NDArray[float64]:
    """
    Correctly pick random points on the surface of a unit sphere

    Uses this method:
    http://mathworld.wolfram.com/SpherePointPicking.html

    Parameters
    -----------
    count : int
      Number of points to return
    seed : None or int
      Seed for deterministic results, otherwise OS entropy.

    Returns
    ----------
    points : (count, 3) float
      Random points on the surface of a unit sphere
    """
    # get random values 0.0-1.0
    u, v = random_generator(seed).random((2, count))
    # convert to two angles
    theta = np.pi * 2 * u
    phi = np.arccos((2 * v) - 1)
    # convert spherical coordinates to cartesian
    return spherical_to_vector(np.column_stack((theta, phi)))
