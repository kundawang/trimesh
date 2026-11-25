"""
proximity.py
---------------

Query mesh- point proximity.
"""

from typing import NamedTuple

import numpy as np

from . import util
from .constants import log_time, tol
from .grouping import group_min
from .triangles import closest_point as _corresponding
from .triangles import points_to_barycentric
from .typed import ArrayLike
from .util import diagonal_dot

try:
    from scipy.spatial import cKDTree
except BaseException as E:
    from .exceptions import ExceptionWrapper

    cKDTree = ExceptionWrapper(E)

# raise a `MemoryError` rather than allocate more than this
_MEMORY_LIMIT = 16 * 2**30


def nearby_faces(mesh, points: ArrayLike):
    """
    For each point find nearby faces relatively quickly.

    The closest point on the mesh to the queried point is guaranteed to be
    on one of the faces listed.

    Does this by finding the nearest vertex on the mesh to each point, and
    then returns all the faces that intersect the axis aligned bounding box
    centered at the queried point and extending to the nearest vertex.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh to query.
    points : (n, 3) float
       Points in space

    Returns
    -----------
    candidates : (points,) int
      Sequence of indexes for mesh.faces
    """
    points = np.asanyarray(points, dtype=np.float64)
    # empty points lists should get empty candidates
    if len(points) == 0:
        return []

    # mishapen points should error
    if not util.is_shape(points, (-1, 3)):
        raise ValueError("points must be (n,3)!")

    # an r-tree containing the axis aligned bounding box for every triangle
    rtree = mesh.triangles_tree
    # a kd-tree containing every vertex of the mesh
    kdtree = cKDTree(mesh.vertices[mesh.referenced_vertices])

    # query in chunks of points that can't exceed the limit even if every
    # face is a candidate and stop before `closest_point` uses ~400b each
    chunk = max(1, _MEMORY_LIMIT // (16 * len(mesh.faces)))
    candidates, total = [], 0
    for start in range(0, len(points), chunk):
        query = points[start : start + chunk]
        # the nearest vertex bounds the distance to the surface
        distance = kdtree.query(query)[0].reshape((-1, 1)) + tol.merge
        try:
            # use the batch API added in 1.4.0 and fixed to actually work in 1.4.1
            ids, counts = rtree.intersection_v(query - distance, query + distance)
            hits = np.array_split(ids, np.cumsum(counts)[:-1])
        except BaseException:
            # fall back to a list comprehension
            hits = [
                list(rtree.intersection(b))
                for b in np.hstack((query - distance, query + distance))
            ]
        total += sum(map(len, hits))
        if total * 400 > _MEMORY_LIMIT:
            raise MemoryError(f"{total} candidate faces exceeds `_MEMORY_LIMIT`")
        candidates.extend(hits)
    return candidates


def closest_point_naive(mesh, points):
    """
    Given a mesh and a list of points find the closest point
    on any triangle.

    Does this by constructing a very large intermediate array and
    comparing every point to every triangle.

    Parameters
    ----------
    mesh : Trimesh
      Takes mesh to have same interfaces as `closest_point`
    points : (m, 3) float
      Points in space

    Returns
    ----------
    closest : (m, 3) float
      Closest point on triangles for each point
    distance : (m,) float
      Distances between point and triangle
    triangle_id : (m,) int
      Index of triangle containing closest point
    """
    # get triangles from mesh
    triangles = mesh.triangles.view(np.ndarray)
    # establish that input points are sane
    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError("triangles shape incorrect")
    if not util.is_shape(points, (-1, 3)):
        raise ValueError("points must be (n,3)")

    # create a giant tiled array of each point tiled len(triangles) times
    points_tiled = np.tile(points, (1, len(triangles)))
    on_triangle = np.array(
        [_corresponding(triangles, i.reshape((-1, 3))) for i in points_tiled]
    )

    # distance squared
    distance_2 = [((i - q) ** 2).sum(axis=1) for i, q in zip(on_triangle, points)]

    triangle_id = np.array([i.argmin() for i in distance_2])

    # closest cartesian point
    closest = np.array([g[i] for i, g in zip(triangle_id, on_triangle)])
    distance = np.array([g[i] for i, g in zip(triangle_id, distance_2)]) ** 0.5

    return closest, distance, triangle_id


def closest_point(mesh, points):
    """
    Given a mesh and a list of points find the closest point
    on any triangle.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh to query
    points : (m, 3) float
      Points in space

    Returns
    ----------
    closest : (m, 3) float
      Closest point on triangles for each point
    distance : (m,)  float
      Distance to mesh.
    triangle_id : (m,) int
      Index of triangle containing closest point
    """
    # a Scene queries every transformed mesh instance in world space
    # and returns a SceneClosestPointResult with node attribution
    from .scene.scene import Scene

    if isinstance(mesh, Scene):
        return closest_point_scene(mesh, points)

    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(points, (-1, 3)):
        raise ValueError("points must be (n,3)!")

    # do a tree- based query for faces near each point
    candidates = nearby_faces(mesh, points)
    # view triangles as an ndarray so we don't have to recompute
    # the MD5 during all of the subsequent advanced indexing
    triangles = mesh.triangles.view(np.ndarray)

    # create the corresponding list of triangles
    # and query points to send to the closest_point function
    all_candidates = np.concatenate(candidates)

    num_candidates = list(map(len, candidates))
    tile_idxs = np.repeat(np.arange(len(points)), num_candidates)
    query_point = points[tile_idxs, :]

    query_tri = triangles[all_candidates]

    # do the computation for closest point
    query_close = _corresponding(query_tri, query_point)
    query_group = np.cumsum(num_candidates)[:-1]

    # vectors and distances for
    # closest point to query point
    query_vector = query_point - query_close
    query_distance = util.diagonal_dot(query_vector, query_vector)

    # get best two candidate indices by arg-sorting the per-query_distances
    qds = np.array_split(query_distance, query_group)
    idxs = np.int32([qd.argsort()[:2] if len(qd) > 1 else [0, 0] for qd in qds])
    idxs[1:] += query_group.reshape(-1, 1)

    # points, distances and triangle ids for best two candidates
    two_points = query_close[idxs]
    two_dists = query_distance[idxs]
    two_candidates = all_candidates[idxs]

    # the first candidate is the best result for unambiguous cases
    result_close = query_close[idxs[:, 0]]
    result_tid = two_candidates[:, 0]
    result_distance = two_dists[:, 0]

    # however: same closest point on two different faces
    # find the best one and correct triangle ids if necessary
    check_distance = np.ptp(two_dists, axis=1) < tol.merge
    check_magnitude = np.all(np.abs(two_dists) > tol.merge, axis=1)

    # mask results where corrections may be apply
    c_mask = np.bitwise_and(check_distance, check_magnitude)

    # get two face normals for the candidate points
    normals = mesh.face_normals[two_candidates[c_mask]]
    # compute normalized surface-point to query-point vectors
    vectors = query_vector[idxs[c_mask]] / two_dists[c_mask].reshape(-1, 2, 1) ** 0.5
    # compare enclosed angle for both face normals
    dots = (normals * vectors).sum(axis=2)

    # take the idx with the most positive angle
    # allows for selecting the correct candidate triangle id
    c_idxs = dots.argmax(axis=1)

    # correct triangle ids where necessary
    # closest point and distance remain valid
    result_tid[c_mask] = two_candidates[c_mask, c_idxs]
    result_distance[c_mask] = two_dists[c_mask, c_idxs]
    result_close[c_mask] = two_points[c_mask, c_idxs]

    # we were comparing the distance squared so
    # now take the square root in one vectorized operation
    result_distance **= 0.5

    return result_close, result_distance, result_tid


# accepted dict-style keys for SceneClosestPointResult mapped to fields
_SCENE_RESULT_KEY_TO_FIELD = {
    "closest": "closest",
    "closest_point": "closest",
    "closest_points": "closest",
    "point": "closest",
    "points": "closest",
    "distance": "distance",
    "distances": "distance",
    "geometry_name": "geometry_name",
    "geometry": "geometry_name",
    "geom_name": "geometry_name",
    "geometry_id": "geometry_name",
    "node_name": "node_name",
    "node": "node_name",
    "node_id": "node_name",
    "face_index": "face_index",
    "face_indices": "face_index",
    "face": "face_index",
    "face_id": "face_index",
    "face_ids": "face_index",
    "triangle_index": "face_index",
    "triangle_indices": "face_index",
    "triangle_id": "face_index",
    "triangle_ids": "face_index",
}


class SceneClosestPointResult(NamedTuple):
    """
    Result of `closest_point_scene`, with one row per query point.

    The canonical tuple order matches the query terms:
    `(closest, distance, geometry_name, node_name, face_index)`.

    Attributes
    -----------
    closest : (m, 3) float
      Closest point on any mesh instance in world coordinates.
    distance : (m,) float
      Euclidean world-space distance to the closest point.
    geometry_name : (m,) object
      Name in `scene.geometry` of the winning geometry, or None
      for query points with no mesh to hit.
    node_name : (m,) object
      Scene graph node carrying the winning instance, or None.
    face_index : (m,) int
      Face index in the *untransformed* source geometry, or -1.
    """

    closest: np.ndarray
    distance: np.ndarray
    geometry_name: np.ndarray
    node_name: np.ndarray
    face_index: np.ndarray

    @property
    def closest_point(self):
        return self.closest

    @property
    def closest_points(self):
        return self.closest

    @property
    def points(self):
        return self.closest

    @property
    def distances(self):
        return self.distance

    @property
    def geometry(self):
        return self.geometry_name

    @property
    def geom_name(self):
        return self.geometry_name

    @property
    def geometry_id(self):
        return self.geometry_name

    @property
    def node(self):
        return self.node_name

    @property
    def node_id(self):
        return self.node_name

    @property
    def face(self):
        return self.face_index

    @property
    def face_id(self):
        return self.face_index

    @property
    def triangle_id(self):
        return self.face_index

    @property
    def triangle_index(self):
        return self.face_index

    def keys(self):
        """
        Canonical mapping keys for dict-style access.
        """
        return ("closest", "distance", "geometry_name", "node_name", "face_index")

    def get(self, key, default=None):
        """
        Dict-style lookup returning `default` for unknown keys.
        """
        if key in _SCENE_RESULT_KEY_TO_FIELD:
            return self[key]
        return default

    def to_dict(self):
        """
        Return the result as a plain dict of arrays.
        """
        return {key: getattr(self, key) for key in self.keys()}

    def __contains__(self, key):
        return key in _SCENE_RESULT_KEY_TO_FIELD

    def __getitem__(self, key):
        # allow result["node_name"] in addition to integer / slice indexing
        if isinstance(key, str):
            field = _SCENE_RESULT_KEY_TO_FIELD.get(key)
            if field is None:
                raise KeyError(key)
            return getattr(self, field)
        return tuple.__getitem__(self, key)


def _scene_mesh_instances(scene):
    """
    Enumerate every mesh instance of a scene in a deterministic order.

    The same geometry instanced under several nodes produces one entry
    per node, each with its own world-frame transform. Non-mesh
    geometries (PointCloud, Path, ...) and meshes without faces are
    skipped.

    Returns
    ----------
    instances : (k,) list of (node, geometry_name, matrix, mesh)
      Sorted by node name so tie resolution is independent of graph
      insertion order.
    """
    from .base import Trimesh

    geometry = getattr(scene, "geometry", None)
    graph = getattr(scene, "graph", None)
    if geometry is None or graph is None:
        raise ValueError("`scene` must be a trimesh.Scene object!")

    instances = []
    for node in graph.nodes_geometry:
        matrix, geometry_name = graph[node]
        if geometry_name is None:
            continue
        mesh = geometry.get(geometry_name) if hasattr(geometry, "get") else None
        # only triangular meshes support closest_point queries
        if not isinstance(mesh, Trimesh) or len(mesh.faces) == 0:
            continue
        instances.append((node, geometry_name, matrix, mesh))

    # hashable node names may be non-string: order by type then value
    instances.sort(key=lambda item: (type(item[0]).__name__, str(item[0])))
    return instances


def _scene_empty_result(count):
    """
    Deterministic sentinel result for `count` query points when the
    scene has no mesh surface to query.
    """
    closest = np.full((count, 3), np.nan, dtype=np.float64)
    distance = np.full(count, np.inf, dtype=np.float64)
    geometry_name = np.full(count, None, dtype=object)
    node_name = np.full(count, None, dtype=object)
    face_index = np.full(count, -1, dtype=np.int64)
    return SceneClosestPointResult(
        closest, distance, geometry_name, node_name, face_index
    )


def closest_point_scene(
    scene=None,
    points=None,
    *,
    mesh=None,
    query_points=None,
    on_empty="return",
    tie_tol=tol.merge,
):
    """
    Find the closest point on the surface of any mesh instance in a
    `trimesh.Scene`, returning world-space results plus the owning
    geometry, node and source face index.

    Every node instance is considered independently, so a geometry
    referenced by multiple nodes is queried once per node with its own
    transform. Transforms are handled exactly for translation, rotation
    and non-uniform scale by baking the node transform into a temporary
    world-space copy of the geometry before querying it. The source
    geometry and its caches are never modified.

    Deterministic conventions:

    * Non-mesh geometry (paths, point clouds, ...) is skipped.
    * A scene with no face-containing mesh returns `inf` distances,
      `NaN` closest points, `None` geometry/node names and `-1` face
      indices. Pass `on_empty="raise"` to raise a `ValueError` instead.
    * Equidistant instances resolve to the first node in the
      lexicographically sorted node order; face ties within a mesh use
      the deterministic rule of `closest_point`.

    Parameters
    ----------
    scene : trimesh.Scene
      Scene to query. May also be passed as the `mesh` keyword.
    points : (m, 3) or (3,) float
      Query points in world coordinates.
    on_empty : "return" or "raise"
      What to do when the scene contains no queryable mesh.
    tie_tol : float or None
      Distances within `tie_tol` plus floating point slack of the
      minimum are considered tied and resolved by node order. Pass 0
      or None to use strict `argmin` selection.

    Returns
    ----------
    result : SceneClosestPointResult
      Named tuple of
      `(closest (m, 3), distance (m,), geometry_name (m,),
      node_name (m,), face_index (m,))` with attribute and dict-style
      aliases such as `result.node_name`, `result["triangle_id"]`.
    """
    # allow `mesh=` / `query_points=` keyword aliases
    if scene is None:
        scene = mesh
    if points is None:
        points = query_points
    if scene is None or points is None:
        raise TypeError("closest_point_scene requires a `scene` and `points` argument")
    if on_empty not in ("return", "raise"):
        raise ValueError('on_empty must be "return" or "raise"')

    points = np.asanyarray(points, dtype=np.float64)
    # be forgiving about a single bare XYZ point
    if points.shape == (3,):
        points = points.reshape((1, 3))
    if not util.is_shape(points, (-1, 3)):
        raise ValueError("points must be (n,3)!")

    count = len(points)
    if count == 0:
        # an empty query has an unambiguous empty result even for empty scenes
        return _scene_empty_result(0)

    instances = _scene_mesh_instances(scene)
    if len(instances) == 0:
        if on_empty == "raise":
            raise ValueError("scene contains no Trimesh geometry with faces!")
        return _scene_empty_result(count)

    k = len(instances)
    all_closest = np.zeros((k, count, 3), dtype=np.float64)
    all_distance = np.zeros((k, count), dtype=np.float64)
    all_face = np.zeros((k, count), dtype=np.int64)
    node_names = np.zeros(k, dtype=object)
    geometry_names = np.zeros(k, dtype=object)

    for i, (node, geometry_name, matrix, source) in enumerate(instances):
        # bake the instance transform into an independent world-space
        # copy so non-uniform scale changes Euclidean distances correctly;
        # visuals are irrelevant to proximity queries
        world = source.copy(include_visual=False)
        world.apply_transform(matrix)
        closest, distance, face_index = closest_point(world, points)

        all_closest[i] = closest
        all_distance[i] = distance
        all_face[i] = face_index
        node_names[i] = node
        geometry_names[i] = geometry_name

    # choose the closest instance per query point; argmin keeps the first
    # instance for exact ties and instances are in sorted node order
    distance_min = all_distance.min(axis=0)
    if tie_tol:
        slack = float(tie_tol) + np.finfo(np.float64).eps * np.maximum(
            1.0, np.abs(distance_min)
        )
        tied = all_distance <= (distance_min + slack)[None, :]
        rank = np.arange(k, dtype=np.int64)[:, None]
        winner = np.where(tied, rank, k).argmin(axis=0)
    else:
        winner = all_distance.argmin(axis=0)

    rows = np.arange(count)
    return SceneClosestPointResult(
        closest=all_closest[winner, rows],
        distance=all_distance[winner, rows],
        geometry_name=geometry_names[winner],
        node_name=node_names[winner],
        face_index=all_face[winner, rows],
    )


# descriptive aliases for the scene-level query
closest_points_scene = closest_point_on_scene = closest_point_scene
scene_closest_point = closest_point_scene


def signed_distance(mesh, points):
    """
    Find the signed distance from a mesh to a list of points.

    * Points OUTSIDE the mesh will have NEGATIVE distance
    * Points within tol.merge of the surface will have POSITIVE distance
    * Points INSIDE the mesh will have POSITIVE distance

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Mesh to query.
    points : (n, 3) float
      Points in space

    Returns
    ----------
    signed_distance : (n,) float
      Signed distance from point to mesh
    """
    # make sure we have a numpy array
    points = np.asanyarray(points, dtype=np.float64)

    # find the closest point on the mesh to the queried points
    closest, distance, triangle_id = closest_point(mesh, points)

    # we only care about nonzero distances
    nonzero = distance > tol.merge

    if not nonzero.any():
        return distance

    # For closest points that project directly in to the triangle, compute sign from
    # triangle normal Project each point in to the closest triangle plane
    nonzero = np.where(nonzero)[0]
    normals = mesh.face_normals[triangle_id]
    projection = (
        points[nonzero]
        - (
            normals[nonzero].T
            * diagonal_dot(points[nonzero] - closest[nonzero], normals[nonzero])
        ).T
    )

    # Determine if the projection lies within the closest triangle
    barycentric = points_to_barycentric(mesh.triangles[triangle_id[nonzero]], projection)
    ontriangle = ~(
        ((barycentric < -tol.merge) | (barycentric > 1 + tol.merge)).any(axis=1)
    )

    # Where projection does lie in the triangle, compare vector to projection to the
    # triangle normal to compute sign
    sign = np.sign(
        diagonal_dot(
            normals[nonzero[ontriangle]],
            points[nonzero[ontriangle]] - projection[ontriangle],
        )
    )
    distance[nonzero[ontriangle]] *= -1.0 * sign

    # For all other triangles, resort to raycasting against the entire mesh
    inside = mesh.ray.contains_points(points[nonzero[~ontriangle]])
    sign = (inside.astype(int) * 2) - 1.0

    # apply sign to previously computed distance
    distance[nonzero[~ontriangle]] *= sign

    return distance


class NearestQueryResult:
    """
    Stores the nearest points and attributes for nearest points queries.
    """

    def __init__(self):
        self.nearest = None
        self.distances = None
        self.normals = None
        self.triangle_indices = None
        self.barycentric_coordinates = None
        self.interpolated_normals = None
        self.vertex_indices = None

    def has_normals(self):
        return self.normals is not None or self.interpolated_normals is not None


class ProximityQuery:
    """
    Proximity queries for the current mesh.
    """

    def __init__(self, mesh):
        self._mesh = mesh

    @log_time
    def on_surface(self, points):
        """
        Given list of points, for each point find the closest point
        on any triangle of the mesh.

        Parameters
        ----------
        points : (m,3) float, points in space

        Returns
        ----------
        closest : (m, 3) float
          Closest point on triangles for each point
        distance : (m,) float
          Distance to surface
        triangle_id : (m,) int
          Index of closest triangle for each point.
        """
        return closest_point(mesh=self._mesh, points=points)

    def vertex(self, points):
        """
        Given a set of points, return the closest vertex index to each point

        Parameters
        ----------
        points : (n, 3) float
          Points in space

        Returns
        ----------
        distance : (n,) float
           Distance from source point to vertex.
        vertex_id : (n,) int
          Index of mesh.vertices for closest vertex.
        """
        tree = self._mesh.kdtree
        return tree.query(points)

    def signed_distance(self, points):
        """
        Find the signed distance from a mesh to a list of points.

        * Points OUTSIDE the mesh will have NEGATIVE distance
        * Points within tol.merge of the surface will have POSITIVE distance
        * Points INSIDE the mesh will have POSITIVE distance

        Parameters
        -----------
        points : (n, 3) float
          Points in space

        Returns
        ----------
        signed_distance : (n,) float
          Signed distance from point to mesh.
        """
        return signed_distance(self._mesh, points)


class SceneProximityQuery:
    """
    Proximity queries against every transformed mesh instance of a
    `trimesh.Scene`, exposed as `scene.nearest`.
    """

    def __init__(self, scene):
        self._scene = scene

    def closest_point(self, points, **kwargs):
        """
        Find the closest world-space surface point in the scene.

        See `closest_point_scene` for parameters and the
        `SceneClosestPointResult` return value.
        """
        return closest_point_scene(self._scene, points, **kwargs)

    # trimesh mesh queries use `nearest.on_surface(points)`
    on_surface = closest_point

    def query(self, points, **kwargs):
        """
        Alias for `closest_point`.
        """
        return closest_point_scene(self._scene, points, **kwargs)

    def __call__(self, points, **kwargs):
        return closest_point_scene(self._scene, points, **kwargs)


def longest_ray(mesh, points, directions):
    """
    Find the lengths of the longest rays which do not intersect the mesh
    cast from a list of points in the provided directions.

    Parameters
    -----------
    points : (n, 3) float
      Points in space.
    directions : (n, 3) float
      Directions of rays.

    Returns
    ----------
    signed_distance : (n,) float
      Length of rays.
    """
    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(points, (-1, 3)):
        raise ValueError("points must be (n,3)!")

    directions = np.asanyarray(directions, dtype=np.float64)
    if not util.is_shape(directions, (-1, 3)):
        raise ValueError("directions must be (n,3)!")

    if len(points) != len(directions):
        raise ValueError("number of points must equal number of directions!")

    _faces, rays, locations = mesh.ray.intersects_id(
        points, directions, return_locations=True, multiple_hits=True
    )
    if len(rays) > 0:
        distances = np.linalg.norm(locations - points[rays], axis=1)
    else:
        distances = np.array([])

    # Reject intersections at distance less than tol.planar
    rays = rays[distances > tol.planar]
    distances = distances[distances > tol.planar]

    # Add infinite length for those with no valid intersection
    no_intersections = np.setdiff1d(np.arange(len(points)), rays)
    rays = np.concatenate((rays, no_intersections))
    distances = np.concatenate((distances, np.repeat(np.inf, len(no_intersections))))
    return group_min(rays, distances)


def max_tangent_sphere(
    mesh, points, inwards=True, normals=None, threshold=1e-6, max_iter=100
):
    """
    Find the center and radius of the sphere which is tangent to
    the mesh at the given point and at least one more point with no
    non-tangential intersections with the mesh.

    Masatomo Inui, Nobuyuki Umezu & Ryohei Shimane (2016)
    Shrinking sphere:
    A parallel algorithm for computing the thickness of 3D objects,
    Computer-Aided Design and Applications, 13:2, 199-207,
    DOI: 10.1080/16864360.2015.1084186

    Parameters
    ----------
    points : (n, 3) float
      Points in space.
    inwards : bool
      Whether to have the sphere inside or outside the mesh.
    normals : (n, 3) float or None
      Normals of the mesh at the given points
      if is None computed automatically.

    Returns
    ----------
    centers : (n,3) float
      Centers of spheres
    radii : (n,) float
      Radii of spheres
    """
    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(points, (-1, 3)):
        raise ValueError("points must be (n,3)!")

    if normals is not None:
        normals = np.asanyarray(normals, dtype=np.float64)
        if not util.is_shape(normals, (-1, 3)):
            raise ValueError("normals must be (n,3)!")

        if len(points) != len(normals):
            raise ValueError("number of points must equal number of normals!")
    else:
        normals = mesh.face_normals[closest_point(mesh, points)[2]]

    if inwards:
        normals = -normals

    # Find initial tangent spheres
    distances = longest_ray(mesh, points, normals)
    radii = distances * 0.5
    not_converged = np.ones(len(points), dtype=bool)  # boolean mask

    # If ray is infinite, find the vertex which is furthest from our point
    # when projected onto the ray. I.e. find v which maximises
    # (v-p).n = v.n - p.n.
    # We use a loop rather a vectorised approach to reduce memory cost
    # it also seems to run faster.
    for i in np.where(np.isinf(distances))[0]:
        projections = np.dot(mesh.vertices - points[i], normals[i])

        # If no points lie outside the tangent plane, then the radius is infinite
        # otherwise we have a point outside the tangent plane, take the one with maximal
        # projection
        if projections.max() < tol.planar:
            radii[i] = np.inf
            not_converged[i] = False
        else:
            vertex = mesh.vertices[projections.argmax()]
            radii[i] = np.dot(vertex - points[i], vertex - points[i]) / (
                2 * np.dot(vertex - points[i], normals[i])
            )

    # Compute centers
    centers = points + normals * np.nan_to_num(radii.reshape(-1, 1))
    centers[np.isinf(radii)] = [np.nan, np.nan, np.nan]

    # Our iterative process terminates when the difference in sphere
    # radius is less than threshold*D
    D = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    convergence_threshold = threshold * D
    n_iter = 0
    while not_converged.sum() > 0 and n_iter < max_iter:
        n_iter += 1
        n_points, n_dists, _n_faces = mesh.nearest.on_surface(centers[not_converged])

        # If the distance to the nearest point is the same as the distance
        # to the start point then we are done.
        done = (
            np.abs(
                n_dists
                - np.linalg.norm(centers[not_converged] - points[not_converged], axis=1)
            )
            < tol.planar
        )
        not_converged[np.where(not_converged)[0][done]] = False

        # Otherwise find the radius and center of the sphere tangent to the mesh
        # at the point and the nearest point.
        diff = n_points[~done] - points[not_converged]
        old_radii = radii[not_converged].copy()
        radii[not_converged] = diagonal_dot(diff, diff) / (
            2 * diagonal_dot(diff, normals[not_converged])
        )
        centers[not_converged] = points[not_converged] + normals[not_converged] * radii[
            not_converged
        ].reshape(-1, 1)

        # If change in radius is less than threshold we have converged
        cvged = old_radii - radii[not_converged] < convergence_threshold
        not_converged[np.where(not_converged)[0][cvged]] = False

    return centers, radii


def thickness(mesh, points, exterior=False, normals=None, method="max_sphere"):
    """
    Find the thickness of the mesh at the given points.

    Parameters
    ----------
    points : (n, 3) float
      Points in space
    exterior : bool
      Whether to compute the exterior thickness
      (a.k.a. reach)
    normals : (n, 3) float
      Normals of the mesh at the given points
      If is None computed automatically.
    method : string
      One of 'max_sphere' or 'ray'

    Returns
    ----------
    thickness : (n,) float
      Thickness at given points.
    """
    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(points, (-1, 3)):
        raise ValueError("points must be (n,3)!")

    if normals is not None:
        normals = np.asanyarray(normals, dtype=np.float64)
        if not util.is_shape(normals, (-1, 3)):
            raise ValueError("normals must be (n,3)!")

        if len(points) != len(normals):
            raise ValueError("number of points must equal number of normals!")
    else:
        normals = mesh.face_normals[closest_point(mesh, points)[2]]

    if method == "max_sphere":
        _centers, radius = max_tangent_sphere(
            mesh=mesh, points=points, inwards=not exterior, normals=normals
        )
        thickness = radius * 2
        return thickness

    elif method == "ray":
        if exterior:
            return longest_ray(mesh, points, normals)
        else:
            return longest_ray(mesh, points, -normals)
    else:
        raise ValueError('Invalid method, use "max_sphere" or "ray"')
