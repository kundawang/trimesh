import numpy as np

from ... import graph, grouping, util
from ...constants import tol_path
from ...typed import ArrayLike, NDArray
from ..entities import Arc, Line


def dict_to_path(as_dict):
    """
    Turn a pure dict into a dict containing entity objects that
    can be sent directly to a Path constructor.

    Parameters
    ------------
    as_dict : dict
      Has keys: 'vertices', 'entities'

    Returns
    ------------
    kwargs : dict
      Has keys: 'vertices', 'entities'
    """
    # start kwargs with initial value
    result = as_dict.copy()
    # map of constructors
    loaders = {"Arc": Arc, "Line": Line}
    # pre- allocate entity array
    entities = [None] * len(as_dict["entities"])
    # run constructor for dict kwargs
    for entity_index, entity in enumerate(as_dict["entities"]):
        if entity["type"] == "Line":
            entities[entity_index] = loaders[entity["type"]](points=entity["points"])
        else:
            entities[entity_index] = loaders[entity["type"]](
                points=entity["points"], closed=entity["closed"]
            )
    result["entities"] = entities

    return result


class FaceIndexMap(list):
    """
    Ordered per-contour source face indices.

    Each item is an `(m,)` int array of original mesh face indexes
    for one assembled entity/closed contour, in entity order. The
    normal list interface (`mapping[i]`, iteration, `len`, `zip`) is
    the stable contour lookup; dict-style (`items`, `values`, `get`)
    and calling `mapping(i)` are also provided for convenience.
    """

    def __call__(self, key=None):
        # allow `face_index_by_contour(i)` as well as `[i]`
        if key is None:
            return self
        return self[key]

    def keys(self):
        return range(len(self))

    def values(self):
        return list(self)

    def items(self):
        return list(enumerate(self))

    def get(self, key, default=None):
        try:
            return self[key]
        except (IndexError, KeyError, TypeError):
            return default

    def as_dict(self) -> dict:
        return dict(enumerate(self))


def assign_face_index(entities, edges, index):
    """
    Associate source face indexes with assembled path entities.

    Every consecutive pair of an entity's `points` is an undirected
    edge in `edges`, so keying source segments by their canonical
    sorted vertex pair tracks face indexes through the DFS assembly
    without depending on traversal order. For every entity this sets
    `entity.metadata["face_index"]` to an `(len(entity.points) - 1,)`
    int array aligned with the entity's segments and returns the
    sorted unique source face indexes touched by each entity.

    Parameters
    -----------
    entities : sequence of Entity
      Entities produced from `edges`, in their final order.
    edges : (n, 2) int
      Merged vertex indexes of the source segments, in source order.
    index : (n,) int
      Original mesh face index for every source segment.

    Returns
    ----------
    grouped : list of (m,) int64
      Sorted unique source face indexes for every entity.
    """
    edges = np.asanyarray(edges, dtype=np.int64)
    index = np.asanyarray(index, dtype=np.int64).reshape(-1)
    if len(index) != len(edges):
        raise ValueError("one face index is required per segment")

    # canonical undirected edge -> every source row using that edge
    edge_rows = {}
    for row in range(len(edges)):
        a, b = int(edges[row, 0]), int(edges[row, 1])
        key = (a, b) if a < b else (b, a)
        edge_rows.setdefault(key, []).append(row)

    # rotate duplicate source edges deterministically across segments
    consumed = {}
    grouped = []
    for entity in entities:
        points = np.asanyarray(entity.points, dtype=np.int64).reshape(-1)
        segment_faces = np.zeros(max(len(points) - 1, 0), dtype=np.int64)
        unique_rows_here = set()
        for j in range(len(points) - 1):
            a, b = int(points[j]), int(points[j + 1])
            key = (a, b) if a < b else (b, a)
            rows = edge_rows.get(key)
            if not rows:
                # assembled entities only contain source edges; this
                # guards against silently emitting a wrong mapping.
                raise KeyError(f"assembled segment {key} has no source edge")
            which = consumed.get(key, 0)
            row = rows[min(which, len(rows) - 1)]
            consumed[key] = which + 1
            segment_faces[j] = index[row]
            # duplicate merged edges are geometrically the same segment,
            # so attribute every source row to this one contour
            unique_rows_here.update(rows)

        faces = np.unique(index[sorted(unique_rows_here)]).astype(np.int64)
        # metadata is the conventional public bag; kwargs keeps the
        # array away from metadata consumers that only encode scalars
        entity.metadata["face_index"] = segment_faces
        entity.metadata["face_index_unique"] = faces
        entity.kwargs["face_index"] = segment_faces
        entity.kwargs["face_index_unique"] = faces
        grouped.append(faces)

    return grouped


def contour_face_metadata(entities, flat_index, grouped=None) -> dict:
    """
    Build the stable contour-to-face-index metadata bundle attached
    to section paths.

    Parameters
    -----------
    entities : sequence of Entity
      Assembled path entities/closed contours.
    flat_index : (n,) int
      Source face index for every raw intersection segment.
    grouped : None or list of (m,) int
      Output of `assign_face_index`; computed if not provided.

    Returns
    ----------
    metadata : dict
      Metadata keys shared by `section` and `section_multiplane`.
    """
    flat_index = np.asanyarray(flat_index, dtype=np.int64).reshape(-1)
    if grouped is None:
        grouped = [np.unique(flat_index).astype(np.int64)] * len(entities)

    mapping = FaceIndexMap(grouped)
    metadata = {
        # raw segment order, retained for backwards compatibility
        "face_index": flat_index,
        "face_index_flat": flat_index,
        # stable lookup by closed contour / entity position
        "face_index_by_entity": mapping,
        "face_index_by_contour": mapping,
        "face_index_per_entity": mapping,
        "face_index_per_contour": mapping,
        "contour_face_index": mapping,
        "entity_face_index": mapping,
        "source_face_index_by_contour": mapping,
        "source_faces_by_contour": mapping,
        # explicit key -> face indexes mapping
        "face_index_map": mapping.as_dict(),
        "contour_face_map": mapping.as_dict(),
    }
    return metadata


def lines_to_path(lines: ArrayLike, index: NDArray[np.int64] | None = None) -> dict:
    """
    Turn line segments into argument to be used for a Path2D or Path3D.

    Parameters
    ------------
    lines : (n, 2, dimension) or (n, dimension) float
      Line segments or connected polyline curve in 2D or 3D
    index : (n,) int64
      If passed save an index for each line segment. For `(n, 2, dim)`
      segments this is one source index per segment; the assembled
      entities additionally carry `entity.metadata["face_index"]`.

    Returns
    -----------
    kwargs : dict
      kwargs for Path constructor
    """
    lines = np.asanyarray(lines, dtype=np.float64)

    if index is not None:
        index = np.asanyarray(index, dtype=np.int64)

    if util.is_shape(lines, (-1, (2, 3))):
        # the case where we have a list of points
        # we are going to assume they are connected
        entity = Line(np.arange(len(lines)))
        if index is not None and len(index) == max(len(lines) - 1, 0):
            entity.metadata["face_index"] = np.asanyarray(index, dtype=np.int64)
            entity.kwargs["face_index"] = entity.metadata["face_index"]
        result = {"entities": np.array([entity]), "vertices": lines}
        if index is not None:
            flat = np.asanyarray(index, dtype=np.int64).reshape(-1)
            result["metadata"] = contour_face_metadata([entity], flat)
        return result
    elif util.is_shape(lines, (-1, 2, (2, 3))):
        # case where we have line segments in 2D or 3D
        dimension = lines.shape[-1]
        # convert lines to even number of (n, dimension) points
        lines = lines.reshape((-1, dimension))
        # merge duplicate vertices
        unique, inverse = grouping.unique_rows(lines, digits=tol_path.merge_digits)
        # merged vertex index for each source segment, source order
        edges = inverse.reshape((-1, 2))
        # use scipy edges_to_path to skip creating
        # a bajillion individual line entities which
        # will be super slow vs. fewer polyline entities
        result = edges_to_path(edges=edges, vertices=lines[unique])
        if index is not None:
            flat = np.asanyarray(index, dtype=np.int64).reshape(-1)
            grouped = assign_face_index(result["entities"], edges, flat)
            result["metadata"] = contour_face_metadata(result["entities"], flat, grouped)
        return result
    else:
        raise ValueError("Lines must be (n,(2|3)) or (n,2,(2|3))")
    return result


def polygon_to_path(polygon):
    """
    Load shapely Polygon objects into a trimesh.path.Path2D object

    Parameters
    -------------
    polygon : shapely.geometry.Polygon
      Input geometry

    Returns
    -----------
    kwargs : dict
      Keyword arguments for Path2D constructor
    """
    # start with a single polyline for the exterior
    entities = []
    # start vertices
    vertices = []

    if hasattr(polygon.boundary, "geoms"):
        boundaries = polygon.boundary.geoms
    else:
        boundaries = [polygon.boundary]

    # append interiors as single Line objects
    current = 0
    for boundary in boundaries:
        entities.append(Line(np.arange(len(boundary.coords)) + current))
        current += len(boundary.coords)
        # append the new vertex array
        vertices.append(np.array(boundary.coords))

    # make sure result arrays are numpy
    kwargs = {
        "entities": entities,
        "vertices": np.vstack(vertices) if len(vertices) > 0 else vertices,
    }

    return kwargs


def linestrings_to_path(multi) -> dict:
    """
    Load shapely LineString objects into arguments to create a Path2D or Path3D.

    Parameters
    -------------
    multi : shapely.geometry.LineString or MultiLineString
      Input 2D or 3D geometry

    Returns
    -------------
    kwargs : dict
      Keyword arguments for Path2D or Path3D constructor
    """
    import shapely

    # append to result as we go
    entities = []
    vertices = []

    if isinstance(multi, shapely.MultiLineString):
        multi = list(multi.geoms)
    else:
        multi = [multi]

    for line in multi:
        # only append geometry with points
        if hasattr(line, "coords"):
            coords = np.array(line.coords)
            if len(coords) < 2:
                continue
            entities.append(Line(np.arange(len(coords)) + len(vertices)))
            vertices.extend(coords)

    kwargs = {"entities": np.array(entities), "vertices": np.array(vertices)}
    return kwargs


def faces_to_path(mesh, face_ids=None, **kwargs):
    """
    Given a mesh and face indices find the outline edges and
    turn them into a Path3D.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Triangulated surface in 3D
    face_ids : (n,) int
      Indexes referencing mesh.faces

    Returns
    ---------
    kwargs : dict
      Kwargs for Path3D constructor
    """
    if face_ids is None:
        edges = mesh.edges_sorted
    else:
        # take advantage of edge ordering to index as single row
        edges = mesh.edges_sorted.reshape((-1, 6))[face_ids].reshape((-1, 2))
    # an edge which occurs onely once is on the boundary
    unique_edges = grouping.group_rows(edges, require_count=1)
    # add edges and vertices to kwargs
    kwargs.update(edges_to_path(edges=edges[unique_edges], vertices=mesh.vertices))

    return kwargs


def edges_to_path(edges: ArrayLike, vertices: ArrayLike, **kwargs) -> dict:
    """
    Given an edge list of indices and associated vertices
    representing lines, generate kwargs for a Path object.

    Parameters
    -----------
    edges : (n, 2) int
      Vertex indices of line segments
    vertices : (m, dimension) float
      Vertex positions where dimension is 2 or 3

    Returns
    ----------
    kwargs : dict
      Kwargs for Path constructor
    """
    # sequence of ordered traversals
    dfs = graph.traversals(edges, mode="dfs")
    # make sure every consecutive index in DFS
    # traversal is an edge in the source edge list
    dfs_connected = graph.fill_traversals(dfs, edges=edges)
    # kwargs for Path constructor
    # turn traversals into Line objects
    lines = [Line(d) for d in dfs_connected]

    kwargs.update({"entities": lines, "vertices": vertices, "process": False})
    return kwargs
