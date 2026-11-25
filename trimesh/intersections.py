"""
intersections.py
------------------

Primarily mesh-plane intersections (slicing).
"""

import numpy as np

from . import geometry, grouping, util
from . import transformations as tf
from . import triangles as tm
from .constants import tol
from .triangles import points_to_barycentric
from .typed import ArrayLike, NDArray


def mesh_plane(
    mesh,
    plane_normal,
    plane_origin,
    return_faces=False,
    local_faces=None,
    cached_dots=None,
):
    """
    Find a the intersections between a mesh and a plane,
    returning a set of line segments on that plane.

    Parameters
    ---------
    mesh : Trimesh object
      Source mesh to slice
    plane_normal : (3,) float
      Normal vector of plane to intersect with mesh
    plane_origin : (3,) float
      Point on plane to intersect with mesh
    return_faces : bool
      If True return face index each line is from
    local_faces : None or (m,) int
      Limit section to just these faces.
    cached_dots : (n, 3) float
      If an external function has stored dot
      products pass them here to avoid recomputing.

    Returns
    ----------
    lines :  (m, 2, 3) float
      List of 3D line segments in space.
    face_index : (m,) int
      Index of mesh.faces for each line
      Only returned if return_faces was True
    """

    def triangle_cases(signs):
        """
        Figure out which faces correspond to which intersection
        case from the signs of the dot product of each vertex.
        Does this by bitbang each row of signs into an 8 bit
        integer.

        code : signs      : intersects
        0    : [-1 -1 -1] : No
        2    : [-1 -1  0] : No
        4    : [-1 -1  1] : Yes; 2 on one side, 1 on the other
        6    : [-1  0  0] : Yes; one edge fully on plane
        8    : [-1  0  1] : Yes; one vertex on plane 2 on different sides
        12   : [-1  1  1] : Yes; 2 on one side, 1 on the other
        14   : [0 0 0]    : No (on plane fully)
        16   : [0 0 1]    : Yes; one edge fully on plane
        20   : [0 1 1]    : No
        28   : [1 1 1]    : No

        Parameters
        ----------
        signs: (n,3) int, all values are -1,0, or 1
          Each row contains the dot product of all three vertices
          in a face with respect to the plane

        Returns
        ---------
        basic : (n,) bool
          Which faces are in the basic intersection case
        one_vertex : (n,) bool
          Which faces are in the one vertex case
        one_edge : (n,) bool
          Which faces are in the one edge case
        """

        signs_sorted = np.sort(signs, axis=1)
        coded = np.zeros(len(signs_sorted), dtype=np.int8) + 14
        for i in range(3):
            coded += signs_sorted[:, i] << 3 - i

        # one edge fully on the plane
        # note that we are only accepting *one* of the on- edge cases,
        # where the other vertex has a positive dot product (16) instead
        # of both on- edge cases ([6, 16])
        # this is so that for regions that are co-planar with the the section plane
        # we don't end up with an invalid boundary
        key = np.zeros(29, dtype=bool)
        key[16] = True
        one_edge = key[coded]

        # one vertex on plane, other two on different sides
        key[:] = False
        key[8] = True
        one_vertex = key[coded]

        # one vertex on one side of the plane, two on the other
        key[:] = False
        key[[4, 12]] = True
        basic = key[coded]

        return basic, one_vertex, one_edge

    def handle_on_vertex(signs, faces, vertices):
        # case where one vertex is on plane
        # and two are on different sides
        vertex_plane = faces[signs == 0]
        edge_thru = faces[signs != 0].reshape((-1, 2))
        point_intersect, valid = plane_lines(
            plane_origin, plane_normal, vertices[edge_thru.T], line_segments=False
        )
        lines = np.column_stack((vertices[vertex_plane[valid]], point_intersect)).reshape(
            (-1, 2, 3)
        )
        return lines

    def handle_on_edge(signs, faces, vertices):
        # case where two vertices are on the plane and one is off
        edges = faces[signs == 0].reshape((-1, 2))
        points = vertices[edges]
        return points

    def handle_basic(signs, faces, vertices):
        # case where one vertex is on one side and two are on the other
        unique_element = grouping.unique_value_in_row(signs, unique=[-1, 1])
        edges = np.column_stack(
            (
                faces[unique_element],
                faces[np.roll(unique_element, 1, axis=1)],
                faces[unique_element],
                faces[np.roll(unique_element, 2, axis=1)],
            )
        ).reshape((-1, 2))
        intersections, valid = plane_lines(
            plane_origin, plane_normal, vertices[edges.T], line_segments=False
        )
        # since the data has been pre- culled, any invalid intersections at all
        # means the culling was done incorrectly and thus things are broken
        assert valid.all()
        return intersections.reshape((-1, 2, 3))

    # check input plane
    plane_normal = np.asanyarray(plane_normal, dtype=np.float64)
    plane_origin = np.asanyarray(plane_origin, dtype=np.float64)
    if plane_origin.shape != (3,) or plane_normal.shape != (3,):
        raise ValueError("Plane origin and normal must be (3,)!")

    if local_faces is None:
        # do a cross section against all faces
        faces = mesh.faces
    else:
        local_faces = np.asanyarray(local_faces, dtype=np.int64)
        # only take the subset of faces if passed
        faces = mesh.faces[local_faces]

    if cached_dots is not None:
        dots = cached_dots
    else:
        # dot product of each vertex with the plane normal indexed by face
        # so for each face the dot product of each vertex is a row
        # shape is the same as mesh.faces (n,3)
        dots = np.dot(mesh.vertices - plane_origin, plane_normal)

    # sign of the dot product is -1, 0, or 1
    # shape is the same as mesh.faces (n,3)
    signs = np.zeros(len(mesh.vertices), dtype=np.int8)
    signs[dots < -tol.merge] = -1
    signs[dots > tol.merge] = 1
    signs = signs[faces]

    # figure out which triangles are in the cross section,
    # and which of the three intersection cases they are in
    cases = triangle_cases(signs)
    # handlers for each case
    handlers = (handle_basic, handle_on_vertex, handle_on_edge)

    # the (m, 2, 3) line segments
    lines = np.vstack(
        [h(signs[c], faces[c], mesh.vertices) for c, h in zip(cases, handlers)]
    )

    if return_faces:
        # everything that hit something
        index = np.hstack([np.nonzero(c)[0] for c in cases])
        assert index.dtype.kind == "i"
        if local_faces is None:
            return lines, index
        # we are considering a subset of faces
        # so we need to take the indexes from original
        return lines, local_faces[index]
    return lines


def mesh_multiplane(mesh, plane_origin, plane_normal, heights):
    """
    A utility function for slicing a mesh by multiple
    parallel planes which caches the dot product operation.

    Parameters
    -------------
    mesh : trimesh.Trimesh
        Geometry to be sliced by planes
    plane_origin : (3,) float
        Point on a plane
    plane_normal : (3,) float
        Normal vector of plane
    heights : (m,) float
      Offset distances from plane to slice at:
      at `height=0` it will be exactly on the passed plane.

    Returns
    --------------
    lines : (m,) sequence of (n, 2, 2) float
        Lines in space for m planes
    to_3D : (m, 4, 4) float
        Transform to move each section back to 3D
    face_index : (m,) sequence of (n,) int
        Indexes of mesh.faces for each segment
    """
    # check input plane
    plane_normal = util.unitize(plane_normal)
    plane_origin = np.asanyarray(plane_origin, dtype=np.float64)
    heights = np.asanyarray(heights, dtype=np.float64)

    # dot product of every vertex with plane
    vertex_dots = np.dot(plane_normal, (mesh.vertices - plane_origin).T)

    # reconstruct transforms for each 2D section
    base_transform = geometry.plane_transform(origin=plane_origin, normal=plane_normal)
    base_transform = np.linalg.inv(base_transform)

    # alter translation Z inside loop
    translation = np.eye(4)

    # store results
    transforms = []
    face_index = []
    segments = []

    # loop through user specified heights
    for height in heights:
        # offset the origin by the height
        new_origin = plane_origin + (plane_normal * height)
        # offset the dot products by height and index by faces
        new_dots = vertex_dots - height
        # run the intersection with the cached dot products
        lines, index = mesh_plane(
            mesh=mesh,
            plane_origin=new_origin,
            plane_normal=plane_normal,
            return_faces=True,
            cached_dots=new_dots,
        )

        # get the transforms to 3D space and back
        translation[2, 3] = height
        to_3D = np.dot(base_transform, translation)
        to_2D = np.linalg.inv(to_3D)
        transforms.append(to_3D)

        # transform points to 2D frame
        lines_2D = tf.transform_points(lines.reshape((-1, 3)), to_2D)

        # if we didn't screw up the transform all
        # of the Z values should be zero
        # assert np.allclose(lines_2D[:, 2], 0.0)

        # reshape back in to lines and discard Z
        lines_2D = lines_2D[:, :2].reshape((-1, 2, 2))
        # store (n, 2, 2) float lines
        segments.append(lines_2D)
        # store (n,) int indexes of mesh.faces
        face_index.append(index)

    # (n, 4, 4) transforms from 2D to 3D
    transforms = np.array(transforms, dtype=np.float64)

    return segments, transforms, face_index


def plane_lines(plane_origin, plane_normal, endpoints, line_segments=True):
    """
    Calculate plane-line intersections

    Parameters
    ---------
    plane_origin : (3,) float
        Point on plane
    plane_normal : (3,) float
        Plane normal vector
    endpoints : (2, n, 3) float
        Points defining lines to be tested
    line_segments : bool
        If True, only returns intersections as valid if
        vertices from endpoints are on different sides
        of the plane.

    Returns
    ---------
    intersections : (m, 3) float
        Cartesian intersection points
    valid : (n, 3) bool
        Indicate whether a valid intersection exists
        for each input line segment
    """
    endpoints = np.asanyarray(endpoints)
    plane_origin = np.asanyarray(plane_origin).reshape(3)
    line_dir = util.unitize(endpoints[1] - endpoints[0])
    plane_normal = util.unitize(np.asanyarray(plane_normal).reshape(3))

    t = np.dot(plane_normal, (plane_origin - endpoints[0]).T)
    b = np.dot(plane_normal, line_dir.T)

    # If the plane normal and line direction are perpendicular, it means
    # the vector is 'on plane', and there isn't a valid intersection.
    # We discard on-plane vectors by checking that the dot product is nonzero
    valid = np.abs(b) > tol.zero
    if line_segments:
        test = np.dot(plane_normal, np.transpose(plane_origin - endpoints[1]))
        different_sides = np.sign(t) != np.sign(test)
        nonzero = np.logical_or(np.abs(t) > tol.zero, np.abs(test) > tol.zero)
        valid = np.logical_and(valid, different_sides)
        valid = np.logical_and(valid, nonzero)

    d = np.divide(t[valid], b[valid])
    intersection = endpoints[0][valid]
    intersection = intersection + np.reshape(d, (-1, 1)) * line_dir[valid]

    return intersection, valid


def planes_lines(
    plane_origins,
    plane_normals,
    line_origins,
    line_directions,
    return_distance=False,
    return_denom=False,
):
    """
    Given one line per plane find the intersection points.

    Parameters
    -----------
    plane_origins : (n,3) float
        Point on each plane
    plane_normals : (n,3) float
        Normal vector of each plane
    line_origins : (n,3) float
        Point at origin of each line
    line_directions : (n,3) float
        Direction vector of each line
    return_distance : bool
      Return distance from origin to point also
    return_denom : bool
      Return denominator, so you can check for small values

    Returns
    ----------
    on_plane : (n,3) float
      Points on specified planes
    valid : (n,) bool
      Did plane intersect line or not
    distance : (n,) float
      [OPTIONAL] Distance from point
    denom : (n,) float
      [OPTIONAL] Denominator
    """

    # check input types
    plane_origins = np.asanyarray(plane_origins, dtype=np.float64)
    plane_normals = np.asanyarray(plane_normals, dtype=np.float64)
    line_origins = np.asanyarray(line_origins, dtype=np.float64)
    line_directions = np.asanyarray(line_directions, dtype=np.float64)

    # vector from line to plane
    origin_vectors = plane_origins - line_origins

    projection_ori = util.diagonal_dot(origin_vectors, plane_normals)
    projection_dir = util.diagonal_dot(line_directions, plane_normals)

    valid = np.abs(projection_dir) > 1e-5

    distance = np.divide(projection_ori[valid], projection_dir[valid])

    on_plane = line_directions[valid] * distance.reshape((-1, 1))
    on_plane += line_origins[valid]

    result = [on_plane, valid]

    if return_distance:
        result.append(distance)
    if return_denom:
        result.append(projection_dir)

    return result


def slice_faces_plane(
    vertices,
    faces,
    plane_normal,
    plane_origin,
    uv=None,
    face_index=None,
    cached_dots=None,
):
    """
    Slice a mesh (given as a set of faces and vertices) with a plane, returning a
    new mesh (again as a set of faces and vertices) that is the
    portion of the original mesh to the positive normal side of the plane.

    Parameters
    ---------
    vertices : (n, 3) float
        Vertices of source mesh to slice
    faces : (n, 3) int
        Faces of source mesh to slice
    plane_normal : (3,) float
        Normal vector of plane to intersect with mesh
    plane_origin :  (3,) float
        Point on plane to intersect with mesh
    uv : (n, 2) float, optional
        UV coordinates of source mesh to slice
    face_index : ((m,) int)
        Indexes of faces to slice. When no mask is provided, the
        default is to slice all faces.
    cached_dots : (n, 3) float
        If an external function has stored dot
        products pass them here to avoid recomputing

    Returns
    ----------
    new_vertices : (n, 3) float
        Vertices of sliced mesh
    new_faces : (n, 3) int
        Faces of sliced mesh
    new_uv : (n, 2) int or None
        UV coordinates of sliced mesh
    """

    if len(vertices) == 0:
        return vertices, faces, uv

    have_uv = uv is not None

    # Construct a mask for the faces to slice.
    if face_index is not None:
        faces = faces[face_index]

    if cached_dots is not None:
        dots = cached_dots
    else:
        # dot product of each vertex with the plane normal indexed by face
        # so for each face the dot product of each vertex is a row
        # shape is the same as faces (n,3)
        dots = np.dot(vertices - plane_origin, plane_normal)

    # Find vertex orientations w.r.t. faces for all triangles:
    #  -1 -> vertex "inside" plane (positive normal direction)
    #   0 -> vertex on plane
    #   1 -> vertex "outside" plane (negative normal direction)
    signs = np.zeros(len(vertices), dtype=np.int8)
    signs[dots < -tol.merge] = 1
    signs[dots > tol.merge] = -1
    signs = signs[faces]

    # Find all triangles that intersect this plane
    # onedge <- indices of all triangles intersecting the plane
    # inside <- indices of all triangles "inside" the plane (positive normal)
    signs_sum = signs.sum(axis=1, dtype=np.int8)
    signs_asum = np.abs(signs).sum(axis=1, dtype=np.int8)

    # Cases:
    # (0,0,0),  (-1,0,0),  (-1,-1,0), (-1,-1,-1) <- inside
    # (1,0,0),  (1,1,0),   (1,1,1)               <- outside
    # (1,0,-1), (1,-1,-1), (1,1,-1)              <- onedge
    onedge = np.logical_and(signs_asum >= 2, np.abs(signs_sum) <= 1)

    inside = signs_sum == -signs_asum

    # for any faces that lie exactly on-the-plane
    # we want to only include them if their normal
    # is backwards from the slicing normal
    on_plane = signs_asum == 0
    if on_plane.any():
        # compute the normals and whether
        # face is degenerate here
        check, valid = tm.normals(vertices[faces[on_plane]])
        # only include faces back from normal
        dot_check = np.dot(check, plane_normal)
        # exclude any degenerate faces from the result
        inside[on_plane] = valid
        # exclude the degenerate face from our mask
        on_plane[on_plane] = valid
        # apply results for this subset
        inside[on_plane] = dot_check < 0.0

    # Automatically include all faces that are "inside"
    new_faces = faces[inside]

    # Separate faces on the edge into two cases: those which will become
    # quads (two vertices inside plane) and those which will become triangles
    # (one vertex inside plane)
    triangles = vertices[faces]
    cut_triangles = triangles[onedge]
    cut_faces_quad = faces[np.logical_and(onedge, signs_sum < 0)]
    cut_faces_tri = faces[np.logical_and(onedge, signs_sum >= 0)]
    cut_signs_quad = signs[np.logical_and(onedge, signs_sum < 0)]
    cut_signs_tri = signs[np.logical_and(onedge, signs_sum >= 0)]

    # If no faces to cut, the surface is not in contact with this plane.
    # Thus, return a mesh with only the inside faces
    if len(cut_faces_quad) + len(cut_faces_tri) == 0:
        if len(new_faces) == 0:
            # if no new faces at all return empty arrays
            empty = (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.int64),
                np.zeros((0, 2), dtype=np.float64) if have_uv else None,
            )
            return empty

        # find the unique indices in the new faces
        # using an integer-only unique function
        unique, inverse = grouping.unique_bincount(
            new_faces.reshape(-1), minlength=len(vertices), return_inverse=True
        )

        # use the unique indices for our final vertices and faces
        final_vert = vertices[unique]
        final_face = inverse.reshape((-1, 3))
        final_uv = uv[unique] if have_uv else None

        return final_vert, final_face, final_uv

    # Extract the intersections of each triangle's edges with the plane
    o = cut_triangles  # origins
    d = np.roll(o, -1, axis=1) - o  # directions
    num = (plane_origin - o).dot(plane_normal)  # compute num/denom
    denom = np.dot(d, plane_normal)
    denom[denom == 0.0] = 1e-12  # prevent division by zero
    dist = np.divide(num, denom)
    # intersection points for each segment
    int_points = dist[:, :, None] * d + o

    # Initialize the array of new vertices with the current vertices
    new_vertices = vertices
    new_quad_vertices = np.zeros((0, 3))
    new_tri_vertices = np.zeros((0, 3))

    # Handle the case where a new quad is formed by the intersection
    # First, extract the intersection points belonging to a new quad
    quad_int_points = int_points[(signs_sum < 0)[onedge], :, :]
    num_quads = len(quad_int_points)
    if num_quads > 0:
        # Extract the vertex on the outside of the plane, then get the vertices
        # (in CCW order of the inside vertices)
        quad_int_inds = np.where(cut_signs_quad == 1)[1]
        quad_int_verts = cut_faces_quad[
            np.stack((range(num_quads), range(num_quads)), axis=1),
            np.stack(((quad_int_inds + 1) % 3, (quad_int_inds + 2) % 3), axis=1),
        ]

        # Fill out new quad faces with the intersection points as vertices
        new_quad_faces = np.append(
            quad_int_verts,
            np.arange(len(new_vertices), len(new_vertices) + 2 * num_quads).reshape(
                num_quads, 2
            ),
            axis=1,
        )

        # Extract correct intersection points from int_points and order them in
        # the same way as they were added to faces
        new_quad_vertices = quad_int_points[
            np.stack((range(num_quads), range(num_quads)), axis=1),
            np.stack((((quad_int_inds + 2) % 3).T, quad_int_inds.T), axis=1),
            :,
        ].reshape(2 * num_quads, 3)

        # Add new vertices to existing vertices, triangulate quads, and add the
        # resulting triangles to the new faces
        new_vertices = np.append(new_vertices, new_quad_vertices, axis=0)
        new_tri_faces_from_quads = geometry.triangulate_quads(new_quad_faces)
        new_faces = np.append(new_faces, new_tri_faces_from_quads, axis=0)

    # Handle the case where a new triangle is formed by the intersection
    # First, extract the intersection points belonging to a new triangle
    tri_int_points = int_points[(signs_sum >= 0)[onedge], :, :]
    num_tris = len(tri_int_points)
    if num_tris > 0:
        # Extract the single vertex for each triangle inside the plane and get the
        # inside vertices (CCW order)
        tri_int_inds = np.where(cut_signs_tri == -1)[1]
        tri_int_verts = cut_faces_tri[range(num_tris), tri_int_inds].reshape(num_tris, 1)

        # Fill out new triangles with the intersection points as vertices
        new_tri_faces = np.append(
            tri_int_verts,
            np.arange(len(new_vertices), len(new_vertices) + 2 * num_tris).reshape(
                num_tris, 2
            ),
            axis=1,
        )

        # Extract correct intersection points and order them in the same way as
        # the vertices were added to the faces
        new_tri_vertices = tri_int_points[
            np.stack((range(num_tris), range(num_tris)), axis=1),
            np.stack((tri_int_inds.T, ((tri_int_inds + 2) % 3).T), axis=1),
            :,
        ].reshape(2 * num_tris, 3)

        # Append new vertices and new faces
        new_vertices = np.append(new_vertices, new_tri_vertices, axis=0)
        new_faces = np.append(new_faces, new_tri_faces, axis=0)

    # find the unique indices in the new faces
    # using an integer-only unique function
    unique, inverse = grouping.unique_bincount(
        new_faces.reshape(-1), minlength=len(new_vertices), return_inverse=True
    )

    # use the unique indexes for our final vertex and faces
    final_vert = new_vertices[unique]
    final_face = inverse.reshape((-1, 3))

    final_uv = None
    if have_uv:
        # Generate barycentric coordinates for intersection vertices
        quad_barycentrics = points_to_barycentric(
            np.repeat(vertices[cut_faces_quad], 2, axis=0), new_quad_vertices
        )
        tri_barycentrics = points_to_barycentric(
            np.repeat(vertices[cut_faces_tri], 2, axis=0), new_tri_vertices
        )
        all_barycentrics = np.concatenate([quad_barycentrics, tri_barycentrics])

        # Interpolate UVs
        cut_uv = np.concatenate([uv[cut_faces_quad], uv[cut_faces_tri]])
        new_uv = np.einsum("ijk,ij->ik", np.repeat(cut_uv, 2, axis=0), all_barycentrics)
        final_uv = np.concatenate([uv, new_uv])[unique]

    return final_vert, final_face, final_uv


def slice_mesh_plane(
    mesh,
    plane_normal,
    plane_origin,
    face_index=None,
    cap=False,
    engine=None,
    **kwargs,
):
    """
    Slice a mesh with a plane returning a new mesh that is the
    portion of the original mesh to the positive normal side
    of the plane.

    Parameters
    ---------
    mesh : Trimesh object
      Source mesh to slice
    plane_normal : (3,) float
      Normal vector of plane to intersect with mesh
    plane_origin :  (3,) float
      Point on plane to intersect with mesh
    cap : bool
      If True, cap the result with a triangulated polygon
    face_index : ((m,) int)
      Indexes of mesh.faces to slice. When no mask is provided, the
      default is to slice all faces.
    cached_dots : (n, 3) float
      If an external function has stored dot
      products pass them here to avoid recomputing
    engine : None or str
      Triangulation engine passed to `triangulate_polygon`
    kwargs : dict
      Passed to the newly created sliced mesh

    Returns
    ----------
    new_mesh : Trimesh object
      Sliced mesh
    """
    # check input for none
    if mesh is None:
        return None

    # avoid circular import
    from scipy.spatial import cKDTree

    from .base import Trimesh
    from .creation import triangulate_polygon
    from .path import polygons
    from .visual import TextureVisuals

    # check input plane
    plane_normal = np.asanyarray(plane_normal, dtype=np.float64)
    plane_origin = np.asanyarray(plane_origin, dtype=np.float64)

    # check to make sure origins and normals have acceptable shape
    shape_ok = (
        (plane_origin.shape == (3,) or util.is_shape(plane_origin, (-1, 3)))
        and (plane_normal.shape == (3,) or util.is_shape(plane_normal, (-1, 3)))
        and plane_origin.shape == plane_normal.shape
    )
    if not shape_ok:
        raise ValueError("plane origins and normals must be (n, 3)!")

    # start with copy of original mesh, faces, and vertices
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()

    # We copy the UV coordinates if available
    has_uv = (
        hasattr(mesh.visual, "uv") and np.shape(mesh.visual.uv) == (len(mesh.vertices), 2)
    ) and not cap
    uv = mesh.visual.uv.copy() if has_uv else None

    if "process" not in kwargs:
        kwargs["process"] = False

    # slice away specified planes
    for origin, normal in zip(
        plane_origin.reshape((-1, 3)), plane_normal.reshape((-1, 3))
    ):
        # save the new vertices and faces
        vertices, faces, uv = slice_faces_plane(
            vertices=vertices,
            faces=faces,
            uv=uv,
            plane_normal=normal,
            plane_origin=origin,
            face_index=face_index,
        )
        # check if cap arg specified
        if cap:
            if face_index:
                # This hasn't been implemented yet.
                raise NotImplementedError("face_index and cap can't be used together")

            # start by deduplicating vertices again
            unique, inverse = grouping.unique_rows(vertices)
            vertices = vertices[unique]
            # will collect additional faces
            f = inverse[faces]
            # remove degenerate faces by checking to make sure
            # that each face has three unique indices
            f = f[(f[:, :1] != f[:, 1:]).all(axis=1)]
            # transform to the cap plane
            to_2D = geometry.plane_transform(origin=origin, normal=-normal)
            to_3D = np.linalg.inv(to_2D)

            vertices_2D = tf.transform_points(vertices, to_2D)
            edges = geometry.faces_to_edges(f)
            edges.sort(axis=1)

            on_plane = np.abs(vertices_2D[:, 2]) < 1e-8
            edges = edges[on_plane[edges].all(axis=1)]
            edges = edges[edges[:, 0] != edges[:, 1]]

            unique_edge = grouping.group_rows(edges, require_count=1)
            if len(unique) < 3:
                continue

            tree = cKDTree(vertices)
            # collect new faces
            faces = [f]
            for p in polygons.edges_to_polygons(edges[unique_edge], vertices_2D[:, :2]):
                # triangulate cap and raise an error if any new vertices were inserted
                vn, fn = triangulate_polygon(p, engine=engine, force_vertices=True)
                # collect the original index for the new vertices
                vn3 = tf.transform_points(util.stack_3D(vn), to_3D)
                distance, vid = tree.query(vn3)
                if distance.max() > 1e-8:
                    util.log.debug("triangulate may have inserted vertex!")
                # triangulation should not have inserted vertices
                nf = vid[fn]
                # hmm but it may have returned faces that are now degenerate
                nf_ok = (nf[:, 1:] != nf[:, :1]).all(axis=1) & (nf[:, 1] != nf[:, 2])
                faces.append(nf[nf_ok])

            faces = np.vstack(faces)

    visual = (
        TextureVisuals(uv=uv, material=mesh.visual.material.copy()) if has_uv else None
    )

    # return the sliced mesh
    return Trimesh(vertices=vertices, faces=faces, visual=visual, **kwargs)


def _segment_candidate_faces(mesh, a, b):
    """
    Find the indexes of faces whose axis aligned bounds intersect
    the bounding box of a single line segment.

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Mesh to query against.
    a : (3,) float
      Start point of the segment.
    b : (3,) float
      End point of the segment.

    Returns
    --------
    face_index : (m,) int64
      Candidate face indexes.
    """
    box = np.concatenate((np.minimum(a, b), np.maximum(a, b)))
    # pad by the merge tolerance so tangency cases are not culled
    box[:3] -= tol.merge
    box[3:] += tol.merge
    return np.fromiter(mesh.triangles_tree.intersection(box), dtype=np.int64)


def _segment_triangle_hits(triangles, face_normals, a, b):
    """
    Intersect a single finite line segment against a group of
    triangles, returning transversal hits and coplanar overlap.

    Parameters
    -----------
    triangles : (n, 3, 3) float
      Candidate triangles in space.
    face_normals : (n, 3) float
      Unit normals of the candidate triangles.
    a : (3,) float
      Segment start point.
    b : (3,) float
      Segment end point.

    Returns
    --------
    hit_t : (h,) float
      Parameter values in ``[0, 1]`` of transversal intersections
      of the segment with triangle interiors, edges or vertices.
      Multiple faces sharing an edge/vertex produce repeated values.
    coplanar_t : (k, 2) float
      ``[t_min, t_max]`` intervals of the segment lying in the plane
      and inside each coplanar triangle.
    """
    direction = b - a
    length = np.linalg.norm(direction)

    # signed distance of the segment endpoints to each triangle plane
    rel = a - triangles[:, 0, :]
    d0 = util.diagonal_dot(rel, face_normals)
    d1 = util.diagonal_dot(b - triangles[:, 0, :], face_normals)

    eps = tol.merge
    coplanar = np.abs(d0) <= eps
    # a segment is parallel to a plane only when both endpoints are
    coplanar &= np.abs(d1) <= eps
    crossing = ~coplanar

    hit_t = np.zeros(0, dtype=np.float64)
    coplanar_t = np.zeros((0, 2), dtype=np.float64)

    if crossing.any():
        idx = np.flatnonzero(crossing)
        # linear interpolation of the signed plane distance
        denom = d1[idx] - d0[idx]
        t = -d0[idx] / denom
        inside = (t >= -tol.zero) & (t <= 1.0 + tol.zero)
        idx = idx[inside]
        t = np.clip(t[inside], 0.0, 1.0)
        if len(idx) > 0:
            points = a + t.reshape((-1, 1)) * direction
            bary = points_to_barycentric(triangles[idx], points)
            on_tri = (bary >= -tol.zero).all(axis=1) & (bary <= 1.0 + tol.zero).all(
                axis=1
            )
            hit_t = t[on_tri]

    if coplanar.any():
        idx = np.flatnonzero(coplanar)
        # barycentric coordinates are affine along the segment so the
        # interval inside each triangle is the intersection of the three
        # half-plane constraints on t
        b0 = points_to_barycentric(triangles[idx], np.tile(a, (len(idx), 1)))
        b1 = points_to_barycentric(triangles[idx], np.tile(b, (len(idx), 1)))
        db = b1 - b0
        lo = np.zeros(len(idx))
        hi = np.ones(len(idx))
        valid = np.ones(len(idx), dtype=bool)
        for k in range(3):
            flat = np.abs(db[:, k]) <= tol.zero
            # constant coordinate must already satisfy the constraint
            fails = flat & ((b0[:, k] < -tol.zero) | (b0[:, k] > 1.0 + tol.zero))
            valid &= ~fails
            slope = np.where(flat, 1.0, db[:, k])
            root = -b0[:, k] / slope
            rising = (~flat) & (db[:, k] > 0.0)
            falling = (~flat) & (db[:, k] < 0.0)
            lo = np.maximum(lo, np.where(rising, root, 0.0))
            hi = np.minimum(hi, np.where(falling, root, 1.0))
        valid &= hi >= lo - tol.zero
        if valid.any():
            lo = np.clip(lo[valid], 0.0, 1.0)
            hi = np.clip(hi[valid], 0.0, 1.0)
            keep = (hi - lo) * length > tol.merge
            coplanar_t = np.column_stack((lo[keep], hi[keep]))

    return hit_t, coplanar_t


def _merge_t_intervals(intervals, tol_abs):
    """
    Merge overlapping ``[t_min, t_max]`` intervals on a single segment.

    Parameters
    -----------
    intervals : (n, 2) float
      Parameter intervals, unordered.
    tol_abs : float
      Gaps smaller than this are merged.

    Returns
    --------
    merged : (m, 2) float
      Sorted, disjoint intervals.
    """
    if len(intervals) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    order = np.argsort(intervals[:, 0])
    intervals = intervals[order]
    merged = [intervals[0].copy()]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1] + tol_abs:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append(np.array([lo, hi]))
    return np.array(merged, dtype=np.float64)


def _unique_t(values, tol_abs):
    """
    Sort and deduplicate parameter values that agree within a tolerance.

    Parameters
    -----------
    values : (n,) float
      Parameter values on a single segment.
    tol_abs : float
      Values closer than this are treated as the same event.

    Returns
    --------
    unique : (m,) float
      Sorted, deduplicated values, each clipped to ``[0, 1]``.
    """
    if len(values) == 0:
        return np.zeros(0, dtype=np.float64)
    values = np.sort(np.clip(np.asanyarray(values, dtype=np.float64), 0.0, 1.0))
    unique = [values[0]]
    for v in values[1:]:
        if v - unique[-1] > tol_abs:
            unique.append(v)
        else:
            # keep the value closest to 0 or 1 when an event lands on
            # an endpoint, otherwise retain the existing estimate
            if abs(v - 0.0) < abs(unique[-1] - 0.0) or abs(v - 1.0) < abs(
                unique[-1] - 1.0
            ):
                unique[-1] = v
    return np.array(unique, dtype=np.float64)


def clip_polyline(mesh, polyline: ArrayLike) -> list[NDArray[np.float64]]:
    """
    Clip a finite polyline against a closed watertight mesh, returning
    the portions of the polyline that lie inside the mesh.

    Unlike filtering sampled points with :meth:`Trimesh.contains`, this
    inserts exact triangle intersection points at every entry and exit,
    so segments crossing the surface between two samples are clipped
    precisely. Tangencies (the line only touches an edge or vertex) do
    not create spurious inside runs, multiple entries and exits are all
    returned, and segments lying entirely inside are kept whole.

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Closed, watertight mesh defining the inside region. Points on the
      surface are treated as inside (a closed set).
    polyline : (n, 3) float
      Ordered vertices of a single finite polyline. Zero-length segments
      (consecutive duplicate points) are skipped.

    Returns
    --------
    inside : list of (m, 3) float
      The inside portions of the polyline as ordered point lists. Each
      array is one maximal inside run; consecutive input segments whose
      junction stays inside are joined into a single run, and every run
      starts and ends either at an exact surface intersection point or
      at an original polyline vertex. An empty list is returned when
      nothing is inside.

    Raises
    -------
    ValueError
      If ``mesh`` is not watertight (open or non-manifold meshes do not
      define an inside/outside classification) or the polyline is not
      shaped ``(n, 3)``.
    """
    polyline = np.asanyarray(polyline, dtype=np.float64)
    if not util.is_shape(polyline, (-1, 3)):
        raise ValueError("polyline must be (n, 3) points")
    if not bool(getattr(mesh, "is_watertight", False)):
        raise ValueError(
            "mesh must be watertight to clip a polyline: open or non-manifold meshes do not define an inside region"
        )
    if len(polyline) < 2:
        return []

    triangles = mesh.triangles
    normals = mesh.face_normals

    # one inside run is built up across consecutive segments
    runs: list[NDArray[np.float64]] = []
    current: list[np.ndarray] = []
    # whether the previous segment was inside at its end vertex,
    # i.e. at the junction with the current segment
    prev_end_inside = False

    for seg_index in range(len(polyline) - 1):
        a = polyline[seg_index]
        b = polyline[seg_index + 1]
        direction = b - a
        length = np.linalg.norm(direction)

        if length <= tol.merge:
            # zero-length segment: a single point. it joins a run when it
            # is inside; a point outside closes any run in progress
            point_inside = bool(mesh.contains(a.reshape((1, 3)))[0])
            if point_inside:
                current.append(a.copy())
            elif len(current) > 0:
                runs.append(np.array(current))
                current = []
            prev_end_inside = point_inside
            continue

        # tolerance in parameter space, scaled to the segment length
        tol_t = tol.merge / length

        face_index = _segment_candidate_faces(mesh, a, b)
        if len(face_index) == 0:
            hit_t = np.zeros(0, dtype=np.float64)
            coplanar_t = np.zeros((0, 2), dtype=np.float64)
        else:
            hit_t, coplanar_t = _segment_triangle_hits(
                triangles[face_index], normals[face_index], a, b
            )

        # portions lying on coplanar faces are inside by convention
        surface = _merge_t_intervals(coplanar_t, tol_t)

        # partition the segment at every transversal surface event
        events = _unique_t(hit_t, tol_t)
        cuts = np.concatenate(([0.0], events, [1.0]))
        cuts = _unique_t(cuts, tol_t)

        # classify every open interval from its midpoint in a single
        # batched contains query; tangencies partition nothing because
        # the intervals on both sides receive the same classification
        if len(cuts) >= 2:
            mids = 0.5 * (cuts[:-1] + cuts[1:])
            midpoints = a + mids.reshape((-1, 1)) * direction
            inside_mid = mesh.contains(midpoints)
        else:
            inside_mid = np.zeros(0, dtype=bool)

        # force intervals covered by coplanar faces to be inside
        for lo, hi in surface:
            for j in range(len(cuts) - 1):
                mid = 0.5 * (cuts[j] + cuts[j + 1])
                if lo - tol_t <= mid <= hi + tol_t:
                    inside_mid[j] = True

        # tangencies need no special handling: the intervals on both
        # sides of a tangent event get the same midpoint classification,
        # and exact event points are attached to inside runs below

        # junction state is read from the interval adjacent to each
        # endpoint: an entry event exactly at t=0 still leaves the first
        # interval outside, so the junction itself is not inside
        start_inside = len(inside_mid) > 0 and bool(inside_mid[0])
        end_inside = len(inside_mid) > 0 and bool(inside_mid[-1])

        # collect each maximal inside piece of this segment separately:
        # a segment crossing the surface multiple times yields several
        pieces: list[list[np.ndarray]] = []
        current_piece: list[np.ndarray] = []
        for j in range(len(cuts) - 1):
            if not inside_mid[j]:
                if len(current_piece) > 0:
                    pieces.append(current_piece)
                    current_piece = []
                continue
            t0, t1 = cuts[j], cuts[j + 1]
            p0 = a if t0 <= tol_t else a + t0 * direction
            p1 = b if t1 >= 1.0 - tol_t else a + t1 * direction
            if len(current_piece) == 0:
                current_piece.append(p0)
            current_piece.append(p1)
        if len(current_piece) > 0:
            pieces.append(current_piece)

        if len(pieces) == 0:
            # segment is fully outside: close any run in progress
            if len(current) > 0:
                runs.append(np.array(current))
                current = []
            prev_end_inside = False
            continue

        for piece_index, piece in enumerate(pieces):
            piece_starts_inside = piece_index == 0 and start_inside
            if len(current) > 0 and prev_end_inside and piece_starts_inside:
                # both adjacent segments are inside at their shared
                # vertex: drop the duplicate junction and extend the run
                current.extend(piece[1:])
            else:
                # junction outside (or no run in progress): start fresh
                if len(current) > 0:
                    runs.append(np.array(current))
                current = list(piece)
            # only the first piece can touch the incoming junction; any
            # later piece starts after an outside gap so its incoming end
            # is an interior surface event, not a shared vertex
            prev_end_inside = False
        # the final piece reaches the segment end vertex iff the last
        # interval was inside
        prev_end_inside = end_inside

    if len(current) > 0:
        runs.append(np.array(current))

    # an isolated tangency collapses to a single point: discard it, as
    # a point-only contact carries no inside portion of the polyline
    return [run for run in runs if len(run) >= 2]
