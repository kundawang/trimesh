"""
repair.py
-------------

Fill holes and fix winding and normals of meshes.
"""

import numpy as np

from . import bounds, graph, triangles
from .constants import log
from .geometry import faces_to_edges, triangulate_quads
from .graph import connected_components
from .grouping import group_rows, hashable_rows

try:
    import networkx as nx
except BaseException as E:
    # create a dummy module which will raise the ImportError
    # or other exception only when someone tries to use networkx
    from .exceptions import ExceptionWrapper

    nx = ExceptionWrapper(E)

try:
    from .path.exchange.misc import faces_to_path
except BaseException as E:
    from .exceptions import ExceptionWrapper

    faces_to_path = ExceptionWrapper(E)


def fix_winding(mesh):
    """
    Traverse and change mesh faces in-place to make sure
    winding is correct with edges on adjacent faces in
    opposite directions.

    Parameters
    -------------
    mesh : Trimesh
      Source geometry to alter in-place.
    """
    # anything we would fix is already done
    if mesh.is_winding_consistent:
        return

    graph_all = nx.from_edgelist(mesh.face_adjacency)
    flipped = 0

    faces = mesh.faces.view(np.ndarray).copy()

    # we are going to traverse the graph using BFS
    # start a traversal for every connected component
    for components in nx.connected_components(graph_all):
        # get a subgraph for this component
        g = graph_all.subgraph(components)
        # get the first node in the graph in a way that works on nx's
        # new API and their old API
        start = next(iter(g.nodes()))

        # we traverse every pair of faces in the graph
        # we modify mesh.faces and mesh.face_normals in place
        for face_pair in nx.bfs_edges(g, start):
            # for each pair of faces, we convert them into edges,
            # find the edge that both faces share and then see if edges
            # are reversed in order as you would expect
            # (2, ) int
            face_pair = np.ravel(face_pair)
            # (2, 3) int
            pair = faces[face_pair]
            # (6, 2) int
            edges = faces_to_edges(pair)
            overlap = group_rows(np.sort(edges, axis=1), require_count=2)
            if len(overlap) == 0:
                # only happens on non-watertight meshes
                continue
            edge_pair = edges[overlap[0]]
            if edge_pair[0][0] == edge_pair[1][0]:
                # if the edges aren't reversed, invert the order of one face
                flipped += 1
                faces[face_pair[1]] = faces[face_pair[1]][::-1]

    if flipped > 0:
        mesh.faces = faces

    log.debug("flipped %d/%d edges", flipped, len(mesh.faces) * 3)


def _shell_interior_point(shell):
    """
    Find a point strictly inside a closed, winding-consistent
    shell regardless of whether its faces are currently wound
    inwards or outwards.

    For a consistently wound closed surface one of the two
    points offset from a face centroid along (and opposite to)
    the face normal lies inside the shell, so both candidates
    are tested against the shell itself.

    Parameters
    -------------
    shell : trimesh.Trimesh
      A watertight, winding-consistent single shell.

    Returns
    -------------
    point : (3,) float or None
      A point inside the shell, or `None` if none of the
      candidate points could be confirmed to be inside.
    """
    # try the largest faces first as their offset distance is
    # less likely to be numerically problematic
    order = np.argsort(shell.area_faces)[::-1]
    step = max(shell.scale, np.finfo(np.float64).eps) * 1e-3
    for index in order:
        centroid = shell.triangles[index].mean(axis=0)
        normal = shell.face_normals[index]
        candidates = np.vstack(
            (centroid - normal * step, centroid + normal * step)
        )
        inside = shell.contains(candidates)
        if inside.any():
            return candidates[np.argmax(inside)]
    return None


def fix_inversion(mesh, multibody: bool = False):
    """
    Check to see if a mesh has normals pointing "out."

    For a single watertight shell a negative signed volume means
    the shell is inverted and every face is flipped.

    For a mesh of multiple disconnected shells the shells may be
    nested: a reversed inner shell is a cavity rather than an
    inverted body. Each shell is oriented according to its nesting
    depth instead of its raw signed volume:

    * even depth (outer shells, standalone bodies): normals out,
      signed volume positive
    * odd depth (cavities, islands inside cavities, ...): normals
      in, signed volume negative, alternating with every layer

    Shells that are not watertight, or whose spatial relationship
    is ambiguous (intersecting shells, or shells for which no
    interior point can be established) are left untouched. Such
    cases are reported through the logger rather than guessed at.

    Parameters
    -------------
    mesh : trimesh.Trimesh
      Mesh to fix in-place.
    multibody : bool
      If True orient every disconnected shell taking their
      spatial nesting into account, otherwise only consider the
      mesh as a single body.
    """

    # we are not considering multiple bodies so we only need to
    # do anything if the mesh is watertight and the volume is negative
    if not multibody:
        if mesh.is_watertight and mesh.volume < 0.0:
            # this will make things worse for non-watertight meshes
            mesh.invert()
        return

    # get groups of connected faces
    groups = connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
    )

    # mask of faces to flip
    flip = np.zeros(len(mesh.faces), dtype=bool)

    # save these to avoid thrashing cache in a loop
    tri = mesh.triangles
    cross = mesh.triangles_cross
    # get the edges indexable as faces
    face_edges = mesh.edges.reshape((-1, 6))

    # watertight shells as (face indices, submesh, signed volume)
    shells = []
    # loop through indexes of `mesh.faces`
    for face_index in groups:
        # we need watertight bodies to do anything so immediately
        # skip anything smaller than 4 faces (tetrahedron)
        if len(face_index) < 4:
            continue
        # check the watertightness and winding of this group
        is_tight, is_wound = graph.is_watertight(
            face_edges[face_index].reshape((-1, 2))
        )

        # if we're not completely correct skip it as flipping
        # faces will make it worse
        if not (is_tight and is_wound):
            log.warning(
                "skipping non-watertight shell with %d faces", len(face_index)
            )
            continue

        # check the volume for just this body
        volume = triangles.mass_properties(
            tri[face_index], crosses=cross[face_index], skip_inertia=True
        ).volume

        shells.append([np.asanyarray(face_index), None, float(volume)])

    if len(shells) <= 1:
        # a single shell uses the plain signed-volume rule which
        # preserves the historical single-body behavior
        if len(shells) == 1 and shells[0][2] < 0.0:
            flip[shells[0][0]] = True
    else:
        # build an independent submesh per shell so ray tests only
        # intersect the geometry of that shell
        for entry in shells:
            entry[1] = mesh.submesh([entry[0]], repair=False, append=True)

        # establish one known interior point per shell
        interior = []
        for entry in shells:
            point = _shell_interior_point(entry[1])
            if point is None:
                log.warning("could not determine an interior point for a shell; leaving nested shells unmodified")
                # bail out entirely rather than orienting on a guess
                interior = None
                break
            interior.append(point)

        if interior is not None:
            # containment[i, j] is True when shell j contains the
            # interior point of shell i, i.e. shell i is inside j
            n = len(shells)
            containment = np.eye(n, dtype=bool)
            for i, point in enumerate(interior):
                for j, entry in enumerate(shells):
                    if i == j:
                        continue
                    # cheap axis-aligned bounding box rejection
                    if not bounds.contains(
                        entry[1].bounds, point.reshape((1, 3))
                    )[0]:
                        continue
                    containment[i, j] = entry[1].contains(
                        point.reshape((1, 3))
                    )[0]

            # every shell pair must form either a clean parent/child
            # nesting (containment in exactly one direction) or be
            # completely disjoint; anything else means the shells
            # intersect and their nesting can't be established
            ambiguous = False
            for i in range(n):
                if ambiguous:
                    break
                for j in range(i + 1, n):
                    nested = containment[i, j] or containment[j, i]
                    if nested:
                        continue
                    # neither shell contains the other: only valid
                    # if their axis aligned bounding boxes are disjoint
                    overlap_aabb = bounds.contains(
                        shells[i][1].bounds, interior[j].reshape((1, 3))
                    )[0] or bounds.contains(
                        shells[j][1].bounds, interior[i].reshape((1, 3))
                    )[0]
                    if overlap_aabb:
                        ambiguous = True
                        break

            if ambiguous:
                log.warning("intersecting or ambiguously nested shells detected; leaving shell orientations unmodified")
            else:
                # nesting depth: number of shells strictly containing
                # this one (the diagonal entry is excluded)
                for i, entry in enumerate(shells):
                    depth = int(containment[i].sum()) - 1
                    # even depth shells point out (positive volume),
                    # odd depth shells point in (negative volume)
                    want_positive = depth % 2 == 0
                    is_positive = entry[2] > 0.0
                    if want_positive != is_positive:
                        flip[entry[0]] = True

    # one or more faces needs flipping
    if flip.any():
        if "face_normals" in mesh._cache:
            # flip normals of necessary faces
            normals = mesh.face_normals.copy()
            normals[flip] *= -1.0
        else:
            normals = None
        # flip faces
        mesh.faces[flip] = np.fliplr(mesh.faces[flip])
        # assign corrected normals
        if normals is not None:
            mesh.face_normals = normals


def fix_normals(mesh, multibody=False):
    """
    Fix the winding and direction of a mesh face and
    face normals in-place.

    Really only meaningful on watertight meshes but will orient all
    faces and winding in a uniform way for non-watertight face
    patches as well.

    Parameters
    -------------
    mesh : trimesh.Trimesh
      Mesh to fix normals on
    multibody : bool
      if True try to correct normals direction
      on every body rather than just one

    Notes
    --------------
    mesh.faces : will flip columns on inverted faces
    """
    # traverse face adjacency to correct winding
    fix_winding(mesh)
    # check to see if a mesh is inverted
    fix_inversion(mesh, multibody=multibody)


def broken_faces(mesh, color=None):
    """
    Return the index of faces in the mesh which break the
    watertight status of the mesh.

    Parameters
    --------------
    mesh : trimesh.Trimesh
      Mesh to check broken faces on
    color: (4,) uint8 or None
      Will set broken faces to this color if not None

    Returns
    ---------------
    broken : (n, ) int
      Indexes of mesh.faces
    """
    adjacency = nx.from_edgelist(mesh.face_adjacency)
    broken = [k for k, v in dict(adjacency.degree()).items() if v != 3]
    broken = np.array(broken)
    if color is not None and broken.size != 0:
        # if someone passed a broken color
        color = np.array(color)
        if not (color.shape == (4,) or color.shape == (3,)):
            color = [255, 0, 0, 255]
        mesh.visual.face_colors[broken] = color
    return broken


def fill_holes(mesh, use_fan: bool = False):
    """
    Fill boundary holes in-place using fans, which may result
    in bad answers if the holes are non convex!

    Face colors and attributes will be padded with default values
    so shapes match.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh will be repaired in-place.
    use_fan
      If passed, holes larger than quads will be triangulated
      using fans which are only valid for non-convex holes.
    """
    if len(mesh.faces) < 3:
        return False
    if mesh.is_watertight:
        return True

    # any edge that occurs once is on the boundary
    boundary_groups = group_rows(mesh.edges_sorted, require_count=1)
    # either watertight or broken in a weird way
    if len(boundary_groups) < 3:
        return False

    # ordered edges on the boundary
    boundary = mesh.edges[boundary_groups]

    # use networkx to find cycles of boundary edges
    holes = nx.cycle_basis(nx.from_edgelist(boundary))

    # this handles mixed tris, quads, and arbitrary polygons
    new_faces = triangulate_quads(holes, use_fan=use_fan)
    if len(new_faces) == 0:
        return False

    # now we need to fix the winding for every new face
    new_edges = faces_to_edges(new_faces)

    # the hashable rows for the mesh boundary edges
    # and the boundary of the original hole in the mesh
    # a well-constructed mesh has edges that are REVERSED
    # so we're going to try a cheap indexing operation
    hashable_new = hashable_rows(new_edges)
    hashable_old = hashable_rows(boundary)

    # the magic trick: if any ordered edge of the NEW face is contained
    # in the OLD boundary edges, it means the new face has edges that
    # are in the same order as the boundary, and that new face
    # needs to be reversed and since new_edges has exactly three
    # edges per triangle we can compact indexes back to triangles
    needs_reverse = np.isin(hashable_new, hashable_old).reshape((-1, 3)).any(axis=1)

    # now we have some shiny new faces that might even be wound correctly!
    new_faces[needs_reverse] = np.fliplr(new_faces[needs_reverse])

    # do the bookkeeping to preserve face normals and pad attributes
    mesh.extend_faces(new_faces)

    return mesh.is_watertight


def stitch(mesh, faces=None, insert_vertices=False):
    """
    Create a fan stitch over the boundary of the specified
    faces. If the boundary is non-convex a triangle fan
    is going to be extremely wonky.

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Mesh to create fan stitch on.
    faces : (n,) int
      Face indexes to stitch with triangle fans.
    insert_vertices : bool
      Allow stitching to insert new vertices?

    Returns
    ----------
    fan : (m, 3) int
      New triangles referencing mesh.vertices.
    vertices : (p, 3) float
      Inserted vertices (only returned `if insert_vertices`)
    """
    if faces is None:
        faces = np.arange(len(mesh.faces))

    # get a sequence of vertex indices representing the
    # boundary of the specified faces
    # will be referencing the same indexes of `mesh.vertices`
    points = [
        e.points
        for e in faces_to_path(mesh, faces)["entities"]
        if len(e.points) > 3 and e.points[0] == e.points[-1]
    ]

    # get properties to avoid querying in loop
    vertices = mesh.vertices
    normals = mesh.face_normals

    # find which faces are associated with an edge
    edges_face = mesh.edges_face
    tree_edge = mesh.edges_sorted_tree

    if insert_vertices:
        # create one new vertex per curve at the centroid
        centroids = np.array([vertices[p].mean(axis=0) for p in points])
        # save the original length of the vertices
        count = len(vertices)
        # for the normal check stack our local vertices
        vertices = np.vstack((vertices, centroids))
        # create a triangle between our new centroid vertex
        # and each one of the boundary curves
        fan = [
            np.column_stack((np.ones(len(p) - 1, dtype=int) * (count + i), p[:-1], p[1:]))
            for i, p in enumerate(points)
        ]
    else:
        # since we're not allowed to insert new vertices
        # create a triangle fan for each boundary curve
        fan = [
            np.column_stack((np.ones(len(p) - 3, dtype=int) * p[0], p[1:-2], p[2:-1]))
            for p in points
        ]

    # now we do a normal check against an adjacent face
    # to see if each region needs to be flipped
    for i, t in zip(range(len(fan)), fan):
        # get the edges from the original mesh
        # for the first `n` new triangles
        e = t[:10, 1:].copy()
        e.sort(axis=1)

        # find which indexes of `mesh.edges` these
        # new edges correspond with by finding edges
        # that exactly correspond with the tree
        query = tree_edge.query_ball_point(e, r=1e-10)
        if len(query) == 0:
            continue
        # stack all the indices that exist
        edge_index = np.concatenate(query)

        # get the normals from the original mesh
        original = normals[edges_face[edge_index]]

        # calculate the normals for a few new faces
        check, valid = triangles.normals(vertices[t[:3]])
        if not valid.any():
            continue
        # take the first valid normal from our new faces
        check = check[0]

        # if our new faces are reversed from the original
        # Adjacent face flip them along their axis
        sign = np.dot(original, check)
        if sign.mean() < 0:
            fan[i] = np.fliplr(t)

    fan = np.vstack(fan)

    if insert_vertices:
        return fan, centroids
    return fan
