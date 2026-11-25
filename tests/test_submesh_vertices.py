"""
Tests for the safe non-mutating vertex-selected submesh entry point
`Trimesh.submesh_from_vertices` / `trimesh.submesh_from_vertices`.

`Trimesh.update_vertices` keeps its historical behavior (the default
inverse map sends removed vertex indices to zero, which can create
degenerate faces); the new entry point additionally drops every face
referencing a removed vertex and reindexes the retained vertices.
"""

import copy

import numpy as np
import pytest

import trimesh


def _colored_box():
    """Unit box with a distinct RGBA color per vertex and per face."""
    mesh = trimesh.creation.box()
    rng = np.random.RandomState(0)
    vertex_colors = rng.randint(0, 255, (len(mesh.vertices), 4), dtype=np.uint8)
    face_colors = rng.randint(0, 255, (len(mesh.faces), 4), dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, vertex_colors=vertex_colors, face_colors=face_colors
    )
    return mesh


def _top_mask(mesh):
    # unit box: exactly the four z == +0.5 corners
    return mesh.vertices[:, 2] > 0.0


def test_submesh_from_vertices_box_top():
    mesh = _colored_box()
    mask = _top_mask(mesh)

    sub = mesh.submesh_from_vertices(mask)

    # exactly the four top corners and the two top triangles survive
    assert len(sub.vertices) == 4
    assert len(sub.faces) == 2
    assert np.allclose(sub.vertices[:, 2], 0.5)

    # every face references a retained, in-range distinct triple
    assert sub.faces.max() < len(sub.vertices)
    for face in sub.faces:
        assert len({int(i) for i in face}) == 3

    # no zero-area / degenerate faces
    assert sub.area_faces.shape == (2,)
    assert (sub.area_faces > 0.0).all()

    # both surviving triangles are the top cap and point up
    assert np.allclose(sub.face_normals[:, 2], 1.0)

    # vertex colors follow the retained vertices through reindexing
    expected_colors = mesh.visual.vertex_colors[mask]
    assert np.array_equal(sub.visual.vertex_colors, expected_colors)

    # face colors follow the two retained faces
    expected_face_colors = mesh.visual.face_colors[mask[mesh.faces].all(axis=1)]
    assert np.array_equal(sub.visual.face_colors, expected_face_colors)


def test_submesh_from_vertices_face_colors_stored():
    # kind == "face" visuals keep an explicit per-face color array that
    # update_faces must subset (not just a vertex-derived color).
    mesh = trimesh.creation.box()
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    face_colors[:, 2] = 255
    top_face_ids = np.flatnonzero(_top_mask(mesh)[mesh.faces].all(axis=1))
    face_colors[top_face_ids, 0] = 255
    mesh.visual = trimesh.visual.ColorVisuals(face_colors=face_colors)

    sub = mesh.submesh_from_vertices(_top_mask(mesh))

    assert sub.visual.kind == "face"
    assert len(sub.visual.face_colors) == 2
    assert np.array_equal(sub.visual.face_colors, face_colors[top_face_ids])
    # both surviving top faces carry the marked magenta (R+B) color
    assert (sub.visual.face_colors[:, 0] == 255).all()
    assert (sub.visual.face_colors[:, 2] == 255).all()


def test_submesh_from_vertices_does_not_mutate_source():
    mesh = _colored_box()
    mask = _top_mask(mesh)
    vertices_before = mesh.vertices.copy()
    faces_before = mesh.faces.copy()
    colors_before = mesh.visual.vertex_colors.copy()

    sub = mesh.submesh_from_vertices(mask)
    sub.vertices[:] = 12345.0

    # source geometry and colors are untouched
    assert np.array_equal(mesh.vertices, vertices_before)
    assert np.array_equal(mesh.faces, faces_before)
    assert np.array_equal(mesh.visual.vertex_colors, colors_before)
    assert len(mesh.vertices) == 8
    assert len(mesh.faces) == 12


@pytest.mark.parametrize(
    "select",
    [
        lambda mask: mask,
        lambda mask: np.flatnonzero(mask),
        # index order and duplicates must not change the resulting set
        lambda mask: np.flatnonzero(mask)[::-1],
        lambda mask: np.repeat(np.flatnonzero(mask), 2),
        lambda mask: {int(i) for i in np.flatnonzero(mask)},
    ],
)
def test_bool_mask_and_indices_equivalent(select):
    mesh = _colored_box()
    mask = _top_mask(mesh)

    from_bool = mesh.submesh_from_vertices(mask)
    from_index = mesh.submesh_from_vertices(select(mask))

    assert np.array_equal(from_index.vertices, from_bool.vertices)
    assert np.array_equal(from_index.faces, from_bool.faces)
    assert np.array_equal(from_index.visual.vertex_colors, from_bool.visual.vertex_colors)


def test_submesh_from_vertices_all_kept_is_independent_copy():
    mesh = trimesh.creation.box()
    keep_all = np.ones(len(mesh.vertices), dtype=bool)

    sub = mesh.submesh_from_vertices(keep_all)

    assert len(sub.vertices) == len(mesh.vertices)
    assert len(sub.faces) == len(mesh.faces)
    assert np.array_equal(sub.vertices, mesh.vertices)
    assert np.array_equal(sub.faces, mesh.faces)

    # independent memory, not a view
    assert not np.shares_memory(sub.vertices, mesh.vertices)
    sub.vertices[0, 0] += 1000.0
    assert not np.allclose(sub.vertices, mesh.vertices)


@pytest.mark.parametrize(
    "empty",
    [
        lambda n: np.zeros(n, dtype=bool),
        lambda n: np.array([], dtype=bool),
        lambda n: [],
        lambda n: np.array([], dtype=np.int64),
    ],
)
def test_submesh_from_vertices_empty_result(empty):
    mesh = _colored_box()

    sub = mesh.submesh_from_vertices(empty(len(mesh.vertices)))

    assert len(sub.vertices) == 0
    assert len(sub.faces) == 0
    assert sub.is_empty
    # an empty colored result must remain copyable/consistent
    # (regression for the empty RGBA -> to_rgba shape path)
    assert len(sub.copy().vertices) == 0
    copy.deepcopy(sub)
    # asking again on the empty mesh must also succeed
    again = sub.submesh_from_vertices(np.zeros(0, dtype=bool))
    assert len(again.vertices) == 0
    assert len(again.faces) == 0


def test_empty_face_colored_submesh_is_copyable():
    mesh = trimesh.creation.box()
    face_colors = np.tile([0, 200, 0, 255], (len(mesh.faces), 1)).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(face_colors=face_colors)

    sub = mesh.submesh_from_vertices([])

    assert len(sub.vertices) == 0
    assert len(sub.faces) == 0
    assert sub.visual.kind == "face"
    assert sub.copy().visual.face_colors.shape == (0, 4)
    copy.deepcopy(sub)


def test_submesh_from_vertices_single_vertex_drops_all_faces():
    mesh = trimesh.creation.box()
    sub = mesh.submesh_from_vertices([7])
    assert len(sub.vertices) == 1
    assert len(sub.faces) == 0


def test_submesh_from_vertices_normals_attributes_and_cache():
    mesh = trimesh.creation.box()
    mask = _top_mask(mesh)

    mesh.vertex_attributes["my_vertex"] = np.arange(len(mesh.vertices)) * 10
    mesh.face_attributes["my_face"] = np.arange(len(mesh.faces)) * 100

    # prime the cache before subsetting
    _ = mesh.vertex_normals
    _ = mesh.face_normals

    sub = mesh.submesh_from_vertices(mask)

    assert sub.vertex_normals.shape == (4, 3)
    assert sub.face_normals.shape == (2, 3)
    assert np.array_equal(sub.vertex_attributes["my_vertex"], [10, 30, 50, 70])
    surviving_face_ids = np.flatnonzero(mask[mesh.faces].all(axis=1))
    assert np.array_equal(sub.face_attributes["my_face"], surviving_face_ids * 100)
    # cached hashes must verify against the new data store
    sub._cache.verify()


def test_submesh_from_vertices_uv():
    mesh = trimesh.creation.box()
    uv = np.random.RandomState(1).rand(len(mesh.vertices), 2)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
    mask = _top_mask(mesh)

    sub = mesh.submesh_from_vertices(mask)

    assert sub.visual.uv is not None
    assert sub.visual.uv.shape == (4, 2)
    assert np.allclose(sub.visual.uv, uv[mask])


def test_submesh_from_vertices_entry_points_and_aliases():
    mesh = trimesh.creation.box()
    mask = _top_mask(mesh)

    via_method = mesh.submesh_from_vertices(mask)
    via_util = trimesh.util.submesh_from_vertices(mesh, mask)
    via_top_level = trimesh.submesh_from_vertices(mesh, mask)
    via_alias_1 = mesh.submesh_by_vertices(mask)
    via_alias_2 = mesh.vertex_submesh(mask)
    via_alias_3 = mesh.select_vertices(mask)

    for other in [
        via_util,
        via_top_level,
        via_alias_1,
        via_alias_2,
        via_alias_3,
    ]:
        assert np.array_equal(other.faces, via_method.faces)
        assert np.array_equal(other.vertices, via_method.vertices)


def test_submesh_from_vertices_bad_mask_raises():
    mesh = trimesh.creation.box()
    with pytest.raises(IndexError):
        mesh.submesh_from_vertices(np.ones(len(mesh.vertices) - 1, dtype=bool))
    with pytest.raises(IndexError):
        mesh.submesh_from_vertices([len(mesh.vertices)])
    with pytest.raises(IndexError):
        mesh.submesh_from_vertices([-len(mesh.vertices) - 1])


def test_update_vertices_default_behavior_compatible():
    # the historical in-place API still exists and still honors an
    # explicitly supplied inverse map
    mesh = trimesh.creation.box()
    mask = np.ones(len(mesh.vertices), dtype=bool)
    mask[[0, 2, 4, 6]] = False

    inverse = np.zeros(len(mesh.vertices), dtype=np.int64)
    inverse[mask] = np.arange(mask.sum())

    before = mesh.vertices.copy()
    mesh.update_vertices(mask, inverse=inverse)

    # mutated in place and vertices subset exactly as before
    assert len(mesh.vertices) == int(mask.sum())
    assert np.allclose(mesh.vertices, before[mask])
    # every face was remapped through the supplied inverse, including
    # references to removed vertices mapping to zero (legacy behavior)
    assert mesh.faces.max() < int(mask.sum())
