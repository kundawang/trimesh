"""
Tests for distance/radius based merging of nearby topological
endpoints on :class:`trimesh.path.Path2D`.

The default decimal-place ``digits`` behavior is covered elsewhere;
these tests focus on the optional actual-distance entry point.
"""

import networkx as nx
import numpy as np
import pytest

import trimesh
from trimesh.path.entities import Line


def gap_quadrilateral(gap=0.007):
    """
    Unit square as four separate line entities, where the scanned
    tail vertex ``4`` misses the start vertex ``0`` by ``gap``.
    """
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, gap],
        ]
    )
    entities = [
        Line(points=[0, 1]),
        Line(points=[1, 2]),
        Line(points=[2, 3]),
        Line(points=[3, 4]),
    ]
    return trimesh.path.Path2D(entities=entities, vertices=vertices, process=False)


def test_radius_closes_0007_gap_quadrilateral():
    path = gap_quadrilateral(gap=0.007)
    # before merging the open chain produces no closed polygon
    assert len(path.polygons_closed) == 0

    path.merge_vertices(radius=0.01)

    # the coincident-in-topology tail/start collapse to one corner
    assert len(path.vertices) == 4
    assert len(path.entities) == 4

    # the chain is now a single closed loop
    assert bool(path.is_closed)
    assert len(path.root) == 1
    assert len(path.polygons_closed) == 1

    polygon = path.polygons_closed[0]
    assert polygon.is_valid
    assert polygon.area == pytest.approx(1.0, abs=1e-9)

    coords = np.asarray(polygon.exterior.coords)
    corners = np.unique(np.round(coords, 8), axis=0)
    assert len(corners) == 4
    expected = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    for corner in expected:
        assert np.min(np.linalg.norm(corners - corner, axis=1)) == pytest.approx(
            0.0, abs=1e-6
        )


def test_radius_keeps_stable_lowest_index_representative():
    path = gap_quadrilateral(gap=0.007)
    path.merge_vertices(radius=0.01)
    # lowest original index (0) is the stable representative and keeps
    # its exact coordinate rather than an averaged position
    assert np.allclose(path.vertices[0], [0.0, 0.0])
    # every entity reference is in range and the loop closes 3 -> 0
    points = [e.points.tolist() for e in path.entities]
    assert [3, 0] in points
    flat = np.concatenate([np.asarray(e.points).ravel() for e in path.entities])
    assert flat.min() >= 0 and flat.max() < len(path.vertices)


def test_radius_merge_is_idempotent_and_deterministic():
    path = gap_quadrilateral()
    path.merge_vertices(radius=0.01)
    state = (len(path.vertices), [e.points.tolist() for e in path.entities])

    path.merge_vertices(radius=0.01)
    again = (len(path.vertices), [e.points.tolist() for e in path.entities])
    assert state == again


def test_radius_smaller_than_gap_is_noop():
    path = gap_quadrilateral(gap=0.007)
    path.merge_vertices(radius=0.005)
    assert len(path.vertices) == 5
    assert len(path.entities) == 4
    assert len(path.polygons_closed) == 0


def test_radius_zero_only_merges_coincident_endpoints():
    # a real 0.007 gap is not merged at radius zero
    near = gap_quadrilateral(gap=0.007)
    near.merge_vertices(radius=0.0)
    assert len(near.vertices) == 5

    # an exactly duplicated closing endpoint is merged
    duplicate = gap_quadrilateral(gap=0.0)
    duplicate.merge_vertices(radius=0.0)
    assert len(duplicate.vertices) == 4
    assert len(duplicate.polygons_closed) == 1


def test_two_nearby_independent_paths_are_not_connected():
    # two separate single segments, facing endpoints only 0.005 apart
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.005], [1.0, 0.005]])
    entities = [Line([0, 1]), Line([2, 3])]
    path = trimesh.path.Path2D(entities=entities, vertices=vertices, process=False)

    components_before = nx.number_connected_components(path.vertex_graph)
    assert components_before == 2

    path.merge_vertices(radius=0.01)

    # nothing is merged across the independent contours
    assert len(path.vertices) == 4
    assert len(path.entities) == 2
    assert [e.points.tolist() for e in path.entities] == [[0, 1], [2, 3]]
    assert nx.number_connected_components(path.vertex_graph) == 2


def test_two_nearby_independent_chains_are_not_connected():
    # two two-edge chains; cross-chain endpoints are within radius but
    # the facing endpoints belong to different connected components
    vertices = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [0.0, 0.006],
            [-0.5, 0.5],
            [1.0, 0.006],
        ]
    )
    entities = [Line([0, 1]), Line([1, 2]), Line([3, 4]), Line([4, 5])]
    path = trimesh.path.Path2D(entities=entities, vertices=vertices, process=False)

    path.merge_vertices(radius=0.01)

    assert len(path.vertices) == 6
    assert len(path.entities) == 4
    assert nx.number_connected_components(path.vertex_graph) == 2


def test_short_edge_is_not_compressed_to_zero_length():
    # a standalone edge shorter than the radius: its two endpoints are
    # already adjacent and must not be merged
    path = trimesh.path.Path2D(
        entities=[Line([0, 1])],
        vertices=np.array([[0.0, 0.0], [0.005, 0.0]]),
        process=False,
    )
    path.merge_vertices(radius=0.01)
    assert len(path.vertices) == 2
    assert len(path.entities) == 1
    assert path.entities[0].points.tolist() == [0, 1]
    assert np.linalg.norm(path.vertices[1] - path.vertices[0]) > 0.0


def test_interior_corners_are_not_radius_merged():
    # open polyline endpoints are far apart; two interior degree-two
    # vertices are very close and must survive even a large radius
    vertices = np.array(
        [[0.0, 0.0], [0.49, 0.5], [0.50, 0.5], [1.0, 0.0]]
    )
    path = trimesh.path.Path2D(
        entities=[Line([0, 1, 2, 3])], vertices=vertices, process=False
    )
    path.merge_vertices(radius=0.25)
    assert len(path.vertices) == 4
    assert len(path.entities) == 1


def test_empty_paths_are_deterministic():
    empty = trimesh.path.Path2D()
    assert len(empty.vertices) == 0
    empty.merge_vertices(radius=0.01)
    empty.merge_vertices()
    assert empty.is_empty

    zero_rows = trimesh.path.Path2D(
        entities=[], vertices=np.zeros((0, 2)), process=False
    )
    zero_rows.merge_vertices(radius=0.01)
    assert len(zero_rows.vertices) == 0


def test_duplicate_points_are_deterministic():
    # repeated references plus a duplicated closing point
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    path = trimesh.path.Path2D(
        entities=[Line([0, 1]), Line([1, 2]), Line([2, 3]), Line([3, 4])],
        vertices=vertices,
        process=False,
    )
    path.merge_vertices(radius=0.01)
    first = (path.vertices.tolist(), [e.points.tolist() for e in path.entities])
    path.merge_vertices(radius=0.01)
    second = (path.vertices.tolist(), [e.points.tolist() for e in path.entities])
    assert first == second
    assert len(path.vertices) == 4


@pytest.mark.parametrize("bad_radius", [-0.1, -1.0, float("nan"), float("inf"), "big"])
def test_invalid_radius_raises_value_error(bad_radius):
    path = gap_quadrilateral()
    with pytest.raises(ValueError):
        path.merge_vertices(radius=bad_radius)


def test_distance_alias_and_helper_methods():
    via_distance = gap_quadrilateral()
    via_distance.merge_vertices(distance=0.01)
    assert len(via_distance.vertices) == 4

    for method_name in (
        "merge_close_vertices",
        "merge_nearby_vertices",
        "merge_endpoints",
        "merge_nearby_endpoints",
        "merge_close_endpoints",
    ):
        path = gap_quadrilateral()
        getattr(path, method_name)(0.01)
        assert len(path.vertices) == 4, method_name

    repair_path = gap_quadrilateral()
    returned = trimesh.path.repair.merge_close_vertices(repair_path, radius=0.01)
    assert returned is repair_path
    assert len(repair_path.vertices) == 4


def test_default_digits_behavior_unchanged():
    # exact duplicate closing vertex: default merge still closes it
    path = gap_quadrilateral(gap=0.0)
    path.merge_vertices()
    assert len(path.vertices) == 4

    # the 0.007 gap is not merged by the default digit tolerance at
    # unit scale, matching the historical behavior
    gapped = gap_quadrilateral(gap=0.007)
    gapped.merge_vertices()
    assert len(gapped.vertices) == 5

    # explicit coarse digits still perform rounding based merging
    coarse = gap_quadrilateral(gap=0.007)
    coarse.merge_vertices(digits=1)
    assert len(coarse.vertices) <= 4
