"""
Regression tests for outline assembly at branching vertices.

Two triangulated squares touching only at a corner share a single mesh
vertex but no edge. Their boundary edges form two independent closed
loops, and outline assembly must return both loops with every boundary
edge represented exactly once.
"""

import numpy as np
import pytest

import trimesh
from trimesh import grouping
from trimesh.graph import edge_traversals, fill_traversals, traversals

# two squares, two triangles each, sharing only corner vertex 2
CORNER_VERTICES = np.array(
    [
        [1.0, 1.0, 0.0],  # 0
        [1.0, 3.0, 0.0],  # 1
        [3.0, 3.0, 0.0],  # 2 shared corner
        [3.0, 1.0, 0.0],  # 3
        [3.0, 5.0, 0.0],  # 4
        [5.0, 5.0, 0.0],  # 5
        [5.0, 3.0, 0.0],  # 6
    ]
)
CORNER_FACES = np.array([[0, 1, 2], [0, 2, 3], [2, 4, 5], [2, 5, 6]])


def _iter_segments(points):
    for i in range(len(points) - 1):
        yield int(points[i]), int(points[i + 1])


def _edge_counts(path):
    """Undirected edge -> number of times an entity segment uses it."""
    counts = {}
    for entity in path.entities:
        points = np.asarray(entity.points)
        for a, b in _iter_segments(points):
            edge = tuple(sorted((a, b)))
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def _boundary_edge_counts(mesh):
    edges_sorted = mesh.edges_sorted
    unique = grouping.group_rows(edges_sorted, require_count=1)
    rows = edges_sorted[unique]
    counts = {}
    for row in rows:
        edge = tuple(int(i) for i in row)
        counts[edge] = counts.get(edge, 0) + 1
    return counts


def _segment_directions(path):
    directions = set()
    for entity in path.entities:
        points = np.asarray(entity.points)
        directions.update(_iter_segments(points))
    return directions


def test_corner_contact_returns_two_closed_loops():
    mesh = trimesh.Trimesh(CORNER_VERTICES, CORNER_FACES)
    outline = mesh.outline()

    # the corner contact must not fragment the second loop into segments
    assert len(outline.entities) == 2
    assert all(entity.closed for entity in outline.entities)

    point_sequences = [entity.points.tolist() for entity in outline.entities]
    assert [0, 1, 2, 3, 0] in point_sequences
    assert [2, 4, 5, 6, 2] in point_sequences

    # paths view reports both loops as separate closed paths
    assert len(outline.paths) == 2

    # every boundary edge appears exactly once, nothing added or dropped
    counts = _edge_counts(outline)
    assert counts == _boundary_edge_counts(mesh)
    assert len(counts) == 8
    assert all(value == 1 for value in counts.values())


def test_corner_contact_preserves_source_direction():
    mesh = trimesh.Trimesh(CORNER_VERTICES, CORNER_FACES, process=False)
    outline = mesh.outline()

    source_directed = {tuple(map(int, edge)) for edge in mesh.edges}
    # consistent face winding means every segment keeps its source direction
    assert _segment_directions(outline) <= source_directed


def test_disjoint_squares_return_two_loops():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    outline = mesh.outline()

    assert len(outline.entities) == 2
    assert all(entity.closed for entity in outline.entities)
    counts = _edge_counts(outline)
    assert counts == _boundary_edge_counts(mesh)
    assert len(counts) == 8
    assert all(value == 1 for value in counts.values())


def test_shared_edge_square_pair_returns_single_outer_loop():
    # two triangulated squares with a truly shared edge (1, 2)
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [2.0, 0.0, 0.0],  # 4
            [2.0, 1.0, 0.0],  # 5
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]])
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    outline = mesh.outline()

    # the shared edge is used by two triangles so it is not on the boundary
    assert len(outline.entities) == 1
    assert outline.entities[0].closed
    counts = _edge_counts(outline)
    assert counts == _boundary_edge_counts(mesh)
    assert (1, 2) not in counts
    assert sum(counts.values()) == 6
    assert all(value == 1 for value in counts.values())


@pytest.mark.parametrize("mode", ["dfs", "bfs"])
def test_fill_traversals_keeps_corner_loops_intact(mode):
    edges = np.array([[0, 1], [0, 3], [1, 2], [2, 3], [2, 4], [2, 6], [4, 5], [5, 6]])
    filled = fill_traversals(traversals(edges, mode=mode), edges=edges)

    sequences = [np.asarray(item).tolist() for item in filled]
    assert [0, 1, 2, 3, 0] in sequences
    assert [2, 4, 5, 6, 2] in sequences

    # every source edge appears exactly once
    included = {}
    for sequence in filled:
        for i in range(len(sequence) - 1):
            edge = tuple(sorted((int(sequence[i]), int(sequence[i + 1]))))
            included[edge] = included.get(edge, 0) + 1
    expected = {tuple(sorted(map(int, edge))) for edge in edges}
    assert set(included) == expected
    assert all(value == 1 for value in included.values())


def test_edge_traversals_open_chain_is_one_path():
    chain = np.column_stack([np.arange(7), np.arange(1, 8)]).astype(np.int64)
    paths = edge_traversals(chain)
    assert len(paths) == 1
    assert paths[0].tolist() == list(range(8))
