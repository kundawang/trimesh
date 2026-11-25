import numpy as np

import trimesh


def test_line_line():

    from trimesh.path.intersections import line_line

    # simple 2D
    hit, point = line_line(origins=[[-10, 0], [0, -10]], directions=[[1, 0], [0, 1]])
    assert hit
    assert np.allclose(point, [0, 0])

    # simple 3D
    hit, point = line_line(
        origins=[[-10, 0, 0], [0, -10, 0]], directions=[[1, 0, 0], [0, 1, 0]]
    )
    assert hit
    assert np.allclose(point, [0, 0, 0])

    # simple 3D with normal
    hit, point = line_line(
        origins=[[-10, 0, 0], [0, -10, 0]],
        directions=[[1, 0, 0], [0, 1, 0]],
        plane_normal=[0, 0, 1],
    )
    assert hit
    assert np.allclose(point, [0, 0, 0])

    # simple 3D with wrong normal
    hit, point = line_line(
        origins=[[-10, 0, 0], [0, -10, 0]],
        directions=[[1, 0, 0], [0, 1, 0]],
        plane_normal=[1, 0, 0],
    )
    assert not hit
    assert point is None


def test_clip_polyline_box():
    # the requested reference case: a segment through the unit box must
    # be clipped to exactly x = -0.5 .. 0.5, not retained as a whole
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    line = np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    # function entry point and Trimesh method entry point agree
    runs = trimesh.intersections.clip_polyline(box, line)
    method_runs = box.clip_polyline(line)
    assert len(method_runs) == len(runs)
    assert all(np.allclose(a, b) for a, b in zip(method_runs, runs))

    assert len(runs) == 1
    assert np.allclose(runs[0], [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])

    # a diagonal segment gets exact triangle intersection points
    diagonal = np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])
    runs = box.clip_polyline(diagonal)
    assert len(runs) == 1
    assert np.allclose(runs[0][0], [-0.5, -0.5, -0.5])
    assert np.allclose(runs[0][-1], [0.5, 0.5, 0.5])


def test_clip_polyline_two_entries():
    # a polyline that enters, leaves and re-enters must produce two
    # separate runs, each with exact entry/exit intersection points
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    polyline = np.array(
        [
            [-2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    runs = box.clip_polyline(polyline)
    assert len(runs) == 2
    assert np.allclose(runs[0], [[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    assert np.allclose(runs[1], [[0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

    # a single segment crossing two disjoint closed regions also splits
    left = trimesh.creation.box(
        extents=(1.0, 1.0, 1.0),
        transform=trimesh.transformations.translation_matrix([-2.0, 0.0, 0.0]),
    )
    right = trimesh.creation.box(
        extents=(1.0, 1.0, 1.0),
        transform=trimesh.transformations.translation_matrix([2.0, 0.0, 0.0]),
    )
    combined = trimesh.util.concatenate([left, right])
    assert combined.is_watertight

    runs = combined.clip_polyline(np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    assert len(runs) == 2
    assert np.allclose(runs[0], [[-2.5, 0.0, 0.0], [-1.5, 0.0, 0.0]])
    assert np.allclose(runs[1], [[1.5, 0.0, 0.0], [2.5, 0.0, 0.0]])


def test_clip_polyline_tangency():
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    # a line touching a single vertex from the outside must not create
    # an inside run: the intervals on both sides are outside
    tangent = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    assert box.clip_polyline(tangent) == []

    # a segment lying on a face is inside under the closed-set convention
    # and is bounded by the exact face edges
    on_face = np.array([[-1.0, 0.0, 0.5], [1.0, 0.0, 0.5]])
    runs = box.clip_polyline(on_face)
    assert len(runs) == 1
    assert np.allclose(runs[0], [[-0.5, 0.0, 0.5], [0.5, 0.0, 0.5]])

    # a faceted sphere tangent line produces no run, while the line
    # through its center does
    sphere = trimesh.creation.uv_sphere(radius=1.0)
    assert sphere.clip_polyline(np.array([[-2.0, 0.0, 1.0], [2.0, 0.0, 1.0]])) == []
    runs = sphere.clip_polyline(np.array([[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
    assert len(runs) == 1
    assert np.allclose(np.abs(runs[0][:, 0]), np.abs(runs[0][0, 0]))
    assert runs[0][0, 0] < 0.0 < runs[0][-1, 0]


def test_clip_polyline_edge_cases():
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    # segment fully inside is kept whole
    inside = np.array([[-0.25, -0.25, -0.25], [0.25, 0.25, 0.25]])
    runs = box.clip_polyline(inside)
    assert len(runs) == 1 and np.allclose(runs[0], inside)

    # adjacent inside segments are joined at their shared vertex
    joined = np.array([[-0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.25, 0.0, 0.0]])
    runs = box.clip_polyline(joined)
    assert len(runs) == 1
    assert np.allclose(runs[0], joined)

    # segment starting exactly on the surface
    runs = box.clip_polyline(np.array([[-0.5, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    assert len(runs) == 1
    assert np.allclose(runs[0], [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])

    # zero-length inputs: inside point is a degenerate (discarded) run,
    # outside point yields nothing
    assert box.clip_polyline(np.zeros((2, 3))) == []
    assert box.clip_polyline(np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])) == []

    # single vertex, empty and malformed inputs
    assert box.clip_polyline(np.zeros((1, 3))) == []
    assert box.clip_polyline(np.zeros((0, 3))) == []
    for bad in (np.zeros(3), np.zeros((2, 4))):
        try:
            box.clip_polyline(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad.shape}")

    # segments with no surface contact at all
    assert box.clip_polyline(np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])) == []


def test_clip_polyline_rejects_open_mesh():
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    # drop faces so the mesh is no longer watertight
    open_mesh = box.submesh([[0, 1, 2, 3, 4]], append=True)
    assert not open_mesh.is_watertight

    try:
        open_mesh.clip_polyline(np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    except ValueError:
        return
    raise AssertionError("open mesh must be rejected")


if __name__ == "__main__":
    trimesh.util.attach_to_log()
    test_line_line()
    test_clip_polyline_box()
    test_clip_polyline_two_entries()
    test_clip_polyline_tangency()
    test_clip_polyline_edge_cases()
    test_clip_polyline_rejects_open_mesh()
