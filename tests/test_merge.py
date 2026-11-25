try:
    from . import generic as g
except BaseException:
    import generic as g


class MargeTest(g.unittest.TestCase):
    def test_cube(self):
        """
        Test PointCloud object
        """

        m = g.trimesh.creation.box()

        assert m.vertices.shape == (8, 3)
        assert m.is_volume
        assert m.euler_number == 2

        # stack a bunch of unreferenced vertices
        m.vertices = g.np.vstack((m.vertices, g.random((10000, 3))))
        assert m.euler_number == 2
        assert m.vertices.shape == (10008, 3)
        assert m.referenced_vertices.sum() == 8
        copied = m.copy()

        # should remove unreferenced vertices
        m.merge_vertices()
        assert len(m.vertices) == 8
        assert m.is_volume
        assert m.euler_number == 2
        assert m.referenced_vertices.sum() == 8

        m.update_vertices(m.referenced_vertices)
        assert len(m.vertices) == 8
        assert m.is_volume
        assert m.euler_number == 2
        assert m.referenced_vertices.sum() == 8

        # check the copy with an int mask and see if inverse is created
        assert copied.referenced_vertices.sum() == 8
        assert copied.vertices.shape == (10008, 3)
        mask = g.np.nonzero(copied.referenced_vertices)[0]

        copied.update_vertices(mask)
        assert len(copied.vertices) == 8
        assert copied.is_volume
        assert copied.euler_number == 2
        assert copied.referenced_vertices.sum() == 8

    def test_nanoscale(self):
        """
        Vertices of a nanometer box must not collapse when the mesh is
        rebuilt with ``process=True``: the historical absolute merge
        tolerance of 1e-8 is larger than the geometry itself.
        """
        # the box from the reported failure
        extents = g.np.array([9.3562e-9, 7.649e-10, 7.38e-10])
        box = g.trimesh.creation.box(extents=extents)
        assert box.is_volume
        true_volume = float(g.np.prod(extents))

        # rebuild from the identical vertices and faces
        rebuilt = g.trimesh.Trimesh(
            vertices=box.vertices.copy(),
            faces=box.faces.copy(),
            process=True,
        )
        # all eight corners and all twelve faces survive
        assert len(rebuilt.vertices) == 8
        assert len(rebuilt.faces) == 12
        assert rebuilt.is_watertight
        # geometry and volume are preserved exactly
        assert g.np.allclose(g.np.sort(rebuilt.extents), g.np.sort(extents))
        assert g.np.isclose(rebuilt.volume, true_volume)

        # genuinely coincident vertices still merge
        duplicated = g.np.vstack((box.vertices, box.vertices[0]))
        merged = g.trimesh.Trimesh(
            vertices=duplicated, faces=box.faces.copy(), process=True
        )
        assert len(merged.vertices) == 8
        assert merged.is_watertight

        # real adjacent-but-distinct vertices (half the smallest edge)
        # must be kept, not collapsed onto their neighbor
        near = box.vertices.copy()
        near[1] = near[0] + g.np.array([extents.min() * 0.5, 0.0, 0.0])
        kept = g.trimesh.Trimesh(
            vertices=near, faces=box.faces.copy(), process=True
        )
        assert len(kept.vertices) == 8

    def test_nanoscale_offset_and_assembly(self):
        """
        Scale-aware merging has to survive large coordinate offsets,
        strongly different axis scales and nanometer components embedded
        in an ordinary-sized multi-component assembly.
        """
        extents = g.np.array([9.3562e-9, 7.649e-10, 7.38e-10])
        box = g.trimesh.creation.box(extents=extents)

        # translated far from the origin (offset stays within float64
        # resolution of the smallest edge)
        offset = g.np.array([1234.0, -5678.0, 900.0])
        shifted = g.trimesh.Trimesh(
            vertices=box.vertices + offset,
            faces=box.faces.copy(),
            process=True,
        )
        assert len(shifted.vertices) == 8
        assert g.np.allclose(shifted.extents, extents)
        assert g.np.isclose(shifted.volume, g.np.prod(extents))
        # absolute coordinates are retained through the merge
        assert g.np.allclose(
            shifted.bounds[0], (box.vertices + offset).min(axis=0)
        )

        # two nanometer components next to an ordinary unit-scale box
        small_a = g.trimesh.creation.box(extents=extents)
        small_b = g.trimesh.creation.box(extents=extents * 1.7)
        small_b.apply_translation([1e-6, 0.0, 0.0])
        unit = g.trimesh.creation.box(extents=g.np.ones(3))
        unit.apply_translation([0.0, 5.0, 0.0])
        assembly = g.trimesh.util.concatenate([small_a, small_b, unit])

        merged = g.trimesh.Trimesh(
            vertices=assembly.vertices.copy(),
            faces=assembly.faces.copy(),
            process=True,
        )
        # three boxes * eight corners, none swallowed by a neighbor
        assert len(merged.vertices) == 24
        assert len(merged.faces) == 36
        expected = g.np.prod(extents) + g.np.prod(extents * 1.7) + 1.0
        assert g.np.isclose(merged.volume, expected)

        # a magnified copy of the same box rebuilds cleanly as well
        scaled = g.trimesh.Trimesh(
            vertices=(box.vertices * 1e9).copy(),
            faces=box.faces.copy(),
            process=True,
        )
        assert len(scaled.vertices) == 8
        assert g.np.allclose(scaled.extents, extents * 1e9)

    def test_ordinary_scale_unchanged(self):
        """
        At ordinary unit scale the new merging must reproduce the
        historical absolute tol.merge behavior, including the merge of
        sub-tolerance jitter and regular-grid seam duplicates.
        """
        unit = g.trimesh.creation.box()
        assert len(unit.vertices) == 8

        # a jittered duplicate well inside tol.merge still merges
        jittered = g.np.vstack(
            (unit.vertices, unit.vertices[0] + 1e-10)
        )
        merged = g.trimesh.Trimesh(
            vertices=jittered,
            faces=g.np.vstack((unit.faces, [0, 1, 2])),
            process=True,
        )
        assert len(merged.vertices) == 8

        # a displacement larger than tol.merge is a real vertex
        separated = g.np.vstack(
            (unit.vertices, unit.vertices[0] + 1e-6)
        )
        new_face = g.np.array([[len(unit.vertices), 1, 2]])
        kept = g.trimesh.Trimesh(
            vertices=separated,
            faces=g.np.vstack((unit.faces, new_face)),
            process=True,
        )
        assert len(kept.vertices) == 9

        # regular parametrized surfaces keep their seam/pole topology
        for name, kwargs in [
            ("uv_sphere", {}),
            ("capsule", {"radius": 1.0, "height": 2.0}),
            ("cylinder", {"radius": 1.0, "height": 2.0}),
        ]:
            surface = getattr(g.trimesh.creation, name)(**kwargs)
            assert surface.is_watertight, name


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
