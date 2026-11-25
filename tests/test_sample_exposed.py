try:
    from . import generic as g
except BaseException:
    import generic as g


class ExposedSampleTest(g.unittest.TestCase):
    def _mated_boxes(self):
        # two unit boxes sharing one full face at X=0.5
        first = g.trimesh.creation.box()
        second = g.trimesh.creation.box()
        second.apply_translation([1.0, 0.0, 0.0])
        return g.trimesh.util.concatenate([first, second])

    def test_mated_boxes_hide_contact_faces(self):
        mesh = self._mated_boxes()

        centers = mesh.triangles_center
        internal = g.np.flatnonzero(
            g.np.isclose(centers[:, 0], 0.5) & (g.np.abs(mesh.face_normals[:, 0]) > 0.9)
        )
        assert len(internal) == 4
        external = g.np.setdiff1d(g.np.arange(len(mesh.faces)), internal)

        count = 20000
        points, index = g.trimesh.sample.sample_surface(
            mesh, count, seed=2024, only_exposed=True
        )
        assert points.shape == (count, 3)

        # no sample may land on the fully mated faces
        assert not g.np.isin(index, internal).any()

        # every external triangle must still be reachable
        assert set(external.tolist()) <= set(g.np.unique(index).tolist())

        # returned indices must refer to the original faces
        assert index.min() >= 0
        assert index.max() < len(mesh.faces)

        # independent outer-area calculation:
        # the union is a 2x1x1 box, outer area 2*(2 + 2 + 1) = 10
        assert g.np.isclose(mesh.area_faces[external].sum(), 10.0)
        assert g.np.isclose(g.trimesh.sample.exposed_area(mesh), 10.0)

        # sampled frequencies follow the exposed area weights
        probability = mesh.area_faces[external]
        probability = probability / probability.sum()
        frequency = g.np.array([(index == i).sum() for i in external]) / count
        assert g.np.abs(frequency - probability).max() < 0.02

    def test_seed_stability_and_wrapper(self):
        mesh = self._mated_boxes()
        internal = [10, 11, 12, 14]

        for seed in range(10):
            _p, index = g.trimesh.sample.sample_surface(
                mesh, 2000, seed=seed, only_exposed=True
            )
            assert not g.np.isin(index, internal).any()

        points, index = g.trimesh.sample.sample_surface(
            mesh, 500, seed=7, only_exposed=True
        )
        wrapped, wrapped_index = mesh.sample(
            500, return_index=True, seed=7, only_exposed=True
        )
        assert g.np.array_equal(wrapped, points)
        assert g.np.array_equal(wrapped_index, index)

        # common alternative keyword spellings route to the same mode
        for keyword in ("exposed_only", "external_only", "exclude_hidden"):
            _p, alt_index = g.trimesh.sample.sample_surface(
                mesh, 500, seed=7, **{keyword: True}
            )
            assert not g.np.isin(alt_index, internal).any()

    def test_normal_mode_unchanged(self):
        mesh = self._mated_boxes()
        internal = [10, 11, 12, 14]

        # default mode still samples the mated faces
        _p, index = g.trimesh.sample.sample_surface(mesh, 20000, seed=2024)
        assert g.np.isin(index, internal).any()

        # and the default path is deterministic
        a = g.trimesh.sample.sample_surface(mesh, 1000, seed=11)[0]
        b = g.trimesh.sample.sample_surface(mesh, 1000, seed=11)[0]
        assert g.np.array_equal(a, b)

    def test_separated_parts_all_exposed(self):
        first = g.trimesh.creation.box()
        second = g.trimesh.creation.box()
        second.apply_translation([5.0, 0.0, 0.0])
        mesh = g.trimesh.util.concatenate([first, second])

        mask = g.trimesh.sample.exposed_face_mask(mesh)
        assert mask.dtype == bool
        assert mask.all()
        assert g.np.isclose(g.trimesh.sample.exposed_area(mesh), 12.0)

        _p, index = g.trimesh.sample.sample_surface(mesh, 4000, seed=0, only_exposed=True)
        assert set(g.np.unique(index).tolist()) == set(range(24))

    def test_internal_cavity_walls_kept(self):
        outer = g.trimesh.creation.box(extents=[3.0, 3.0, 3.0])
        inner = g.trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        inner.invert()
        mesh = g.trimesh.util.concatenate([outer, inner])

        mask = g.trimesh.sample.exposed_face_mask(mesh)
        # the second shell (the cavity walls, faces 12..23) stays exposed
        assert mask[12:].all()

        _p, index = g.trimesh.sample.sample_surface(
            mesh, 20000, seed=1, only_exposed=True
        )
        assert g.np.isin(index, g.np.arange(12, 24)).any()

        # a cavity may also be represented by nested outward shells;
        # parity containment interprets the inner shell as cavity walls,
        # so those faces must be retained there as well
        nested = g.trimesh.util.concatenate(
            [
                g.trimesh.creation.box(extents=[3.0, 3.0, 3.0]),
                g.trimesh.creation.box(extents=[1.0, 1.0, 1.0]),
            ]
        )
        assert g.trimesh.sample.exposed_face_mask(nested)[12:].all()

    def test_internal_partition_hidden(self):
        box = g.trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        vertices = g.np.array(
            [[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]],
            dtype="float64",
        )
        divider = g.trimesh.Trimesh(
            vertices=vertices,
            faces=g.np.array([[0, 1, 2], [0, 2, 3]]),
            process=False,
        )
        mesh = g.trimesh.util.concatenate([box, divider])

        exposed = g.trimesh.sample.exposed_faces(mesh)
        assert 12 not in exposed
        assert 13 not in exposed
        assert set(range(12)) <= set(exposed.tolist())

    def test_open_mesh_conservative(self):
        source = g.trimesh.creation.box()
        mesh = g.trimesh.Trimesh(
            vertices=source.vertices, faces=source.faces[2:], process=False
        )
        # visibility cannot be fully determined: keep every face
        assert g.trimesh.sample.exposed_face_mask(mesh).all()
        points, index = g.trimesh.sample.sample_surface(
            mesh, 100, seed=2, only_exposed=True
        )
        assert points.shape == (100, 3)
        assert g.np.isin(index, g.np.arange(len(mesh.faces))).all()

    def test_empty_mesh(self):
        mesh = g.trimesh.Trimesh()
        assert g.trimesh.sample.exposed_face_mask(mesh).shape == (0,)
        assert g.trimesh.sample.exposed_faces(mesh).shape == (0,)

        points, index = g.trimesh.sample.sample_surface(mesh, 0, only_exposed=True)
        assert points.shape == (0, 3)
        assert index.shape == (0,)

        with self.assertRaises(ValueError):
            g.trimesh.sample.sample_surface(mesh, 8, only_exposed=True)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
