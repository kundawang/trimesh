try:
    from . import generic as g
except BaseException:
    import generic as g


class MultiboxTest(g.unittest.TestCase):
    def test_multibox_internal_face_removal(self):
        np = g.np

        # Create a small 2 * 2 * 2 binary voxelgrid
        voxelgrid = np.ones((2, 2, 2), dtype=bool)
        indices = np.argwhere(voxelgrid)
        points = g.trimesh.voxel.ops.indices_to_points(indices)

        # With internal faces (remove_internal_faces=False)
        mesh_w_internal_faces = g.trimesh.voxel.ops.multibox(
            points, pitch=1.0, remove_internal_faces=False
        )

        # Without internal faces (remove_internal_faces=True)
        mesh_wo_internal_faces = g.trimesh.voxel.ops.multibox(
            points, pitch=1.0, remove_internal_faces=True
        )
        self.assertLess(
            len(mesh_wo_internal_faces.faces), len(mesh_w_internal_faces.faces)
        )

        # Check bounding boxes are equal
        g.np.testing.assert_allclose(
            mesh_w_internal_faces.bounds, mesh_wo_internal_faces.bounds
        )

        # A 2 * 2 * 2 voxelgrid constitutes 8 voxels
        # Each voxel has 6 sides, with 2 triangular faces per side, totaling 6 * 2 = 12 triangular faces
        # 8 voxels * 12 triangular faces = 96 triangluar faces (when all the internal faces kept)
        self.assertEqual(len(mesh_w_internal_faces.faces), 96)

        # A 2 * 2 * 2 block has 6 exposed sides (i.e., top, bottom, front, back, left, right)
        # Each exposed side has 4 sub-sides corresponding to the 4 voxels, with 2 triangular faces per sub-side, totaling 4 * 2 = 8 triangular faces
        # 6 exposed sides * 8 triangular face = 48 triangular faces (when all the internal faces removed)
        self.assertEqual(len(mesh_wo_internal_faces.faces), 48)

    def test_multibox_manifold_edge_touching(self):
        np = g.np

        # two voxels touching along a single edge (no shared face)
        centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

        # the default representation merges the contact and is not watertight:
        # the shared geometric edge carries four triangles
        rough = g.trimesh.voxel.ops.multibox(centers)
        _, rough_counts = np.unique(
            rough.edges_sorted, axis=0, return_counts=True
        )
        self.assertTrue(np.any(rough_counts == 4))
        self.assertFalse(rough.is_watertight)

        # the manifold output keeps the two voxels as independent closed shells
        mesh = g.trimesh.voxel.ops.multibox(centers, manifold=True)

        # occupied volume is preserved
        np.testing.assert_allclose(mesh.volume, 2.0)

        # every edge is incident on exactly two triangles
        _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
        self.assertTrue(np.all(counts == 2))
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)

        # exactly two independent closed shells, each a unit-volume cube
        shells = mesh.split(only_watertight=False)
        self.assertEqual(len(shells), 2)
        for shell in shells:
            self.assertTrue(shell.is_watertight)
            np.testing.assert_allclose(shell.volume, 1.0)
            self.assertEqual(len(shell.faces), 12)

        # exterior surface positions are unchanged relative to the rough boxes
        np.testing.assert_allclose(mesh.bounds, rough.bounds)

    def test_multibox_manifold_face_adjacent(self):
        np = g.np

        # two face-adjacent voxels fuse into a single watertight 2-unit block
        edge_touch = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        face_adjacent = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        edge_mesh = g.trimesh.voxel.ops.multibox(edge_touch, manifold=True)
        face_mesh = g.trimesh.voxel.ops.multibox(face_adjacent, manifold=True)

        # same occupied volume, but face adjacency gives one shell, edge contact two
        np.testing.assert_allclose(face_mesh.volume, edge_mesh.volume)
        self.assertEqual(len(face_mesh.split(only_watertight=False)), 1)
        self.assertEqual(len(edge_mesh.split(only_watertight=False)), 2)

        # the fused block loses the shared face: 2 cubes * 12 triangles minus
        # 2 triangles for the removed face -> 20 triangles
        self.assertEqual(len(face_mesh.faces), 20)
        _, counts = np.unique(
            face_mesh.edges_sorted, axis=0, return_counts=True
        )
        self.assertTrue(np.all(counts == 2))
        self.assertTrue(face_mesh.is_volume)

    def test_multibox_manifold_empty_and_disconnected(self):
        np = g.np

        # empty grid produces a deterministic empty mesh
        empty = g.trimesh.voxel.ops.multibox(
            np.zeros((0, 3), dtype=np.float64), manifold=True
        )
        self.assertEqual(len(empty.faces), 0)
        self.assertEqual(len(empty.vertices), 0)

        # single voxel is one watertight cube with the standard bounds
        single = g.trimesh.voxel.ops.multibox(
            np.array([[0.0, 0.0, 0.0]]), manifold=True
        )
        self.assertTrue(single.is_watertight)
        np.testing.assert_allclose(single.volume, 1.0)
        np.testing.assert_allclose(
            single.bounds, np.array([[-0.5] * 3, [0.5] * 3])
        )

        # voxels touching only at a vertex stay separate closed shells
        point_touch = g.trimesh.voxel.ops.multibox(
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), manifold=True
        )
        self.assertEqual(
            len(point_touch.split(only_watertight=False)), 2
        )
        _, counts = np.unique(
            point_touch.edges_sorted, axis=0, return_counts=True
        )
        self.assertTrue(np.all(counts == 2))

        # widely separated voxels are independent shells too
        separated = g.trimesh.voxel.ops.multibox(
            np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]), manifold=True
        )
        self.assertEqual(len(separated.split(only_watertight=False)), 2)
        np.testing.assert_allclose(separated.volume, 2.0)

    def test_multibox_manifold_random_grids(self):
        np = g.np
        rng = np.random.default_rng(0)

        for _ in range(100):
            shape = tuple(rng.integers(1, 5, size=3))
            occupied = rng.random(shape) > rng.uniform(0.1, 0.9)
            centers = np.argwhere(occupied).astype(np.float64)
            if len(centers) == 0:
                continue

            mesh = g.trimesh.voxel.ops.multibox(centers, manifold=True)
            rough = g.trimesh.voxel.ops.multibox(centers)
            # the fair boundary reference: same internal-face removal,
            # without the manifold welding
            boundary = g.trimesh.voxel.ops.multibox(
                centers, remove_internal_faces=True
            )

            # no boundary edge is ever left dangling, volume is exact,
            # and the exterior surface stays in the same place
            _, counts = np.unique(
                mesh.edges_sorted, axis=0, return_counts=True
            )
            self.assertFalse(np.any(counts == 1))
            np.testing.assert_allclose(mesh.volume, float(len(centers)))
            np.testing.assert_allclose(mesh.bounds, rough.bounds)
            # the manifold mesh occupies exactly the same set of grid
            # vertices (exterior surface positions) as the boundary mesh;
            # only the vertex records at edge/vertex contacts are doubled
            manifold_grid = set(
                map(tuple, np.round(mesh.vertices * 2).astype(int))
            )
            boundary_grid = set(
                map(tuple, np.round(boundary.vertices * 2).astype(int))
            )
            self.assertEqual(manifold_grid, boundary_grid)

    def test_multibox_manifold_colors(self):
        np = g.np
        centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

        single_color = [255, 0, 0, 255]
        mesh = g.trimesh.voxel.ops.multibox(
            centers, manifold=True, colors=single_color
        )
        np.testing.assert_allclose(
            mesh.visual.face_colors, np.tile(single_color, (len(mesh.faces), 1))
        )

        per_voxel = np.array(
            [[255, 0, 0, 255], [0, 255, 0, 255]], dtype=np.uint8
        )
        colored = g.trimesh.voxel.ops.multibox(
            centers, manifold=True, colors=per_voxel
        )
        unique_colors = np.unique(
            colored.visual.face_colors, axis=0
        )
        # one color per independent shell, 12 triangles each
        np.testing.assert_allclose(
            np.sort(unique_colors, axis=0), np.sort(per_voxel, axis=0)
        )

    def test_as_boxes_manifold(self):
        np = g.np

        # two voxels that only touch at a vertex inside a 3x3x3 grid
        matrix = np.zeros((3, 3, 3), dtype=bool)
        matrix[0, 0, 0] = True
        matrix[2, 2, 2] = True
        grid = g.trimesh.voxel.VoxelGrid(matrix)

        # default one-box-per-voxel representation keeps both cubes and
        # globally merges the coincident contact vertex, which is not a
        # valid volume; the manifold output splits them into two shells
        manifold = grid.as_boxes(manifold=True)
        self.assertTrue(manifold.is_watertight)
        self.assertEqual(
            len(manifold.split(only_watertight=False)), 2
        )
        np.testing.assert_allclose(
            np.round(manifold.bounds, 8),
            np.round(grid.as_boxes().bounds, 8),
        )

        # a transformed grid keeps the correct scaled volume and bounds
        transformed = (
            g.trimesh.voxel.VoxelGrid(matrix)
            .apply_scale(0.5)
            .apply_translation([2.0, 3.0, 4.0])
        )
        mesh = transformed.as_boxes(manifold=True)
        self.assertTrue(mesh.is_watertight)
        np.testing.assert_allclose(mesh.volume, 2.0 * 0.5 ** 3)
        np.testing.assert_allclose(
            np.round(mesh.bounds, 8),
            np.round(transformed.as_boxes().bounds, 8),
        )


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
