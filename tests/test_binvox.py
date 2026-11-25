try:
    from . import generic as g
except BaseException:
    import generic as g

from io import BytesIO

from trimesh import transformations as tr
from trimesh import voxel as v
from trimesh.exchange import binvox
from trimesh.voxel import encoding as enc
from trimesh.voxel import runlength as rl


def _independent_binvox_parse(raw):
    """
    Parse binvox bytes without using any trimesh binvox code:
    plain header parsing plus a manual RLE payload decode.

    Returns
    -------
    dim : tuple of int
    translate : tuple of float
    scale : float
    dense : (dim) bool array in xyz order (binvox stores xzy)
    """
    np = g.np
    fp = BytesIO(raw)
    assert fp.readline().decode().strip().startswith("#binvox")
    header = {}
    while True:
        line = fp.readline().decode().strip()
        if line == "data":
            break
        key, *vals = line.split()
        header[key] = vals
    dim = tuple(int(x) for x in header["dim"])
    translate = tuple(float(x) for x in header["translate"])
    scale = float(header["scale"][0])

    # RLE payload: alternating (value, count) uint8 pairs
    bits = np.frombuffer(fp.read(), dtype=np.uint8).reshape(-1, 2)
    flat = np.concatenate(
        [np.full(count, value, dtype=np.uint8) for value, count in bits]
    ).astype(bool)
    dx, dy, dz = dim
    assert flat.size == dx * dy * dz
    # binvox flat order is xzy: index = x + dim_x * (z + dim_z * y)
    dense = flat.reshape(dx, dz, dy).transpose(0, 2, 1)
    return dim, translate, scale, dense


class BinvoxTest(g.unittest.TestCase):
    def test_load_save_invariance(self):
        np = g.np
        n = 4
        dense = np.random.uniform(size=(n,) * 3) > 0.8
        dense[0, 0, 0] = dense[-1, -1, -1] = 1  # ensure extent test works
        shape = dense.shape
        rl_data = rl.dense_to_rle(dense.flatten(), dtype=np.uint8)
        translate = np.array([2, 5, 10], dtype=np.float32)
        scale = 5.0
        base = binvox.voxel_from_binvox(
            rl_data, shape, translate, scale, axis_order="xzy"
        )
        s = scale / (n - 1)
        np.testing.assert_equal(
            base.transform,
            np.array([[s, 0, 0, 2], [0, s, 0, 5], [0, 0, s, 10], [0, 0, 0, 1]]),
        )
        dense = dense.transpose((0, 2, 1))
        bound_min = translate - 0.5 * s
        bound_max = translate + scale + 0.5 * s
        np.testing.assert_allclose(base.bounds, [bound_min, bound_max])
        np.testing.assert_equal(base.encoding.dense, dense)

        if binvox.binvox_encoder is None:
            g.log.warning("No binvox encoder found, skipping binvox export tests")
            return

        file_obj = BytesIO(binvox.export_binvox(base))
        file_obj.seek(0)
        loaded = binvox.load_binvox(file_obj)
        np.testing.assert_equal(loaded.encoding.dense, base.encoding.dense)
        self.assertTrue(isinstance(base, v.VoxelGrid))
        self.assertTrue(isinstance(loaded, v.VoxelGrid))
        np.testing.assert_equal(base.transform, loaded.transform)
        np.testing.assert_equal(base.shape, loaded.shape)

    def test_non_cubic_grid_export_and_readback(self):
        """
        A 3x4x5 grid with uniform isotropic pitch must export and read
        back exactly: non-cubic cell counts are not anisotropic voxels.
        """
        np = g.np
        shape = (3, 4, 5)
        dense = np.random.uniform(size=shape) > 0.85
        dense[0, 0, 0] = True
        dense[2, 3, 4] = True
        occupied = np.argwhere(dense)

        pitch = 0.5
        translate = np.array([1.0, 2.0, 3.0])
        matrix = tr.scale_and_translate(
            scale=[pitch] * 3, translate=translate
        )
        vox = v.VoxelGrid(enc.DenseEncoding(dense), matrix)

        # this used to raise "Can only export binvox with uniform scale"
        raw = binvox.export_binvox(vox)

        # independently parse the header and RLE payload
        dim, hdr_translate, hdr_scale, dense_disk = _independent_binvox_parse(raw)
        np.testing.assert_equal(dim, shape)
        np.testing.assert_allclose(hdr_translate, translate)
        # header scale references the longest axis: pitch * (5 - 1)
        np.testing.assert_allclose(hdr_scale, pitch * (max(shape) - 1))
        np.testing.assert_equal(dense_disk, dense)

        # read back through trimesh and check occupancy indices
        loaded = binvox.load_binvox(BytesIO(raw))
        self.assertEqual(loaded.shape, shape)
        np.testing.assert_equal(loaded.encoding.dense, dense)
        np.testing.assert_array_equal(
            np.sort(loaded.sparse_indices, axis=0),
            np.sort(occupied, axis=0),
        )

        # transform and world-space geometry round trip
        np.testing.assert_allclose(loaded.transform, vox.transform)
        np.testing.assert_allclose(loaded.scale, [pitch] * 3)
        np.testing.assert_allclose(loaded.bounds, vox.bounds)
        expected_min = translate + (occupied.min(axis=0) - 0.5) * pitch
        expected_max = translate + (occupied.max(axis=0) + 0.5) * pitch
        np.testing.assert_allclose(loaded.bounds[0], expected_min)
        np.testing.assert_allclose(loaded.bounds[1], expected_max)
        centers = translate + occupied * pitch
        np.testing.assert_allclose(
            np.sort(loaded.points, axis=0), np.sort(centers, axis=0)
        )

        # xyz axis ordering must round trip as well
        raw_xyz = binvox.export_binvox(vox, axis_order="xyz")
        loaded_xyz = binvox.load_binvox(BytesIO(raw_xyz), axis_order="xyz")
        np.testing.assert_equal(loaded_xyz.encoding.dense, dense)
        np.testing.assert_allclose(loaded_xyz.transform, vox.transform)

    def test_non_cubic_header_convention(self):
        """
        Files written with the documented non-cubic convention
        (`scale = pitch * (max(shape) - 1)`) reconstruct uniform pitch.
        """
        np = g.np
        shape = (3, 4, 5)
        dense = np.zeros(shape, dtype=bool)
        dense[0, 0, 0] = dense[2, 3, 4] = True
        pitch = 0.25
        translate = np.array([-1.0, 0.5, 7.0])

        rle_data = rl.dense_to_rle(
            dense.transpose(0, 2, 1).flatten(), dtype=np.uint8
        )
        raw = binvox.binvox_bytes(
            rle_data,
            shape=shape,
            translate=translate,
            scale=pitch * (max(shape) - 1),
        )
        loaded = binvox.load_binvox(BytesIO(raw))
        np.testing.assert_equal(loaded.encoding.dense, dense)
        np.testing.assert_allclose(loaded.scale, [pitch] * 3)
        np.testing.assert_allclose(loaded.translation, translate)

        # re-exporting preserves occupancy and geometry
        reloaded = binvox.load_binvox(
            BytesIO(binvox.export_binvox(loaded))
        )
        np.testing.assert_equal(reloaded.encoding.dense, dense)
        np.testing.assert_allclose(reloaded.transform, loaded.transform)

    def test_cubic_backward_compatible(self):
        """Cubic grids keep the historical scale and transform."""
        np = g.np
        n = 4
        dense = np.random.uniform(size=(n, n, n)) > 0.7
        pitch = 5.0 / (n - 1)
        translate = np.array([2.0, 5.0, 10.0])
        vox = v.VoxelGrid(
            enc.DenseEncoding(dense),
            tr.scale_and_translate(scale=[pitch] * 3, translate=translate),
        )
        raw = binvox.export_binvox(vox)
        header = raw[: raw.index(b"data\n")].decode()
        assert f"dim {n} {n} {n}" in header
        assert "scale 5.0" in header

        loaded = binvox.load_binvox(BytesIO(raw))
        np.testing.assert_equal(loaded.encoding.dense, dense)
        np.testing.assert_allclose(loaded.transform, vox.transform)

    def test_unrepresentable_transforms_raise(self):
        """
        Anisotropic pitch, rotation and shear cannot be expressed in
        binvox and must raise instead of being silently deformed.
        """
        np = g.np
        dense = np.zeros((3, 4, 5), dtype=bool)
        dense[0, 0, 0] = True

        # genuinely anisotropic voxel pitch
        aniso = v.VoxelGrid(
            enc.DenseEncoding(dense),
            tr.scale_and_translate(scale=[0.5, 0.5, 1.0]),
        )
        with self.assertRaises(ValueError):
            binvox.export_binvox(aniso)

        # rotated transform
        angle = 0.3
        rot = np.eye(4)
        rot[:2, :2] = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
        with self.assertRaises(ValueError):
            binvox.export_binvox(v.VoxelGrid(enc.DenseEncoding(dense), rot))

        # sheared transform
        shear = np.eye(4)
        shear[0, 1] = 0.4
        with self.assertRaises(ValueError):
            binvox.export_binvox(v.VoxelGrid(enc.DenseEncoding(dense), shear))

    def test_negative_scale_flip_preserves_geometry(self):
        """
        A reflected axis is representable in binvox via a flipped grid:
        the occupied world points and bounds must be preserved.
        """
        np = g.np
        shape = (3, 4, 5)
        dense = np.random.uniform(size=shape) > 0.8
        dense[0, 0, 0] = dense[2, 3, 4] = True
        pitch = 0.5
        matrix = tr.scale_and_translate(
            scale=[-pitch, pitch, pitch], translate=[9.0, 8.0, 7.0]
        )
        flipped = v.VoxelGrid(enc.DenseEncoding(dense), matrix)

        loaded = binvox.load_binvox(BytesIO(binvox.export_binvox(flipped)))
        np.testing.assert_allclose(
            np.sort(loaded.points, axis=0),
            np.sort(flipped.points, axis=0),
        )
        np.testing.assert_allclose(loaded.bounds, flipped.bounds)
        np.testing.assert_allclose(loaded.scale, [pitch] * 3)
        np.testing.assert_allclose(
            loaded.translation[0], 9.0 - pitch * (shape[0] - 1)
        )


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
