try:
    from . import generic as g
except BaseException:
    import generic as g

import numpy as np


class CameraTests(g.unittest.TestCase):
    def test_K(self):
        resolution = (320, 240)
        fov = (60, 40)
        camera = g.trimesh.scene.Camera(resolution=resolution, fov=fov)

        # ground truth matrix
        K_expected = np.array(
            [[277.128, 0, 160], [0, 329.697, 120], [0, 0, 1]], dtype=np.float64
        )

        assert np.allclose(camera.K, K_expected, rtol=1e-3)

        # check to make sure assignment from matrix works
        K_set = K_expected.copy()
        K_set[:2, 2] = 300
        camera.K = K_set
        assert np.allclose(camera.resolution, 600)

    def test_consistency(self):
        resolution = (320, 240)
        focal = None
        fov = (60, 40)
        camera = g.trimesh.scene.Camera(resolution=resolution, focal=focal, fov=fov)
        assert np.allclose(camera.fov, fov)
        camera = g.trimesh.scene.Camera(
            resolution=resolution, focal=camera.focal, fov=None
        )
        assert np.allclose(camera.fov, fov)

    def test_focal_updates_on_resolution_change(self):
        """
        Test changing resolution with set fov updates focal.
        """
        base_res = (320, 240)
        updated_res = (640, 480)
        fov = (60, 40)

        # start with initial data
        base_cam = g.trimesh.scene.Camera(resolution=base_res, fov=fov)
        # update both focal length and resolution
        base_focal = base_cam.focal
        base_cam.resolution = updated_res

        assert not g.np.allclose(base_cam.focal, base_focal)

        # camera created with same arguments should
        # have the same values
        new_cam = g.trimesh.scene.Camera(resolution=updated_res, fov=fov)
        assert g.np.allclose(base_cam.focal, new_cam.focal)

    def test_fov_updates_on_resolution_change(self):
        """
        Test changing resolution with set focal updates fov.
        """
        base_res = (320, 240)
        updated_res = (640, 480)
        focal = (100, 100)
        base_cam = g.trimesh.scene.Camera(resolution=base_res, focal=focal)
        base_fov = base_cam.fov
        base_cam.resolution = updated_res
        assert base_cam.fov is not base_fov
        new_cam = g.trimesh.scene.Camera(
            resolution=updated_res,
            focal=focal,
        )
        np.testing.assert_allclose(base_cam.fov, new_cam.fov)

    def test_lookat(self):
        """
        Test the "look at points" function
        """
        # original points
        ori = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])

        # draw inside the block: `g.random` gave every iteration the same
        # extents and offset so this tested one camera ten times
        with g.RandomSeed() as r:
            for _i in range(10):
                # set the extents to be random but positive
                extents = r.random() * 10
                points = g.trimesh.util.stack_3D(ori.copy() * extents)

                fov = g.np.array([20, 50])

                # offset the points by a random amount
                offset = (r.random(3) - 0.5) * 100
                moved = points + offset
                T = g.trimesh.scene.cameras.look_at(moved, fov)

                # check using trig
                check = (np.ptp(points, axis=0)[:2] / 2.0) / g.np.tan(np.radians(fov / 2))

                # `look_at` returns a camera-to-world pose so the distance is
                # from the camera back to what it's pointed at: comparing the
                # world-to-camera translation against this only ever passed
                # because the offset was a fixed large negative number
                distance = T[2, 3] - moved[:, 2].mean()

                # camera should be far enough back to fit every point in frame
                assert distance + 1e-8 >= check.max()

        # just run to test other arguments
        # TODO(unknown): find the way to test it correctly
        g.trimesh.scene.cameras.look_at(points, fov, center=points[0])
        g.trimesh.scene.cameras.look_at(points, fov, distance=1)

    def test_ray_index(self):
        # make sure to_rays is giving valid indexes
        s = g.trimesh.scene.Scene()
        res = g.np.array([512, 512])
        for i in range(0, 1000, 79):
            current = res + i
            s.camera.resolution = current
            # get ray index of camera
            rid = s.camera.to_rays()[1]
            assert all(rid.min(axis=0) == 0)
            assert all(rid.max(axis=0) == current - 1)

    def test_ray_pixel_pinhole(self):
        # rays must match an independent pinhole-camera calculation,
        # including non-square resolutions and non-square fields of view
        for res in [(64, 32), (32, 64), (96, 48), (64, 64)]:
            for fov in [(60.0, 40.0), (40.0, 60.0), (55.0, 55.0)]:
                camera = g.trimesh.scene.Camera(resolution=res, fov=fov)
                width, height = res
                xy, pixels = g.trimesh.scene.cameras.ray_pixel_coords(camera)

                # every (column, row) pair appears exactly once
                assert pixels.shape == (width * height, 2)
                assert set(map(tuple, pixels.tolist())) == {
                    (c, r) for c in range(width) for r in range(height)
                }

                # independent pinhole formula from the intrinsic matrix:
                # pixel centers at col + 0.5, image y grows downward so
                # camera y is negated; camera looks down negative z
                fx, fy = camera.focal
                cols, rows = pixels[:, 0], pixels[:, 1]
                expected = g.np.column_stack(
                    (
                        (cols + 0.5 - width / 2.0) / fx,
                        -(rows + 0.5 - height / 2.0) / fy,
                    )
                )
                g.np.testing.assert_allclose(xy, expected, atol=1e-12)

                # unit ray directions share the same ordering
                rays, ray_pixels = camera.to_rays()
                g.np.testing.assert_array_equal(ray_pixels, pixels)
                expected_rays = g.trimesh.util.unitize(
                    g.np.column_stack((xy, -g.np.ones(len(xy))))
                )
                g.np.testing.assert_allclose(rays, expected_rays, atol=1e-12)

                # corner rays point into the expected image quadrants:
                # left is negative x, right positive x, top positive y
                def xy_at(col, row, xy=xy, pixels=pixels):
                    i = ((pixels == g.np.array([col, row])).all(axis=1)).argmax()
                    return xy[i]

                assert xy_at(0, 0)[0] < 0 and xy_at(0, 0)[1] > 0
                assert xy_at(width - 1, 0)[0] > 0
                assert xy_at(0, height - 1)[1] < 0
                assert xy_at(width - 1, height - 1)[0] > 0
                assert xy_at(width - 1, height - 1)[1] < 0

    def test_rays_match_projection(self):
        # ray-mesh hits and an independent pinhole rasterization of the
        # same geometry must agree per pixel, for non-square resolutions
        # and a rotated camera pose. An off-center marker makes the check
        # sensitive to scrambled pixel/ray ordering, not just silhouette
        from PIL import Image, ImageDraw

        box = g.trimesh.creation.box(extents=(1.6, 1.0, 1.0))
        # off-center marker: its asymmetric projection catches a
        # scrambled pixel/ray ordering, not just silhouette coverage
        marker = g.trimesh.creation.box(extents=(0.12, 0.12, 0.06))
        marker.apply_translation([0.0, 0.12, 0.54])
        scene = g.trimesh.scene.Scene([box, marker])
        angle = g.np.radians(10.0)
        pose = g.trimesh.transformations.rotation_matrix(angle, [0, 1, 0])

        def erode(mask, steps):
            for _ in range(steps):
                mask &= g.np.roll(mask, 1, 0) & g.np.roll(mask, -1, 0)
                mask &= g.np.roll(mask, 1, 1) & g.np.roll(mask, -1, 1)
            return mask

        def rasterize(mesh):
            world_to_cam = g.np.linalg.inv(scene.camera_transform)
            points = g.trimesh.transformations.transform_points(
                mesh.vertices, world_to_cam
            )
            K = scene.camera.K
            image = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(image)
            for face in mesh.faces:
                p = points[face]
                if (p[:, 2] >= 0).any():
                    continue
                u = K[0, 0] * p[:, 0] / (-p[:, 2]) + K[0, 2]
                # camera +y points up while image rows grow downward,
                # so the projected row coordinate must be flipped
                v = height - (K[1, 1] * p[:, 1] / (-p[:, 2]) + K[1, 2])
                draw.polygon(list(zip(u.tolist(), v.tolist())), fill=255)
            return g.np.array(image) > 0

        for res in [(64, 32), (48, 96)]:
            scene.camera.resolution = res
            scene.camera.fov = 60.0 * (g.np.array(res) / max(res))
            rotated = pose.copy()
            rotated[:3, 3] = scene.camera_transform[:3, 3]
            scene.camera_transform = rotated

            width, height = res
            origins, vectors, pixels = scene.camera_rays()

            # ray mask for the off-center marker alone
            _points, index_ray, _tri = marker.ray.intersects_location(
                origins, vectors, multiple_hits=False
            )
            ray_mask = g.np.zeros((height, width), dtype=bool)
            hit = pixels[index_ray]
            ray_mask[hit[:, 1], hit[:, 0]] = True

            image_mask = rasterize(marker)

            # per-pixel masks must match everywhere except the one-pixel
            # rasterized contour, where polygon fill rounds edges
            interior = erode(image_mask.copy(), 1)
            background_interior = erode((~image_mask).copy(), 1)
            assert g.np.array_equal(ray_mask & interior, image_mask & interior)
            assert g.np.array_equal(
                ray_mask & background_interior,
                image_mask & background_interior,
            )
            assert ray_mask.sum() > 5 and image_mask.sum() > 5
            # the whole scene silhouette must agree the same way
            full_ray = g.np.zeros((height, width), dtype=bool)
            for geometry in scene.geometry.values():
                _points, index_ray, _tri = geometry.ray.intersects_location(
                    origins, vectors, multiple_hits=False
                )
                hit = pixels[index_ray]
                full_ray[hit[:, 1], hit[:, 0]] = True
            full_image = g.np.zeros((height, width), dtype=bool)
            for geometry in scene.geometry.values():
                full_image |= rasterize(geometry)
            interior = erode(full_image.copy(), 1)
            background_interior = erode((~full_image).copy(), 1)
            assert g.np.array_equal(full_ray & interior, full_image & interior)
            assert g.np.array_equal(
                full_ray & background_interior,
                full_image & background_interior,
            )

    def test_rays_update_on_resize(self):
        # changing the viewport size must keep rays consistent with
        # the new pinhole geometry
        scene = g.trimesh.scene.Scene(g.trimesh.creation.box(extents=(1.0, 1.0, 1.0)))
        camera = scene.camera
        for res in [(64, 64), (64, 32), (32, 64), (128, 40)]:
            camera.resolution = res
            width, height = res
            origins, vectors, pixels = scene.camera_rays()
            assert origins.shape == vectors.shape == (width * height, 3)
            assert tuple(pixels.min(axis=0)) == (0, 0)
            assert tuple(pixels.max(axis=0)) == (width - 1, height - 1)
            g.np.testing.assert_allclose(g.np.linalg.norm(vectors, axis=1), 1.0)
            xy, _ = g.trimesh.scene.cameras.ray_pixel_coords(camera)
            fx, fy = camera.focal
            expected = g.np.column_stack(
                (
                    (pixels[:, 0] + 0.5 - width / 2.0) / fx,
                    -(pixels[:, 1] + 0.5 - height / 2.0) / fy,
                )
            )
            g.np.testing.assert_allclose(xy, expected, atol=1e-12)

    def test_scaled_copy(self):
        s = g.get_mesh("cycloidal.3DXML")

        s.units = "mm"
        assert s.camera_transform.shape == (4, 4)

        # the camera node should have been removed on copy
        b = s.convert_units("m")
        assert b.camera_transform.shape == (4, 4)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
