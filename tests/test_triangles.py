try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.triangles import barycentric_to_points, points_to_barycentric


class TrianglesTest(g.unittest.TestCase):
    def test_barycentric(self):
        for m in g.get_meshes(4):
            # a simple test which gets the barycentric coordinate at each of the three
            # vertices, checks to make sure the barycentric is [1,0,0] for the vertex
            # and then converts back to cartesian and makes sure the original points
            #  are the same as the conversion and back
            for method in ["cross", "cramer"]:
                for i in range(3):
                    barycentric = points_to_barycentric(
                        m.triangles, m.triangles[:, i], method=method
                    )
                    assert (
                        g.np.abs(barycentric - g.np.roll([1.0, 0, 0], i)) < 1e-8
                    ).all()

                    points = barycentric_to_points(m.triangles, barycentric)
                    assert (g.np.abs(points - m.triangles[:, i]) < 1e-8).all()

    def test_barycentric_2D(self):
        # points_to_barycentric should work in 2D and 3D
        tri = g.np.array([[[0, 0], [1, 0], [0, 1]]], dtype=g.np.float64)
        points = g.np.array([[0.25, 0.25]])
        b = points_to_barycentric(tri, points)
        assert ((b > 0) & (b < 1.0)).all()

        r = barycentric_to_points(tri, b)

        assert g.np.allclose(r, points)

    def test_closest(self):
        closest = g.trimesh.triangles.closest_point(
            triangles=g.data["triangles"]["triangles"],
            points=g.data["triangles"]["points"],
        )

        comparison = (closest - g.data["triangles"]["closest"]).all()

        assert (comparison < 1e-8).all()
        g.log.info("finished closest check on %d triangles", len(closest))

    def test_closest_obtuse(self):
        # simple triangle in the xy-plane with an obtuse corner at vertex A
        ABC = g.np.float32([[0, 0, 0], [2, 0, 0], [-2, 1, 0]])
        D = g.np.float32([1, -1, 0])

        # ground truth: closest point from D is the center of the AB edge:
        # (1,0,0)
        gt_closest = g.np.float32([1, 0, 0])
        tm_closest = g.trimesh.triangles.closest_point([ABC], [D])[0]

        assert g.np.linalg.norm(gt_closest - tm_closest) < g.tol.merge

        # create a circle of points around the triangle
        # with a radius so that all points are outside of the triangle
        radius = 3
        nPtsOnCircle = 100
        alphas = g.np.linspace(
            g.np.pi / nPtsOnCircle, g.np.pi * 2 - g.np.pi / nPtsOnCircle, nPtsOnCircle
        )
        ptsOnCircle = (
            g.np.transpose([g.np.cos(alphas), g.np.sin(alphas), g.np.zeros(nPtsOnCircle)])
            * radius
        )

        def norm(v):
            return g.np.sqrt(g.np.einsum("...i,...i", v, v))

        def distToLine(o, v, p):
            return norm((o - p) - g.np.dot(o - p, v) * v)

        def distPointToEdge(U, V, P):  # edge [U, V], point P
            UtoV = V - U
            UtoP = P - U
            VtoP = P - V

            if g.np.dot(UtoV, UtoP) <= 0:
                # P is 'behind' U
                return norm(UtoP)
            elif g.np.dot(-UtoV, VtoP) <= 0:
                # P is 'behind' V
                return norm(VtoP)
            else:
                # P is 'between' U and V
                return distToLine(U, UtoV / norm(UtoV), P)

        # get closest points from trimesh and compute distances to the circle
        # points
        tm_dists = norm(
            ptsOnCircle
            - g.trimesh.triangles.closest_point([ABC] * nPtsOnCircle, ptsOnCircle)
        )
        # compute naive point-to-edge distances for all points and take the min of
        # the three edges
        gt_dists = g.np.float32(
            [
                [distPointToEdge(ABC[i], ABC[(i + 1) % 3], pt) for i in range(3)]
                for pt in ptsOnCircle
            ]
        ).min(axis=1)
        diff_dists = tm_dists - gt_dists

        assert g.np.dot(diff_dists, diff_dists) < g.tol.merge

    def test_closest_degenerate(self):
        # Degenerate faces reduce to a segment or a single point.
        triangles = g.np.array(
            [
                [[0, 0, 0], [0, 0, 0], [2, 0, 0]],
                [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                [[0, 0, 0], [0, 0, 0], [0, 2, 0]],
                [[0, 0, 0], [2, 0, 0], [2, 0, 0]],
                [[0, 0, 0], [2, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                [[0, 0, 0], [2, 0, 0], [0, 2, 0]],
            ],
            dtype=g.np.float64,
        )
        points = g.np.array(
            [
                [1, 1, 0],
                [5, 2, 3],
                [0, 3, 1],
                [1, 1, 0],
                [1, 1, 0],
                [1.5, 1, 0],
                [0.5, 0.5, 1],
            ]
        )
        expected = g.np.array(
            [
                [1, 0, 0],
                [1, 2, 3],
                [0, 2, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1.5, 0, 0],
                [0.5, 0.5, 0],
            ]
        )

        with g.np.errstate(divide="raise", invalid="raise"):
            closest = g.trimesh.triangles.closest_point(triangles, points)
        g.np.testing.assert_allclose(closest, expected)

    def test_closest_scale_invariant(self):
        # The classification tolerance must not depend on triangle size:
        # a triangle with 1e-4 edges used to push an interior point onto
        # an edge because the region tests were compared against an
        # absolute `tol.zero`.
        triangles = g.np.array(
            [[[0, 0, 0], [1e-4, 0, 0], [0, 1e-4, 0]]], dtype=g.np.float64
        )
        # strictly interior, edge-near and vertex-near queries for the
        # same triangle, all in the original small units
        points = g.np.array(
            [
                [3e-5, 3e-5, 0],  # inside the face region
                [1e-4 - 1e-7, 3e-5, 0],  # just inside edge AB
                [-1e-7, 1e-7, 0.0],  # just outside vertex A
            ]
        )
        # one triangle corresponds to each query point
        triangles = g.np.broadcast_to(triangles, (len(points), 3, 3)).copy()

        # reference result at the original scale
        reference = g.trimesh.triangles.closest_point(triangles, points)
        reference_distance = g.np.linalg.norm(points - reference, axis=1)

        # the interior point must stay interior: result equals the query
        # projected onto the z=0 plane, i.e. the point itself
        g.np.testing.assert_allclose(reference[0], points[0], atol=1e-12)
        assert reference_distance[0] < 1e-12

        # check a wide range of uniform scales, including tiny and huge
        for scale in [1e-8, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e8]:
            scaled = g.trimesh.triangles.closest_point(triangles * scale, points * scale)
            # the result should scale exactly with the triangle
            g.np.testing.assert_allclose(
                scaled, reference * scale, rtol=1e-9, atol=1e-12 * scale
            )
            # and so should the distance
            g.np.testing.assert_allclose(
                g.np.linalg.norm(points * scale - scaled, axis=1),
                reference_distance * scale,
                rtol=1e-9,
                atol=1e-12 * scale,
            )
            # never produce NaN regardless of scale
            assert g.np.isfinite(scaled).all()

    def test_closest_scale_invariant_degenerate(self):
        # Degenerate triangles must remain deterministic and NaN-free
        # when they are very small (their extent sets a zero scale).
        triangles = g.np.array(
            [
                [[0, 0, 0], [0, 0, 0], [2e-4, 0, 0]],  # AB collapsed to a point
                [[1e-4, 2e-4, 3e-4]] * 3,  # a single point
                [[0, 0, 0], [2e-4, 0, 0], [2e-4, 0, 0]],  # BC collapsed
                [[0, 0, 0], [1e-4, 0, 0], [2e-4, 0, 0]],  # collinear
            ],
            dtype=g.np.float64,
        )
        points = g.np.array(
            [
                [1e-4, 1e-4, 0],
                [5e-4, 2e-4, 3e-4],
                [1e-4, 1e-4, 0],
                [1.5e-4, 1e-4, 0],
            ]
        )
        with g.np.errstate(divide="raise", invalid="raise"):
            closest = g.trimesh.triangles.closest_point(triangles, points)
        assert g.np.isfinite(closest).all()

        # the result for a collapsed triangle is a point on its segment
        # or itself; verify against a uniformly scaled copy is exact
        for scale in [1e-6, 1e-4, 1.0, 1e4, 1e6]:
            scaled = g.trimesh.triangles.closest_point(triangles * scale, points * scale)
            g.np.testing.assert_allclose(
                scaled, closest * scale, rtol=1e-9, atol=1e-12 * scale
            )

    def test_degenerate(self):
        tri = [
            [[0, 0, 0], [1, 0, 0], [-0.5, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [10, 10, 0]],
            [[0, 0, 0], [0, 0, 2], [0, 0, 2.2]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ]

        tri_gt = [False, False, False, True]

        r = g.trimesh.triangles.nondegenerate(tri)
        assert len(r) == len(tri)
        assert (r == tri_gt).all()

    def test_angles(self):
        # a zero- area triangle
        tris = g.np.array([[[0, 0, 0], [1, 0, 0], [1, 0, 0]]], dtype=g.np.float64)
        angles = g.trimesh.triangles.angles(tris)
        # degenerate angles should be zero, not NaN
        g.log.debug(angles)
        assert g.np.allclose(angles, 0.0)

        # an equilateral triangle
        tris = g.np.array(
            [[[-1, 0, 0], [1, 0, 0], [0, g.np.sqrt(3), 0]]], dtype=g.np.float64
        )
        angles = g.trimesh.triangles.angles(tris)
        # degenerate angles should be zero, not NaN
        assert g.np.allclose(angles, g.np.radians(60))

        # an equilateral triangle transformed into space
        tris = g.trimesh.transform_points(
            g.np.array([[-1, 0, 0], [1, 0, 0], [0, g.np.sqrt(3), 0]], dtype=g.np.float64),
            next(g.random_transforms(1, translate=0.0)),
        ).reshape((-1, 3, 3))
        angles = g.trimesh.triangles.angles(tris)
        # all angles should be 60 degrees
        assert g.np.allclose(angles, g.np.radians(60))

        # an 3-4-5 right triangle
        tris = g.trimesh.transform_points(
            g.np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]], dtype=g.np.float64),
            next(g.random_transforms(1, translate=0.0)),
        ).reshape((-1, 3, 3))
        # get angles
        angles = g.trimesh.triangles.angles(tris)
        # make sure they match a 3-4-5
        assert g.np.allclose(
            g.np.sort(angles.ravel()),
            [g.np.arcsin(3.0 / 5), g.np.arcsin(4.0 / 5), g.np.pi / 2],
        )


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
