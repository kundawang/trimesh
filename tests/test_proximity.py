try:
    from . import generic as g
except BaseException:
    import generic as g


class NearestTest(g.unittest.TestCase):
    def test_naive(self):
        """
        Test the naive nearest point function
        """

        # generate a unit sphere mesh
        sphere = g.trimesh.primitives.Sphere(subdivisions=4)

        # randomly sample surface of a unit sphere, then expand to radius 2.0
        points = g.trimesh.sample.sample_surface_sphere(100) * 2

        # use the triangles from the unit sphere
        triangles = sphere.triangles  # NOQA

        # do the check
        closest, distance, _tid = g.trimesh.proximity.closest_point_naive(sphere, points)

        # the distance from a sphere of radius 1.0 to a sphere of radius 2.0
        # should be pretty darn close to 1.0
        assert (g.np.abs(distance - 1.0) < 0.01).all()

        # the vector for the closest point should be the same as the vector
        # to the query point
        vector = g.trimesh.util.diagonal_dot(closest, points / 2.0)
        assert (g.np.abs(vector - 1.0) < 0.01).all()

    def test_helper(self):
        # just make sure the plumbing returns something
        for mesh in g.get_meshes(2):
            points = (g.random((100, 3)) - 0.5) * 100

            a = mesh.nearest.on_surface(points)
            assert a is not None

            b = mesh.nearest.vertex(points)
            assert b is not None

    def test_nearest_naive(self):
        funs = [
            g.trimesh.proximity.closest_point_naive,
            g.trimesh.proximity.closest_point,
        ]

        data_points = g.deque()
        data_dist = g.deque()

        tic = [g.time.time()]
        for i in funs:
            p, d = self.check_nearest_point_function(i)
            data_points.append(p)
            data_dist.append(d)
            tic.append(g.time.time())

        assert g.np.ptp(data_points, axis=0).max() < g.tol.merge
        assert g.np.ptp(data_dist, axis=0).max() < g.tol.merge

        log_msg = "\n".join(
            f"{i}: {j}s" for i, j in zip([i.__name__ for i in funs], g.np.diff(tic))
        )
        g.log.info("Compared the following nearest point functions:\n" + log_msg)

    def check_nearest_point_function(self, fun):
        # def plot_tri(tri, color='g'):
        #     plottable = g.np.vstack((tri, tri[0]))
        #     plt.plot(plottable[:, 0], plottable[:, 1], color=color)

        def points_on_circle(count):
            theta = g.np.linspace(0, g.np.pi * 2, count + 1)[:count]
            s = g.np.column_stack((theta, [g.np.pi / 2] * count))
            t = g.trimesh.util.spherical_to_vector(s)
            return t

        # generate some pseudorandom triangles
        # use our random to avoid spurious failures
        triangles = g.random((100, 3, 3)) - 0.5
        # put them on- plane
        triangles[:, :, 2] = 0.0

        # make one of the triangles equilaterial
        triangles[-1] = points_on_circle(3)

        # a circle of points surrounding the triangle
        query = points_on_circle(63) * 2
        # set the points up in space
        query[:, 2] = 10
        # a circle of points inside-ish the triangle
        query = g.np.vstack((query, query * 0.1))

        # loop through each triangle
        for triangle in triangles:
            # create a mesh with one triangle
            mesh = g.Trimesh(**g.trimesh.triangles.to_kwargs([triangle]))

            result, result_distance, _result_tid = fun(mesh, query)

            polygon = g.Polygon(triangle[:, 0:2])
            polygon_buffer = polygon.buffer(1e-5)

            # all of the points returned should be on the triangle we're
            # querying
            assert all(polygon_buffer.intersects(g.Point(i)) for i in result[:, 0:2])

            # see what distance shapely thinks the nearest point
            # is for the 2D triangle and the query points
            distance_shapely = g.np.array(
                [polygon.distance(g.Point(i)) for i in query[:, :2]]
            )

            # see what distance our function returned for the nearest point
            distance_ours = ((query[:, :2] - result[:, :2]) ** 2).sum(axis=1) ** 0.5

            # how far was our distance from the one shapely gave
            distance_test = g.np.abs(distance_shapely - distance_ours)  # NOQA

            # we should have calculated the same distance as shapely
            assert g.np.allclose(distance_ours, distance_shapely)

        # now check to make sure closest point doesn't depend on
        # the frame, IE the results should be the same after
        # any rigid transform
        # chop query off to same length as triangles
        assert len(query) > len(triangles)
        query = query[: len(triangles)]
        # run the closest point query as a corresponding query
        close = g.trimesh.triangles.closest_point(triangles=triangles, points=query)
        # distance between closest point and query point
        # this should stay the same regardless of frame
        distance = g.np.linalg.norm(close - query, axis=1)
        for T in g.transforms:
            # transform the query points
            points = g.trimesh.transform_points(query, T)
            # transform the triangles we're checking
            tri = g.trimesh.transform_points(triangles.reshape((-1, 3)), T).reshape(
                (-1, 3, 3)
            )
            # run the closest point check
            check = g.trimesh.triangles.closest_point(triangles=tri, points=points)
            check_distance = g.np.linalg.norm(check - points, axis=1)
            # should be the same in any frame
            assert g.np.allclose(check_distance, distance)

        return result, result_distance

    def test_coplanar_signed_distance(self):
        mesh = g.trimesh.primitives.Box()

        # should be well outside the box but coplanar with a face
        # so the signed distance should be negative
        distance = mesh.nearest.signed_distance([mesh.bounds[0] + [100, 0, 0]])

        assert distance[0] < 0.0

        # constructed so origin is inside but also coplanar with
        # the nearest face
        mesh = g.get_mesh("origin_inside.STL")

        # origin should be inside, so distance should be positive
        distance = mesh.nearest.signed_distance([[0, 0, 0]])

        assert distance[0] > 0.0

    def test_noncoplanar_signed_distance(self):
        mesh = g.trimesh.primitives.Box()

        # should be well outside the box and not coplanar with a face
        # so the signed distance should be negative
        distance = mesh.nearest.signed_distance([mesh.bounds[0] + [100, 100, 100]])

        assert distance[0] < 0.0

    def test_edge_case(self):
        mesh = g.get_mesh("20mm-xyz-cube.stl")
        assert (mesh.nearest.signed_distance([[-51, 4.7, -20.6]]) < 0.0).all()

    def test_acute_edge_case(self):
        # acute tetrahedron with a sharp edge
        vertices = [[-1, 0.5, 0], [1, 0.5, 0], [0, -1, -0.5], [0, -1, 0.5]]
        faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [3, 2, 1]]
        mesh = g.trimesh.Trimesh(vertices, faces)

        # a set of points on a line outside of the tetrahedron
        # their closest surface point is [0, 0.5, 0] on the sharp edge
        # for a point exactly in the middle a closest face is still ambiguous
        # -> take an even number of points
        n = 20
        n += n % 2
        pts = g.np.transpose([g.np.zeros(n), g.np.ones(n), g.np.linspace(-1, 1, n)])

        # the faces facing the points should differ for first and second half of the set
        # check their indices for inequality
        faceIdxsA, faceIdxsB = g.np.split(mesh.nearest.on_surface(pts)[-1], 2)
        assert (
            g.np.all(faceIdxsA == faceIdxsA[0])
            and g.np.all(faceIdxsB == faceIdxsB[0])
            and faceIdxsA[0] != faceIdxsB[0]
        )

    def test_candidates(self):
        mesh = g.trimesh.creation.random_soup(2000)
        points = g.random((2000, 3))
        g.trimesh.proximity.nearby_faces(mesh=mesh, points=points)

    def test_candidates_rtree_versions(self):
        mesh = g.trimesh.Trimesh(
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces=[[0, 1, 2], [0, 1, 2]],
            process=False,
        )
        points = g.np.array([[0.2, 0.2, 0.5], [0.2, 0.2, 0.5]], dtype=g.np.float32)

        expected = g.trimesh.proximity.nearby_faces(mesh, points)
        assert g.np.allclose(expected, [[0, 1], [0, 1]])

    def test_candidates_empty(self):
        # an empty (0, 3) query set must return an empty list of candidates
        mesh = g.trimesh.creation.box()
        candidates = g.trimesh.proximity.nearby_faces(
            mesh=mesh, points=g.np.zeros((0, 3))
        )
        assert len(candidates) == 0

    def test_returns_correct_point_in_ambiguous_cases(self):
        mesh = g.trimesh.Trimesh(
            vertices=[
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            faces=[[0, 1, 2], [0, 1, 3]],
            process=False,
        )
        # Query point is closer to face 0 but lies in face 1 normal direction
        closest, _, closest_face_index = g.trimesh.proximity.closest_point(
            mesh, [[-0.25 - 1e-9, 0.0, -0.25]]
        )
        g.np.testing.assert_almost_equal(closest[0], [0.0, 0.0, -0.25])
        self.assertEqual(closest_face_index[0], 1)

    def test_unreferenced_vertex(self):
        # check to see that meshes with unreferenced vertices
        # return correct values and ignore the unreferenced points
        query_point = [-1.0, -1.0, -1.0]
        mesh = g.trimesh.Trimesh(
            vertices=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-0.5, -0.5, -0.5],
            ],
            faces=[[0, 1, 2]],
            process=False,
        )

        proximity_query = g.trimesh.proximity.ProximityQuery(mesh)
        q = proximity_query.on_surface([query_point])
        assert len(q) == 3
        assert all(len(i) == 1 for i in q)


class SceneClosestPointTest(g.unittest.TestCase):
    @staticmethod
    def _boxes_scene():
        # one shared unit-box geometry instanced three times:
        # box_a at x=0, box_b 3 units away at x=3, and a non-uniformly
        # scaled instance above the origin
        box = g.trimesh.creation.box()
        scene = g.trimesh.Scene()
        scene.add_geometry(
            box,
            geom_name="box",
            node_name="box_a",
            transform=g.trimesh.transformations.translation_matrix([0.0, 0.0, 0.0]),
        )
        # second node referencing the *same* geometry entry
        scene.graph.update(
            frame_to="box_b",
            geometry="box",
            matrix=g.trimesh.transformations.translation_matrix([3.0, 0.0, 0.0]),
        )
        scaled = g.trimesh.transformations.translation_matrix(
            [0.0, 4.0, 0.0]
        ) @ g.trimesh.transformations.scale_matrix(3.0)
        scene.graph.update(frame_to="box_scaled", geometry="box", matrix=scaled)
        return scene

    def test_two_boxes_and_scaled_instance(self):
        # requirement: hit node + analytic distances for two boxes
        # 3 units apart and a non-uniform scaled instance
        scene = self._boxes_scene()
        points = g.np.array(
            [
                [0.75, 0.0, 0.0],  # inside the gap, nearer box_a face x=0.5
                [2.25, 0.0, 0.0],  # nearer box_b face x=2.5
                [2.0, 4.0, 0.0],  # scaled instance face x=1.5 (scale 3)
            ]
        )
        result = g.trimesh.proximity.closest_point_scene(scene, points)

        g.np.testing.assert_array_equal(
            result.node_name, ["box_a", "box_b", "box_scaled"]
        )
        g.np.testing.assert_allclose(result.distance, [0.25, 0.25, 0.5], atol=1e-10)
        g.np.testing.assert_allclose(
            result.closest,
            [[0.5, 0.0, 0.0], [2.5, 0.0, 0.0], [1.5, 4.0, 0.0]],
            atol=1e-10,
        )
        # geometry attribution and valid source face indices
        g.np.testing.assert_array_equal(result.geometry_name, ["box"] * 3)
        assert result.face_index.shape == (3,)
        assert (result.face_index >= 0).all()
        assert (result.face_index < len(scene.geometry["box"].faces)).all()

    def test_cross_check_independent_instance_queries(self):
        # cross-check the scene query against an independent
        # closest_point call for every baked instance
        scene = self._boxes_scene()
        points = g.random((40, 3)) * 9.0 - 3.0

        result = scene.closest_point(points)

        independent = {}
        for node in scene.graph.nodes_geometry:
            matrix, geometry_name = scene.graph[node]
            baked = scene.geometry[geometry_name].copy()
            baked.apply_transform(matrix)
            independent[node] = (
                geometry_name,
                baked,
                g.trimesh.proximity.closest_point(baked, points),
            )

        for i in range(len(points)):
            distances = {node: data[2][1][i] for node, data in independent.items()}
            best_distance = min(distances.values())
            g.np.testing.assert_allclose(result.distance[i], best_distance, atol=1e-10)

            node = result.node_name[i]
            geometry_name, _baked, (close, distance, face_id) = independent[node]
            assert result.geometry_name[i] == geometry_name
            g.np.testing.assert_allclose(distance[i], result.distance[i], atol=1e-10)
            g.np.testing.assert_allclose(close[i], result.closest[i], atol=1e-8)
            assert face_id[i] == result.face_index[i]

    def test_random_transforms_and_naive_reference(self):
        # rotation, translation and non-uniform scale must produce exact
        # world distances matching a brute-force triangle query
        rng = g.np.random.default_rng(7)
        meshes = [
            g.trimesh.creation.box(extents=rng.uniform(0.5, 2.0, 3)),
            g.trimesh.creation.icosphere(subdivisions=2, radius=0.7),
        ]
        scene = g.trimesh.Scene()
        for i, mesh in enumerate(meshes):
            affine = (
                g.trimesh.transformations.translation_matrix(rng.uniform(-5, 5, 3))
                @ g.trimesh.transformations.rotation_matrix(
                    float(rng.uniform(0, g.np.pi)), rng.normal(size=3)
                )
                @ g.np.diag([*rng.uniform(0.4, 2.5, 3), 1.0])
            )
            scene.add_geometry(
                mesh, geom_name=f"g{i}", node_name=f"n{i}", transform=affine
            )
        # second non-uniformly scaled instance of g0
        affine = g.trimesh.transformations.translation_matrix(
            [2.0, -3.0, 1.0]
        ) @ g.np.diag([2.2, 0.6, 1.4, 1.0])
        scene.graph.update(frame_to="n0_inst", geometry="g0", matrix=affine)

        points = rng.uniform(-7, 7, (50, 3))
        result = g.trimesh.proximity.closest_point_scene(scene, points)

        naive_min = g.np.full(len(points), g.np.inf)
        for node in scene.graph.nodes_geometry:
            matrix, geometry_name = scene.graph[node]
            baked = scene.geometry[geometry_name].copy()
            baked.apply_transform(matrix)
            _close, distance, _face = g.trimesh.proximity.closest_point_naive(
                baked, points
            )
            naive_min = g.np.minimum(naive_min, distance)

        g.np.testing.assert_allclose(result.distance, naive_min, atol=1e-8)
        assert g.np.isfinite(result.distance).all()
        assert {str(n) for n in result.node_name}.issubset({"n0", "n1", "n0_inst"})

    def test_tie_is_deterministic(self):
        # symmetric boxes one unit either side of the origin tie at d=1;
        # lexicographic node order decides and repeat calls agree
        box = g.trimesh.creation.box()
        scene = g.trimesh.Scene()
        # add in reverse name order to prove ordering is not insertion order
        scene.add_geometry(
            box,
            geom_name="box",
            node_name="right",
            transform=g.trimesh.transformations.translation_matrix([1.5, 0, 0]),
        )
        scene.graph.update(
            frame_to="left",
            geometry="box",
            matrix=g.trimesh.transformations.translation_matrix([-1.5, 0, 0]),
        )
        first = g.trimesh.proximity.closest_point_scene(scene, [[0.0, 0.0, 0.0]])
        second = g.trimesh.proximity.closest_point_scene(scene, [[0.0, 0.0, 0.0]])
        assert first.node_name[0] == "left"
        g.np.testing.assert_array_equal(first.node_name, second.node_name)
        g.np.testing.assert_allclose(first.distance, [1.0])

    def test_empty_scene_convention(self):
        scene = g.trimesh.Scene()
        result = g.trimesh.proximity.closest_point_scene(scene, [[1.0, 2.0, 3.0]])
        g.np.testing.assert_array_equal(result.distance, [g.np.inf])
        assert result.node_name[0] is None
        assert result.geometry_name[0] is None
        g.np.testing.assert_array_equal(result.face_index, [-1])
        assert g.np.isnan(result.closest).all()

        with g.pytest.raises(ValueError):
            g.trimesh.proximity.closest_point_scene(
                scene, [[0.0, 0.0, 0.0]], on_empty="raise"
            )

        # an empty query batch is an empty result even on an empty scene
        empty = g.trimesh.proximity.closest_point_scene(scene, g.np.zeros((0, 3)))
        assert len(empty.distance) == 0
        assert empty.closest.shape == (0, 3)

    def test_non_mesh_geometry_is_skipped(self):
        points_cloud = g.trimesh.PointCloud(g.random((20, 3)))
        # point cloud only: same sentinel convention as an empty scene
        cloud_scene = g.trimesh.Scene(points_cloud)
        result = g.trimesh.proximity.closest_point_scene(cloud_scene, [[0.0, 0.0, 0.0]])
        assert g.np.isinf(result.distance).all()

        # mixed scene still finds the mesh and ignores the point cloud
        box = g.trimesh.creation.box()
        scene = g.trimesh.Scene()
        scene.add_geometry(points_cloud, node_name="cloud")
        scene.add_geometry(box, node_name="box")
        mixed = g.trimesh.proximity.closest_point_scene(scene, [[0.6, 0.0, 0.0]])
        g.np.testing.assert_array_equal(mixed.node_name, ["box"])
        g.np.testing.assert_allclose(mixed.distance, [0.1], atol=1e-10)

    def test_accessors_dispatch_and_aliases(self):
        scene = self._boxes_scene()
        points = [[0.75, 0.0, 0.0]]

        via_function = g.trimesh.proximity.closest_point_scene(scene, points)
        via_dispatch = g.trimesh.proximity.closest_point(scene, points)
        via_nearest = scene.nearest.on_surface(points)
        via_method = scene.closest_point(points)
        via_toplevel = g.trimesh.closest_point_scene(scene, points)

        for other in (via_dispatch, via_nearest, via_method, via_toplevel):
            g.np.testing.assert_array_equal(other.node_name, via_function.node_name)
            g.np.testing.assert_allclose(other.distance, via_function.distance)

        # canonical five-way unpack and attribute / dict aliases
        closest, distance, geometry, node, face = via_function
        assert closest.shape == (1, 3)
        g.np.testing.assert_array_equal(node, ["box_a"])
        assert via_function["node_name"][0] == "box_a"
        assert via_function["triangle_id"][0] == face[0]
        assert via_function.distances[0] == distance[0]
        assert via_function.get("geometry_name")[0] == geometry[0]
        assert "node" in via_function

    def test_single_mesh_api_and_cache_unchanged(self):
        mesh = g.trimesh.creation.box()

        # classic single-mesh API still returns exactly three arrays
        classic = g.trimesh.proximity.closest_point(mesh, [[0.75, 0.0, 0.0]])
        assert len(classic) == 3
        close, distance, face_id = classic
        assert close.shape == (1, 3)
        assert face_id.shape == (1,)
        g.np.testing.assert_allclose(distance, [0.25])

        # let the single-mesh query populate its caches, then snapshot them
        keys_before = set(mesh._cache.cache.keys())
        hash_before = hash(mesh)

        # running a scene query must not mutate the source mesh or its caches
        scene = g.trimesh.Scene([mesh])
        g.trimesh.proximity.closest_point_scene(scene, [[2.0, 0.0, 0.0]])
        assert hash(mesh) == hash_before
        assert set(mesh._cache.cache.keys()) == keys_before


def test_closest_point_memory_error():
    # the exact query that got us SIGKILLed: points deep inside a closed
    # mesh have most faces as candidates so it must raise instead
    mesh = g.trimesh.creation.icosphere(subdivisions=7)
    with g.RandomSeed() as random:
        points = random.random((100000, 3)) - 0.5
    # our own check rather than a failed numpy allocation under a `ulimit`
    with g.pytest.raises(MemoryError, match="_MEMORY_LIMIT"):
        mesh.nearest.on_surface(points)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
