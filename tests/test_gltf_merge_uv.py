"""
Regression tests for preserving UV coordinates when multiple glTF
primitives are merged with ``merge_primitives=True``.

The fixture is a fixed in-memory GLB containing one mesh with two
triangle primitives, each with three vertices and its own UV block.
"""

import io
import json
import struct

import numpy as np
import pytest

import trimesh


def _solid_png(color):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (2, 2), tuple(color)).save(buf, format="PNG")
    return buf.getvalue()


def _two_primitive_glb(
    *,
    first_uv=True,
    second_uv=True,
    materials=2,
    textures=True,
    bad_uv_ref=False,
    uv_count_override=None,
):
    """
    Build a GLB with two 3-vertex triangle primitives.

    The two POSITION and TEXCOORD accessors are disjoint, so merging
    stacks them into exactly six vertices.
    """
    vertices = [
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
    ]
    uvs = [
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.array([[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]], dtype=np.float32),
    ]
    index = np.array([0, 1, 2], dtype=np.uint16)

    images = (
        [_solid_png((255, 0, 0, 255)), _solid_png((0, 0, 255, 255))] if textures else []
    )

    chunks = (
        [v.tobytes() for v in vertices]
        + [u.tobytes() for u in uvs]
        + [index.tobytes(), index.tobytes()]
        + images
    )
    blob = b""
    offsets = []
    for chunk in chunks:
        offsets.append(len(blob))
        blob += chunk
        if len(blob) % 4:
            blob += b"\x00" * (4 - len(blob) % 4)

    buffer_views = [
        {"buffer": 0, "byteOffset": offset, "byteLength": len(chunk)}
        for offset, chunk in zip(offsets, chunks)
    ]
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "count": 3,
            "type": "VEC3",
            "min": [0, 0, 0],
            "max": [1, 1, 0],
        },
        {
            "bufferView": 1,
            "componentType": 5126,
            "count": 3,
            "type": "VEC3",
            "min": [2, 0, 0],
            "max": [3, 1, 0],
        },
        {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
        {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC2"},
        {"bufferView": 4, "componentType": 5123, "count": 3, "type": "SCALAR"},
        {"bufferView": 5, "componentType": 5123, "count": 3, "type": "SCALAR"},
    ]
    if uv_count_override is not None:
        accessors[2]["count"] = uv_count_override

    prim0 = {
        "attributes": {"POSITION": 0},
        "indices": 4,
        "material": 0,
    }
    if first_uv:
        prim0["attributes"]["TEXCOORD_0"] = 99 if bad_uv_ref else 2
    prim1 = {
        "attributes": {"POSITION": 1},
        "indices": 5,
        "material": min(1, materials - 1),
    }
    if second_uv:
        prim1["attributes"]["TEXCOORD_0"] = 3

    material_docs = [{} for _ in range(materials)]
    texture_docs = []
    image_docs = []
    if textures:
        for i in range(materials):
            material_docs[i] = {
                "pbrMetallicRoughness": {"baseColorTexture": {"index": i}}
            }
            texture_docs.append({"source": i})
        image_docs = [
            {"bufferView": 6 + i, "mimeType": "image/png"} for i in range(materials)
        ]

    header = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [prim0, prim1]}],
        "materials": material_docs,
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    if texture_docs:
        header["textures"] = texture_docs
        header["images"] = image_docs

    content = json.dumps(header).encode("utf-8")
    content += b" " * (-len(content) % 4)
    total = 12 + 8 + len(content) + 8 + len(blob)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(content), 0x4E4F534A)
        + content
        + struct.pack("<II", len(blob), 0x004E4942)
        + blob
    )


def _load(glb, **kwargs):
    scene = trimesh.load_scene(
        trimesh.util.wrap_as_stream(glb),
        file_type="glb",
        merge_primitives=True,
        **kwargs,
    )
    return next(iter(scene.geometry.values()))


def _flipped(uv):
    # the glTF loader flips V on load
    out = np.array(uv, dtype=np.float64).copy()
    out[:, 1] = 1.0 - out[:, 1]
    return out


def test_merge_primitives_keeps_uvs_aligned_with_vertices():
    glb = _two_primitive_glb()

    # separately each primitive has its own UV block
    separate = trimesh.load_scene(trimesh.util.wrap_as_stream(glb), file_type="glb")
    separate_uv = [m.visual.uv for m in separate.geometry.values()]
    assert len(separate_uv) == 2
    assert all(uv is not None for uv in separate_uv)

    mesh = _load(glb)
    assert len(mesh.vertices) == 6
    assert len(mesh.faces) == 2

    # merged UV rows are exactly the concatenated per-primitive UVs and
    # therefore line up one-to-one with the stacked vertices
    expected = np.vstack(separate_uv)
    assert mesh.visual.uv is not None
    assert mesh.visual.uv.shape == (6, 2)
    np.testing.assert_array_equal(np.asarray(mesh.visual.uv), expected)

    # faces and material slots keep primitive order
    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [3, 4, 5]])
    assert mesh.visual.face_materials == [0, 1]
    assert len(mesh.visual.material.materials) == 2


def test_merge_primitives_texture_sampling_matches_primitive():
    pytest.importorskip("PIL")
    mesh = _load(_two_primitive_glb())
    uv = np.asarray(mesh.visual.uv)
    materials = mesh.visual.material.materials

    first = np.asarray(materials[0].to_color(uv[:3]))[:, :3]
    second = np.asarray(materials[1].to_color(uv[3:]))[:, :3]
    # first texture is solid red, second solid blue
    np.testing.assert_allclose(first, np.tile([255, 0, 0], (3, 1)), atol=2)
    np.testing.assert_allclose(second, np.tile([0, 0, 255], (3, 1)), atol=2)


def test_merge_primitives_uv_glb_roundtrip():
    glb = _two_primitive_glb()
    mesh = _load(glb)
    expected = np.asarray(mesh.visual.uv).copy()

    exported = mesh.export(file_type="glb")
    reloaded = next(
        iter(
            trimesh.load_scene(
                trimesh.util.wrap_as_stream(exported), file_type="glb"
            ).geometry.values()
        )
    )
    assert len(reloaded.vertices) == 6
    assert reloaded.visual.uv is not None
    np.testing.assert_allclose(reloaded.visual.uv, expected, atol=1e-6)


def test_merge_primitives_uv_gltf_directory_roundtrip(tmp_path):
    scene = trimesh.load_scene(
        trimesh.util.wrap_as_stream(_two_primitive_glb()),
        file_type="glb",
        merge_primitives=True,
    )
    expected = np.asarray(next(iter(scene.geometry.values())).visual.uv).copy()

    files = scene.export(file_type="gltf", merge_buffers=True)
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)

    reloaded = trimesh.load(tmp_path / "model.gltf")
    mesh = next(iter(reloaded.geometry.values()))
    assert mesh.visual.uv is not None
    np.testing.assert_allclose(mesh.visual.uv, expected, atol=1e-6)


def test_merge_primitives_one_primitive_without_uv():
    mesh = _load(_two_primitive_glb(second_uv=False))
    # a full length UV block keeps one row per merged vertex and stays
    # export compatible; the UV-less primitive gets zero coordinates
    assert mesh.visual.uv is not None
    assert mesh.visual.uv.shape == (6, 2)
    np.testing.assert_array_equal(np.asarray(mesh.visual.uv)[3:], 0.0)
    np.testing.assert_allclose(
        np.asarray(mesh.visual.uv)[:3],
        _flipped([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    )
    # material/face attribution is unaffected
    assert mesh.visual.face_materials == [0, 1]

    # export still emits a TEXCOORD_0 accessor sized to all six vertices
    exported = mesh.export(file_type="glb")
    reloaded = next(
        iter(
            trimesh.load_scene(
                trimesh.util.wrap_as_stream(exported), file_type="glb"
            ).geometry.values()
        )
    )
    assert reloaded.visual.uv is not None
    assert len(reloaded.visual.uv) == 6


def test_merge_primitives_all_without_uv_is_none():
    mesh = _load(_two_primitive_glb(first_uv=False, second_uv=False))
    assert mesh.visual.uv is None
    assert mesh.visual.face_materials == [0, 1]


def test_merge_primitives_skip_materials_has_no_visual_material():
    mesh = _load(_two_primitive_glb(), skip_materials=True)
    assert not hasattr(mesh.visual, "material")


def test_merge_primitives_corrupt_uv_accessor_raises():
    with pytest.raises((IndexError, KeyError, ValueError)):
        _load(_two_primitive_glb(bad_uv_ref=True))
    with pytest.raises(ValueError):
        _load(_two_primitive_glb(uv_count_override=2))
