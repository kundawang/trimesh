"""
Tests for the opt-in export preflight / support diagnostics
(:func:`trimesh.exchange.export.check_export`) and strict export mode.

The checks are driven by the actual visual properties of each mesh rather
than the file extension alone.
"""

import io

import numpy as np
import pytest

import trimesh
from trimesh.exchange.export import ExportPreflightError, check_export
from trimesh.visual import ColorVisuals, TextureVisuals

VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
FACES = np.array([[0, 1, 2]])
UV = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
VERTEX_COLORS = np.array(
    [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8
)
FACE_COLORS = np.array([[10, 20, 30, 255]], dtype=np.uint8)


def _uv_mesh():
    mesh = trimesh.Trimesh(vertices=VERTICES, faces=FACES, process=False)
    mesh.visual = TextureVisuals(uv=UV.copy())
    return mesh


def _vertex_color_mesh():
    mesh = trimesh.Trimesh(vertices=VERTICES, faces=FACES, process=False)
    mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=VERTEX_COLORS.copy())
    return mesh


def _face_color_mesh():
    mesh = trimesh.Trimesh(vertices=VERTICES, faces=FACES, process=False)
    mesh.visual = ColorVisuals(mesh=mesh, face_colors=FACE_COLORS.copy())
    return mesh


def _roundtrip(blob, file_type):
    if isinstance(blob, str):
        blob = blob.encode()
    return trimesh.load(
        file_obj=trimesh.util.wrap_as_stream(blob),
        file_type=file_type,
        process=False,
    )


def test_report_covers_builtin_attributes():
    report = check_export(trimesh.creation.box(), "stl")
    assert set(report.keys()) == {"uv", "vertex_colors", "face_colors"}
    for name in ("uv", "vertex_colors", "face_colors"):
        entry = report[name]
        assert entry.status in {
            "supported",
            "format_unsupported",
            "exporter_unsupported",
            "not_present",
            "unknown",
        }
        assert isinstance(entry.present, bool)


def test_uv_triangle_stl_reports_loss_but_ply_obj_do_not():
    mesh = _uv_mesh()

    stl = check_export(mesh, "stl")
    assert stl["uv"].present is True
    assert stl["uv"] == "format_unsupported"
    assert not stl.compatible
    assert "uv" in stl.unsupported_attributes

    for file_type in ("ply", "obj"):
        report = check_export(mesh, file_type)
        assert report["uv"].present is True
        assert report["uv"] == "supported"
        assert report.compatible
        assert "uv" not in report.unsupported_attributes


def test_uv_roundtrip_ply_obj():
    mesh = _uv_mesh()
    for file_type in ("ply", "obj"):
        loaded = _roundtrip(mesh.export(file_type=file_type), file_type)
        assert loaded.visual.uv is not None
        assert np.asarray(loaded.visual.uv).shape[1] == 2


def test_plain_mesh_has_no_losses():
    mesh = trimesh.creation.box()
    for file_type in ("stl", "ply", "obj"):
        report = check_export(mesh, file_type)
        assert report.compatible
        assert report.unsupported_attributes == []


def test_vertex_color_support_and_roundtrip():
    mesh = _vertex_color_mesh()
    assert not check_export(mesh, "stl").compatible
    for file_type in ("ply", "obj"):
        report = check_export(mesh, file_type)
        assert report["vertex_colors"] == "supported"
        assert report.compatible
        loaded = _roundtrip(mesh.export(file_type=file_type), file_type)
        assert np.array_equal(
            loaded.visual.vertex_colors[:, :3], VERTEX_COLORS[:, :3]
        )


def test_face_color_ply_support_and_roundtrip():
    mesh = _face_color_mesh()
    report = check_export(mesh, "ply")
    assert report["face_colors"] == "supported"
    assert report.compatible
    loaded = _roundtrip(mesh.export(file_type="ply"), "ply")
    assert loaded.visual.kind == "face"
    assert np.array_equal(loaded.visual.face_colors, FACE_COLORS)
    # STL cannot keep face colors
    assert check_export(mesh, "stl")["face_colors"] != "supported"


def test_combined_required_attributes():
    report = check_export(
        _uv_mesh(),
        "stl",
        required_attributes=["uv", "vertex_colors", "face_colors"],
    )
    assert not report.compatible
    assert set(report.unsupported_attributes) == {
        "uv",
        "vertex_colors",
        "face_colors",
    }


def test_issue_example_api():
    report = trimesh.exchange.export.check_export(
        trimesh.creation.box(),
        file_type="stl",
        required_attributes=["uv", "vertex_colors"],
    )
    assert not report.compatible
    assert "uv" in report.unsupported_attributes
    assert "vertex_colors" in report.unsupported_attributes


def test_strict_raises_before_creating_path(tmp_path):
    path = tmp_path / "textured.stl"
    with pytest.raises((ValueError, ExportPreflightError)):
        _uv_mesh().export(str(path), strict=True)
    assert not path.exists()


def test_strict_raises_before_writing_file_object():
    stream = io.BytesIO()
    with pytest.raises(ValueError):
        _uv_mesh().export(stream, file_type="stl", strict=True)
    assert stream.getvalue() == b""


def test_strict_allows_supported_exports(tmp_path):
    for file_type in ("ply", "obj"):
        path = tmp_path / f"textured.{file_type}"
        _uv_mesh().export(str(path), strict=True)
        assert path.exists()
        assert path.stat().st_size > 0
    # a visual-property-free mesh stays strict-compatible even for STL
    trimesh.creation.box().export(str(tmp_path / "plain.stl"), strict=True)


def test_default_export_bytes_unchanged():
    uv_mesh = _uv_mesh()
    for file_type in ("stl", "stl_ascii", "ply", "obj"):
        default = uv_mesh.export(file_type=file_type)
        non_strict = uv_mesh.export(file_type=file_type, strict=False)
        assert default == non_strict


def test_method_and_top_level_entry_points():
    mesh = _uv_mesh()
    assert mesh.export_preflight("stl").compatible is False
    assert mesh.check_export("ply").compatible is True
    assert trimesh.check_export(mesh, "obj").compatible is True
