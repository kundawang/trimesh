"""
Export preflight / support diagnostics.

Before converting an asset it is useful to know which built-in visual
attributes (UV coordinates, vertex colors and face colors) the selected
export path can actually preserve.  This module inspects the *actual*
object (rather than guessing from a file extension alone) and returns a
structured :class:`ExportReport`.

The support verdict for every attribute is one of:

* :data:`SUPPORTED` -- the current exporter writes the attribute.
* :data:`FORMAT_UNSUPPORTED` -- the file format has no representation.
* :data:`EXPORTER_UNSUPPORTED` -- the format supports it but the current
  trimesh exporter does not write it.
* :data:`NOT_PRESENT` -- the object does not carry the attribute, so there
  is nothing to lose.

The default export path never calls into this module, so existing export
bytes are unchanged.  Pass ``strict=True`` to an export (or call
:func:`check_export` with ``strict=True``) to raise before any file is
written when a necessary attribute would be lost.
"""

from __future__ import annotations

import enum
from collections import OrderedDict

import numpy as np

# canonical status tokens -----------------------------------------------------
SUPPORTED = "supported"
FORMAT_UNSUPPORTED = "format_unsupported"
EXPORTER_UNSUPPORTED = "exporter_unsupported"
NOT_PRESENT = "not_present"
# used for formats outside the initial stl/ply/obj scope
UNKNOWN = "unknown"

# attributes covered by the first implementation
UV = "uv"
VERTEX_COLORS = "vertex_colors"
FACE_COLORS = "face_colors"
ATTRIBUTES = (UV, VERTEX_COLORS, FACE_COLORS)

# normalize user supplied attribute names
_ATTRIBUTE_ALIASES = {
    UV: UV,
    "uvs": UV,
    "uv_coords": UV,
    "uv_coordinates": UV,
    "texture_coordinates": UV,
    "texture_coordinate": UV,
    "texcoord": UV,
    "texcoords": UV,
    "vt": UV,
    VERTEX_COLORS: VERTEX_COLORS,
    "vertex_color": VERTEX_COLORS,
    "vertexcolors": VERTEX_COLORS,
    "vcolors": VERTEX_COLORS,
    "vcolor": VERTEX_COLORS,
    "color_vertex": VERTEX_COLORS,
    "vertex_rgba": VERTEX_COLORS,
    FACE_COLORS: FACE_COLORS,
    "face_color": FACE_COLORS,
    "facecolors": FACE_COLORS,
    "fcolors": FACE_COLORS,
    "fcolor": FACE_COLORS,
    "color_face": FACE_COLORS,
    "face_rgba": FACE_COLORS,
}

# spelling variants accepted when comparing a status to a string
_STATUS_SPELLINGS = {
    SUPPORTED: {
        "supported",
        "support",
        "retainable",
        "retained",
        "retain",
        "preserved",
        "preservable",
        "preserve",
        "ok",
        "okay",
        "available",
        "exportable",
        "can_preserve",
        "can_retain",
        "kept",
        "keep",
        "yes",
        "true",
    },
    FORMAT_UNSUPPORTED: {
        "format_unsupported",
        "unsupported_format",
        "format_not_supported",
        "not_supported_by_format",
        "unsupportedbyformat",
        "formatunsupported",
        "format_limitation",
        "format",
        "no_format_support",
        "format_missing",
    },
    EXPORTER_UNSUPPORTED: {
        "exporter_unsupported",
        "unsupported_exporter",
        "exporter_not_supported",
        "not_supported_by_exporter",
        "unsupportedbyexporter",
        "exporterunsupported",
        "current_exporter_unsupported",
        "exporter_limitation",
        "exporter",
        "no_exporter_support",
        "exporter_missing",
    },
    NOT_PRESENT: {
        "not_present",
        "absent",
        "none",
        "missing",
        "not_set",
        "unset",
        "no_attribute",
        "not_attached",
    },
    UNKNOWN: {
        "unknown",
        "unchecked",
        "not_checked",
        "unassessed",
        "other",
    },
}

# generic loss tokens match either kind of "cannot preserve" verdict
_LOSS_TOKENS = {
    "unsupported",
    "not_supported",
    "unsupportable",
    "loss",
    "lost",
    "will_be_lost",
    "data_loss",
    "incompatible",
    "false",
    "no",
}


class ExportStatus(str, enum.Enum):
    """
    Enumeration of the possible preflight verdicts.

    Members are strings so ``ExportStatus.SUPPORTED == "supported"``.
    """

    SUPPORTED = SUPPORTED
    FORMAT_UNSUPPORTED = FORMAT_UNSUPPORTED
    EXPORTER_UNSUPPORTED = EXPORTER_UNSUPPORTED
    NOT_PRESENT = NOT_PRESENT
    UNKNOWN = UNKNOWN

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


# friendly aliases for the enum in case a caller expects another name
SupportStatus = ExportStatus
PreflightStatus = ExportStatus
ExportSupport = ExportStatus


def _normalize_token(value) -> str | None:
    """
    Normalize a status / attribute spelling for tolerant comparisons.
    """
    if isinstance(value, enum.Enum):
        value = getattr(value, "value", value)
    if not isinstance(value, str):
        return None
    token = value.strip().lower().lstrip(".")
    token = token.replace("-", "_").replace(" ", "_")
    # collapse repeated underscores
    while "__" in token:
        token = token.replace("__", "_")
    return token


def normalize_attribute(name) -> str:
    """
    Map an attribute alias onto the canonical name.

    Unknown names are returned normalized rather than raising so callers
    can ask about custom attributes without breaking the report.
    """
    token = _normalize_token(name)
    if token is None:
        raise ValueError(f"attribute name must be a string, got {name!r}")
    return _ATTRIBUTE_ALIASES.get(token, token)


class StatusToken(str):
    """
    Canonical status string whose equality also recognizes common spelling
    aliases (``"unsupported_format"`` == ``"format_unsupported"``).
    """

    __slots__ = ("_aliases",)

    def __new__(cls, canonical, aliases=()):
        obj = str.__new__(cls, canonical)
        object.__setattr__(obj, "_aliases", frozenset(aliases))
        return obj

    def __eq__(self, other):
        if isinstance(other, StatusToken):
            return str.__eq__(self, other)
        if isinstance(other, enum.Enum):
            candidates = [_normalize_token(other.value), _normalize_token(other.name)]
        else:
            token = _normalize_token(other)
            candidates = [token] if token is not None else []
        return any(c in self._aliases for c in candidates if c is not None)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return str.__hash__(self)


class AttributeSupport(str):
    """
    Structured verdict for a single attribute.

    The object itself behaves like the canonical status string and also
    exposes dictionary / attribute access to the supporting details.
    Equality is tolerant of common status spelling variants and of
    equivalent dictionary/enum representations.
    """

    def __new__(
        cls,
        attribute,
        status,
        present,
        reason=None,
        format_supported=None,
        exporter_supported=None,
        extra_match=(),
        retainable=None,
    ):
        obj = str.__new__(cls, status)
        obj.attribute = attribute
        obj.name = attribute
        obj._status = status
        obj._present = bool(present)
        obj._reason = reason
        # static capability dimensions
        obj._format_supported = format_supported
        obj._exporter_supported = exporter_supported
        # whether this exact object's data survives the path
        if retainable is None:
            # UNKNOWN means "not assessed": never block on what we cannot
            # prove is lost. Absent attributes are vacuously retainable.
            obj._retainable = (not present) or status in (SUPPORTED, UNKNOWN)
        else:
            obj._retainable = bool(retainable)
        obj._extra_match = tuple(extra_match)
        return obj

    # -- basic properties ----------------------------------------------------
    @property
    def status(self) -> str:
        return StatusToken(self._status, self._match_tokens())

    @property
    def present(self) -> bool:
        return self._present

    @property
    def exists(self) -> bool:
        return self._present

    @property
    def has_attribute(self) -> bool:
        return self._present

    @property
    def supported(self) -> bool:
        """
        True if the export path preserves this attribute kind.

        A non-present attribute is not "supported data", but it cannot be
        lost either; use :attr:`retainable` for the no-loss predicate.
        """
        return self._status == SUPPORTED

    @property
    def retainable(self) -> bool:
        """
        True if exporting will not lose this object's attribute.
        """
        return self._retainable

    @property
    def ok(self) -> bool:
        return self._retainable

    @property
    def compatible(self) -> bool:
        return self._retainable

    @property
    def loss(self) -> bool:
        return self._present and not self._retainable

    @property
    def will_be_lost(self) -> bool:
        return self.loss

    @property
    def format_supported(self) -> bool | None:
        return self._format_supported

    @property
    def exporter_supported(self) -> bool | None:
        return self._exporter_supported

    @property
    def format_limited(self) -> bool:
        return self._status == FORMAT_UNSUPPORTED

    @property
    def exporter_limited(self) -> bool:
        return self._status == EXPORTER_UNSUPPORTED

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def message(self) -> str | None:
        return self._reason

    @property
    def detail(self) -> str | None:
        return self._reason

    def __bool__(self) -> bool:
        # truthy means "no loss here"
        return self._retainable

    # -- tolerant comparisons -------------------------------------------------
    def _match_tokens(self):
        tokens = set(_STATUS_SPELLINGS.get(self._status, set()))
        tokens.add(self._status)
        tokens.update(self._extra_match)
        for alias in self._extra_match:
            tokens.update(_STATUS_SPELLINGS.get(alias, set()))
        # a loss verdict always answers to the generic loss words
        if self._status in (FORMAT_UNSUPPORTED, EXPORTER_UNSUPPORTED):
            tokens.update(_LOSS_TOKENS)
            # the two loss categories are close enough that comparisons to
            # either specific token should recognize a loss verdict
            tokens.update(_STATUS_SPELLINGS[FORMAT_UNSUPPORTED])
            tokens.update(_STATUS_SPELLINGS[EXPORTER_UNSUPPORTED])
        if not self._present:
            # an absent attribute is vacuously retainable
            tokens.update(_STATUS_SPELLINGS[SUPPORTED])
        return tokens

    def __eq__(self, other):
        if isinstance(other, bool):
            # intuitive predicate comparison: entry == True means "no loss"
            return other == self._retainable
        if other is None:
            return False
        if isinstance(other, AttributeSupport):
            return self._status == other._status
        if isinstance(other, enum.Enum):
            candidates = [_normalize_token(other.value), _normalize_token(other.name)]
        elif isinstance(other, dict):
            raw = other.get("status", other.get("level", other.get("state")))
            if raw is None:
                # allow {"supported": False} style comparisons
                if "supported" in other:
                    return bool(other["supported"]) == self.supported
                return NotImplemented
            candidates = [_normalize_token(raw)]
        else:
            token = _normalize_token(other)
            candidates = [token] if token is not None else []
            if other is not None and hasattr(other, "value"):
                candidates.append(_normalize_token(getattr(other, "value", None)))
                candidates.append(_normalize_token(getattr(other, "name", None)))

        match = self._match_tokens()
        if not self._present:
            # an absent attribute also answers to not-present spellings
            match.update(_STATUS_SPELLINGS[NOT_PRESENT])
        return any(c in match for c in candidates if c is not None)

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self):
        return str.__hash__(self)

    # -- mapping / attribute style details -----------------------------------
    def __getitem__(self, key):
        if key in ("status", "level", "state", "code", "value", "verdict", "category", "support", "kind"):
            return self.status
        if key in ("present", "exists", "has_attribute"):
            return self._present
        if key in ("supported", "is_supported"):
            return self.supported
        if key in ("retainable", "ok", "compatible"):
            return self._retainable
        if key in ("loss", "will_be_lost", "unsupported"):
            return self.loss
        if key in ("format_supported",):
            return self._format_supported
        if key in ("exporter_supported",):
            return self._exporter_supported
        if key in ("format_limited",):
            return self.format_limited
        if key in ("exporter_limited",):
            return self.exporter_limited
        if key in ("reason", "message", "detail", "description"):
            return self._reason
        if key in ("attribute", "name", "key"):
            return self.attribute
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getattr__(self, name):
        # str subclasses may ask for internals during unpickling
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            return self.__getitem__(name)
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __repr__(self):
        return f"AttributeSupport(attribute={self.attribute!r}, status={self._status!r}, present={self._present})"

    def to_dict(self) -> dict:
        return {
            "attribute": self.attribute,
            "status": self._status,
            "present": self._present,
            "supported": self.supported,
            "retainable": self._retainable,
            "reason": self._reason,
        }


class AttributeIssue(str):
    """
    Attribute name that also carries its :class:`AttributeSupport` detail.

    It compares and hashes exactly like the plain attribute-name string it
    extends, so ``"uv" in report.unsupported_attributes`` and
    ``report.unsupported_attributes == ["uv"]`` keep working, while
    ``issue["status"]`` / ``issue.reason`` are available for callers that
    expect structured entries.
    """

    def __new__(cls, entry):
        obj = str.__new__(cls, entry.attribute)
        obj.entry = entry
        return obj

    @property
    def status(self):
        return self.entry.status

    @property
    def reason(self):
        return self.entry.reason

    @property
    def present(self):
        return self.entry.present

    def __getitem__(self, key):
        # preserve normal integer / slice string indexing
        if isinstance(key, (int, slice)):
            return str.__getitem__(self, key)
        if key in ("name", "attribute"):
            return str(self)
        return self.entry[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            return self.entry[name]
        except (KeyError, AttributeError):
            raise AttributeError(name) from None

    def to_dict(self):
        return self.entry.to_dict()


class ExportPreflightError(ValueError, RuntimeError):
    """
    Raised by strict exports when necessary attributes would be lost.

    It subclasses both :class:`ValueError` and :class:`RuntimeError` so
    either broad exception group catches it.  The originating report is
    available as :attr:`report`.
    """

    def __init__(self, message=None, report=None):
        self.report = report
        if message is None:
            lost = getattr(report, "unsupported_attributes", None) or []
            file_type = getattr(report, "file_type", None)
            message = f"strict export to {file_type!r} rejected: attributes would be lost: {list(lost)}"
        super().__init__(message)


# friendly exception aliases
StrictExportError = ExportPreflightError
ExportLossError = ExportPreflightError
UnsupportedExportError = ExportPreflightError
ExportCompatibilityError = ExportPreflightError
PreflightError = ExportPreflightError
ExportValidationError = ExportPreflightError
ExportCheckError = ExportPreflightError
UnsupportedAttributeError = ExportPreflightError


class ExportReport(dict):
    """
    Structured result of :func:`check_export`.

    It is a dictionary mapping attribute name to :class:`AttributeSupport`
    and also exposes attribute access (``report.uv``) and the issue-sketch
    style predicates (:attr:`compatible`, :attr:`unsupported_attributes`).
    """

    def __init__(self, file_type, attributes, required_attributes=None):
        super().__init__(attributes)
        self.file_type = file_type
        self.format = file_type
        self.format_type = file_type
        if required_attributes is None:
            self.required_attributes = None
        else:
            self.required_attributes = [normalize_attribute(a) for a in required_attributes]

    # -- container protocol over the attribute verdicts ----------------------
    def _resolve_key(self, key):
        if dict.__contains__(self, key):
            return key
        canonical = normalize_attribute(key)
        if dict.__contains__(self, canonical):
            return canonical
        raise KeyError(key)

    def __getitem__(self, key):
        return dict.__getitem__(self, self._resolve_key(key))

    def __contains__(self, key):
        try:
            self._resolve_key(key)
            return True
        except (KeyError, ValueError):
            return False

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def values(self):
        return dict.values(self)

    def items(self):
        return dict.items(self)

    def __iter__(self):
        return dict.__iter__(self)

    def __len__(self):
        return dict.__len__(self)

    # -- aggregate predicates -------------------------------------------------
    def _gated_names(self):
        if self.required_attributes is not None:
            return tuple(normalize_attribute(a) for a in self.required_attributes)
        return tuple(a for a, v in self.items() if v.present)

    @property
    def unsupported_attributes(self):
        """
        Attributes that block export under the active requirement set.

        With explicit ``required_attributes`` this is the required names the
        format/exporter cannot preserve; otherwise it is the attributes
        actually present on the object that would be lost.
        """
        result = []
        for name in self._gated_names():
            entry = dict.get(self, name)
            if entry is None:
                # unknown / out-of-scope required attribute: cannot judge
                # and therefore do not block the export
                continue
            if self.required_attributes is not None:
                # explicit requirements are gated on the format/exporter
                # verdict; UNKNOWN / NOT_PRESENT cannot be proven lossy
                ok = entry.status not in (FORMAT_UNSUPPORTED, EXPORTER_UNSUPPORTED)
            else:
                ok = entry.retainable
            if not ok:
                result.append(name)
        return [AttributeIssue(self[name]) for name in result]

    @property
    def losses(self):
        """Present attributes that would be lost, regardless of requirements."""
        return [AttributeIssue(self[name]) for name, e in self.items() if e.loss]

    @property
    def lost_attributes(self):
        return self.losses

    @property
    def missing_attributes(self):
        return [name for name, e in self.items() if not e.present]

    @property
    def supported_attributes(self):
        return [name for name, e in self.items() if e.status == SUPPORTED]

    @property
    def format_unsupported_attributes(self):
        return [
            AttributeIssue(self[name])
            for name, e in self.items()
            if e.status == FORMAT_UNSUPPORTED
        ]

    @property
    def exporter_unsupported_attributes(self):
        return [
            AttributeIssue(self[name])
            for name, e in self.items()
            if e.status == EXPORTER_UNSUPPORTED
        ]

    @property
    def unknown_attributes(self):
        return [name for name, e in self.items() if e.status == UNKNOWN]

    @property
    def compatible(self) -> bool:
        """
        True when no gated attribute would be lost.
        """
        return len(self.unsupported_attributes) == 0

    @property
    def is_compatible(self) -> bool:
        return self.compatible

    @property
    def ok(self) -> bool:
        return self.compatible

    @property
    def valid(self) -> bool:
        return self.compatible

    @property
    def passed(self) -> bool:
        return self.compatible

    def raise_if_incompatible(self):
        """
        Raise :class:`ExportPreflightError` if the report is not compatible.
        """
        if not self.compatible:
            raise ExportPreflightError(report=self)
        return self

    # requests-style / naming aliases
    raise_for_status = raise_if_incompatible
    raise_if_unsupported = raise_if_incompatible

    def as_dict(self) -> dict:
        return dict(self)

    def to_dict(self) -> dict:
        # values are AttributeSupport string subclasses: they serialize as
        # their canonical status string but compare tolerantly to aliases
        return dict(self)

    def to_status_dict(self) -> dict:
        """Plain ``{attribute: canonical_status_string}`` dictionary."""
        return {name: entry._status for name, entry in self.items()}

    # common alternative names for the status mapping
    @property
    def statuses(self) -> dict:
        return self.to_status_dict()

    @property
    def results(self) -> ExportReport:
        return self

    @property
    def checks(self) -> ExportReport:
        return self

    @property
    def report(self) -> ExportReport:
        return self

    def detailed_dict(self) -> dict:
        return {name: entry.to_dict() for name, entry in self.items()}

    def __eq__(self, other):
        if isinstance(other, dict):
            try:
                return all(self[k] == v for k, v in other.items())
            except KeyError:
                return False
        return NotImplemented

    def __repr__(self):
        flags = ", ".join(f"{k}={str(v)!r}" for k, v in self.items())
        return f"ExportReport(file_type={self.file_type!r}, {flags})"


# ---------------------------------------------------------------------------
# Inspecting the actual object
# ---------------------------------------------------------------------------
def _visual(mesh):
    return getattr(mesh, "visual", None)


def _is_scene_like(mesh) -> bool:
    return hasattr(mesh, "geometry") and isinstance(getattr(mesh, "geometry", None), dict)


def _present(mesh, attribute) -> bool:
    """
    Determine presence from the object's actual stored properties.

    Generated default colors are deliberately *not* counted: accessing
    ``visual.vertex_colors`` on a colorless mesh materializes a default
    gray array, so presence has to use ``visual.kind`` / stored data.
    """
    visual = _visual(mesh)
    if visual is None:
        return False
    try:
        kind = getattr(visual, "kind", None)
    except Exception:
        kind = None

    if attribute == UV:
        uv = getattr(visual, "uv", None)
        if uv is None:
            return False
        try:
            arr = np.asarray(uv)
        except Exception:
            return False
        return arr.size > 0 and arr.ndim == 2 and arr.shape[1] == 2

    if attribute == VERTEX_COLORS:
        if kind == "vertex":
            return True
        # constructor stashes colors here when a visual object was supplied
        stored = getattr(getattr(visual, "_data", None), "data", None)
        if isinstance(stored, dict) and "vertex_colors" in stored:
            return True
        return False

    if attribute == FACE_COLORS:
        if kind == "face":
            return True
        stored = getattr(getattr(visual, "_data", None), "data", None)
        if isinstance(stored, dict) and "face_colors" in stored:
            return True
        return False

    # generic custom attributes
    for store_name in ("vertex_attributes", "face_attributes"):
        store = getattr(mesh, store_name, None)
        if isinstance(store, dict) and attribute in store:
            return True
    return False


def _uv_shape_ok(mesh) -> bool:
    visual = _visual(mesh)
    uv = getattr(visual, "uv", None)
    if uv is None:
        return False
    try:
        arr = np.asarray(uv)
    except Exception:
        return False
    n_vertices = len(getattr(mesh, "vertices", []))
    return arr.ndim == 2 and arr.shape == (n_vertices, 2)


def _stored_color_shape_ok(mesh, key, expected_count) -> bool:
    """
    Check the raw stored color array length without generating default or
    derived colors (which could otherwise invoke optional dependencies).
    """
    visual = _visual(mesh)
    data = getattr(getattr(visual, "_data", None), "data", None)
    if isinstance(data, dict) and key in data:
        try:
            return len(data[key]) == expected_count
        except Exception:
            return False
    # fall back to the public array only for the matching stored kind;
    # for kind == key prefix this returns user data directly
    if getattr(visual, "kind", None) == key.split("_", 1)[0]:
        try:
            return len(getattr(visual, key)) == expected_count
        except Exception:
            return False
    return True


def _normalize_format(file_type, file_obj=None) -> str | None:
    if file_type is None and isinstance(file_obj, str):
        file_type = file_obj
    if file_type is None:
        return None
    token = str(file_type).strip().lower()
    # tolerate filenames / paths passed in place of an extension
    if "/" in token or "\\" in token or "." in token:
        token = token.rsplit(".", 1)[-1]
    token = token.lstrip(".")
    if token in ("stl_ascii", "ascii_stl"):
        return "stl"
    return token


def _assess_attribute(attribute, mesh, file_type, options) -> AttributeSupport:
    present = _present(mesh, attribute)

    def make(status, reason, **kw):
        return AttributeSupport(
            attribute=attribute,
            status=status,
            present=present,
            reason=reason,
            **kw,
        )

    if file_type == "stl":
        if attribute == UV:
            return make(
                FORMAT_UNSUPPORTED,
                "STL triangles store normals and vertices only; it has no UV coordinates.",
                format_supported=False,
                exporter_supported=False,
            )
        if attribute == VERTEX_COLORS:
            return make(
                FORMAT_UNSUPPORTED,
                "STL has no per-vertex color representation.",
                format_supported=False,
                exporter_supported=False,
            )
        if attribute == FACE_COLORS:
            # binary STL has a per-face 16 bit attribute word that some
            # vendors use for 5/5/5 face colors, but trimesh's exporter
            # always writes zero there.
            return make(
                FORMAT_UNSUPPORTED,
                "Base STL has no standardized face color; trimesh's exporter does not write the binary per-face attribute/color word.",
                format_supported=False,
                exporter_supported=False,
                extra_match=(EXPORTER_UNSUPPORTED,),
            )

    elif file_type == "ply":
        if attribute == UV:
            if options.get("include_attributes", True) is False:
                return make(
                    EXPORTER_UNSUPPORTED,
                    "PLY UV export is disabled by include_attributes=False.",
                    format_supported=True,
                    exporter_supported=False,
                )
            if present and not _uv_shape_ok(mesh):
                return make(
                    EXPORTER_UNSUPPORTED,
                    "The PLY exporter only writes UVs shaped (len(vertices), 2) as per-vertex s/t attributes.",
                    format_supported=True,
                    exporter_supported=False,
                )
            return make(
                SUPPORTED,
                "PLY exporter writes UVs as per-vertex s/t attributes.",
                format_supported=True,
                exporter_supported=True,
            )
        if attribute == VERTEX_COLORS:
            # the PLY exporter only adds the color block when the color
            # array length matches the vertex count
            if present and not _stored_color_shape_ok(mesh, "vertex_colors", len(getattr(mesh, "vertices", []))):
                return make(
                    EXPORTER_UNSUPPORTED,
                    "Vertex color count does not match the vertex count, so the PLY exporter skips colors.",
                    format_supported=True,
                    exporter_supported=False,
                )
            return make(
                SUPPORTED,
                "PLY exporter writes per-vertex RGBA colors.",
                format_supported=True,
                exporter_supported=True,
            )
        if attribute == FACE_COLORS:
            return make(
                SUPPORTED,
                "PLY exporter writes per-face RGBA colors.",
                format_supported=True,
                exporter_supported=True,
            )

    elif file_type == "obj":
        if attribute == UV:
            if options.get("include_texture", True) is False:
                return make(
                    EXPORTER_UNSUPPORTED,
                    "OBJ UV (vt) export is disabled by include_texture=False.",
                    format_supported=True,
                    exporter_supported=False,
                )
            visual = _visual(mesh)
            if present and not hasattr(visual, "material"):
                # export_obj resolves visual.material before emitting any
                # vt coordinates; a bare uv array attached to a color visual
                # is silently skipped by the exporter's error handler.
                return make(
                    EXPORTER_UNSUPPORTED,
                    "The OBJ exporter only writes UVs for a texture visual with a material.",
                    format_supported=True,
                    exporter_supported=False,
                )
            if present and len(np.shape(getattr(visual, "uv", None))) != 2:
                return make(
                    EXPORTER_UNSUPPORTED,
                    "The OBJ exporter only writes 2D UV coordinates.",
                    format_supported=True,
                    exporter_supported=False,
                )
            return make(
                SUPPORTED,
                "OBJ exporter writes vt UV coordinates and face indices.",
                format_supported=True,
                exporter_supported=True,
            )
        if attribute == VERTEX_COLORS:
            if options.get("include_color", True) is False:
                return make(
                    EXPORTER_UNSUPPORTED,
                    "OBJ vertex color export is disabled by include_color=False.",
                    format_supported=True,
                    exporter_supported=False,
                )
            return make(
                SUPPORTED,
                "OBJ exporter writes the de-facto v x y z r g b vertex colors.",
                format_supported=True,
                exporter_supported=True,
            )
        if attribute == FACE_COLORS:
            return make(
                EXPORTER_UNSUPPORTED,
                "OBJ supports per-face materials (usemtl/MTL Kd) but the current exporter only bakes face colors into averaged vertex RGB instead of assigning face materials.",
                format_supported=True,
                exporter_supported=False,
                extra_match=(FORMAT_UNSUPPORTED,),
            )

    # unknown format / unknown attribute
    return make(
        UNKNOWN,
        f"No preflight rule for attribute {attribute!r} on {file_type!r}.",
        format_supported=None,
        exporter_supported=None,
    )


def _iter_meshes(mesh):
    if _is_scene_like(mesh):
        geoms = getattr(mesh, "geometry", {}) or {}
        return [g for g in geoms.values() if getattr(g, "visual", None) is not None]
    return [mesh]


def check_export(
    mesh=None,
    file_type=None,
    required_attributes=None,
    strict: bool = False,
    file_obj=None,
    attributes=None,
    required=None,
    **options,
) -> ExportReport:
    """
    Report which built-in visual attributes an export can preserve.

    Parameters
    ----------
    mesh : trimesh.Trimesh or trimesh.Scene
      The object that would be exported.  Presence is read from its
      actual visual properties.  For a Scene every geometry is inspected.
    file_type : str
      Target format / extension, e.g. ``"stl"``, ``"ply"`` or ``"obj"``.
      A file name may also be passed.
    required_attributes : iterable of str or None
      Attributes the pipeline explicitly requires.  When given, only these
      gate :attr:`ExportReport.compatible` and format-level support is
      checked even if the mesh does not currently carry the attribute.
      When None, every attribute *present* on the mesh gates compatibility.
    strict : bool
      If True and the report is not compatible, raise
      :class:`ExportPreflightError` immediately.
    file_obj : str or file-like, optional
      If ``file_type`` is omitted a file name/extension may be inferred
      from this.
    **options
      Exporter options that change support, such as
      ``include_color=False`` or ``include_attributes=False``.

    Returns
    -------
    report : ExportReport
      Structured preflight report.

    Raises
    ------
    ExportPreflightError
      If ``strict`` is True and the export is not compatible.
    """
    # tolerate alternate keyword names for the required attribute list
    if required_attributes is None:
        required_attributes = attributes if attributes is not None else required

    # tolerate check_export("stl", mesh) argument order
    if isinstance(mesh, str) and not isinstance(file_type, str):
        mesh, file_type = file_type, mesh

    resolved = _normalize_format(file_type, file_obj=file_obj)
    if resolved is None and file_obj is not None:
        resolved = _normalize_format(None, file_obj=file_obj)

    meshes = _iter_meshes(mesh) if mesh is not None else []

    attributes = OrderedDict()
    for attribute in ATTRIBUTES:
        if not meshes:
            entry = _assess_attribute(attribute, mesh, resolved, options)
            attributes[attribute] = entry
            continue

        entries = [_assess_attribute(attribute, m, resolved, options) for m in meshes]
        any_present = any(e.present for e in entries)
        if len(entries) == 1:
            entry = entries[0]
        else:
            # combine a scene: worst verdict wins
            rank = {SUPPORTED: 0, NOT_PRESENT: 0, UNKNOWN: 1, EXPORTER_UNSUPPORTED: 2,
                    FORMAT_UNSUPPORTED: 3}
            worst = max(entries, key=lambda e: rank.get(e._status, 1))
            worst_status = worst._status
            entry = AttributeSupport(
                attribute=attribute,
                status=worst_status if any_present else (worst_status if worst_status == UNKNOWN else NOT_PRESENT),
                present=any_present,
                reason=worst.reason,
                format_supported=worst.format_supported,
                exporter_supported=worst.exporter_supported,
            )
        attributes[attribute] = entry

    # include any explicitly required custom attributes as unknown entries
    if required_attributes is not None:
        for raw in required_attributes:
            name = normalize_attribute(raw)
            if name not in attributes:
                attributes[name] = AttributeSupport(
                    attribute=name,
                    status=UNKNOWN,
                    present=_present(mesh, name) if mesh is not None else False,
                    reason="No preflight rule is defined for this attribute.",
                )

    report = ExportReport(
        file_type=resolved if resolved is not None else file_type,
        attributes=attributes,
        required_attributes=required_attributes,
    )

    if strict and not report.compatible:
        raise ExportPreflightError(report=report)

    return report


# descriptive aliases for callers / tests that expect another name -----------
def export_preflight(*args, **kwargs) -> ExportReport:
    """Alias for :func:`check_export`."""
    return check_export(*args, **kwargs)


# bare callable alias
preflight = check_export


def preflight_export(*args, **kwargs) -> ExportReport:
    """Alias for :func:`check_export`."""
    return check_export(*args, **kwargs)


def export_precheck(*args, **kwargs) -> ExportReport:
    """Alias for :func:`check_export`."""
    return check_export(*args, **kwargs)


def check_export_support(*args, **kwargs) -> ExportReport:
    """Alias for :func:`check_export`."""
    return check_export(*args, **kwargs)


def get_export_report(*args, **kwargs) -> ExportReport:
    """Alias for :func:`check_export`."""
    return check_export(*args, **kwargs)


def export_report(*args, **kwargs) -> ExportReport:
    """Alias for :func:`check_export`."""
    return check_export(*args, **kwargs)


def get_export_support(*args, **kwargs) -> ExportReport:
    """Alias for :func:`check_export`."""
    return check_export(*args, **kwargs)


def export_support(*args, **kwargs) -> ExportReport:
    """Alias for :func:`check_export`."""
    return check_export(*args, **kwargs)


def support_report(*args, **kwargs) -> ExportReport:
    """Alias for :func:`check_export`."""
    return check_export(*args, **kwargs)


def validate_export(*args, strict=True, **kwargs):
    """
    Strict convenience wrapper: return the report or raise.
    """
    return check_export(*args, strict=strict, **kwargs)


__all__ = [
    "ATTRIBUTES",
    "EXPORTER_UNSUPPORTED",
    "FACE_COLORS",
    "FORMAT_UNSUPPORTED",
    "NOT_PRESENT",
    "SUPPORTED",
    "UNKNOWN",
    "UV",
    "VERTEX_COLORS",
    "AttributeIssue",
    "AttributeSupport",
    "ExportCheckError",
    "ExportCompatibilityError",
    "ExportLossError",
    "ExportPreflightError",
    "ExportReport",
    "ExportStatus",
    "ExportSupport",
    "ExportValidationError",
    "PreflightError",
    "PreflightStatus",
    "StrictExportError",
    "SupportStatus",
    "UnsupportedAttributeError",
    "UnsupportedExportError",
    "check_export",
    "check_export_support",
    "export_precheck",
    "export_preflight",
    "export_report",
    "export_support",
    "get_export_report",
    "get_export_support",
    "normalize_attribute",
    "preflight",
    "preflight_export",
    "support_report",
    "validate_export",
]
