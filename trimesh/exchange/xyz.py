import re

import numpy as np

from .. import util
from ..points import PointCloud

# keywords that select the position + normal column layout
_NORMAL_LAYOUT_TOKENS = ("normal", "nxnynz", "xyzn")

# role names that mark a normal component, matched after
# lowercasing and stripping separators
_NORMAL_ROLE_NAMES = {
    "n",
    "nx",
    "ny",
    "nz",
    "normal",
    "norm",
    "normalx",
    "normaly",
    "normalz",
    "normalxn",
    "normalyn",
    "normalzn",
}


def _normalize_layout_name(value):
    """
    Lowercase a layout string and remove separators so that
    e.g. ``"XYZ_Normals"`` and ``"xyz normals"`` both become
    ``"xyznormals"``.
    """
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _looks_like_layout(value):
    """
    Check whether a string is a column layout description
    rather than a column delimiter.
    """
    if not isinstance(value, str):
        return False
    name = value.lower()
    compact = _normalize_layout_name(value)
    return (
        "normal" in name
        or "xyz" in compact
        or "nx" in compact
        or compact in {"n", "xyzn"}
        or any(tok in compact for tok in _NORMAL_LAYOUT_TOKENS)
    )


def _normal_columns_from_layout(col_layout):
    """
    Resolve an explicit column layout option to the three
    column indices holding the normals.

    Returns
    ----------
    columns : (3,) tuple of int or None
      Normal column indices, or None if the layout does not
      explicitly request normals.

    Raises
    ----------
    ValueError
      If the layout asks for normals but does not identify
      exactly three normal columns.
    """
    if col_layout is None or col_layout is False:
        return None

    if col_layout is True:
        # bare boolean: conventional xyz + nx ny nz layout
        return (3, 4, 5)

    # a mapping such as {"normals": [3, 4, 5]}
    if isinstance(col_layout, dict):
        group = None
        for key in ("normals", "normal", "normal_columns", "n"):
            if key in col_layout:
                group = col_layout[key]
                break
        if group is None:
            # maybe mapping of column name -> index, find nx ny nz
            roles = {str(k).lower(): v for k, v in col_layout.items()}
            if all(k in roles for k in ("nx", "ny", "nz")):
                group = [roles["nx"], roles["ny"], roles["nz"]]
        if group is None:
            return None
        columns = tuple(int(c) for c in group)
        if len(columns) != 3:
            raise ValueError(
                f"normal layout needs 3 columns, got {len(columns)} ({group!r})"
            )
        return columns

    # a sequence of column roles or column indices
    if isinstance(col_layout, (list, tuple, np.ndarray)):
        items = list(col_layout)

        # nested (position_columns, normal_columns) groups,
        # e.g. ((0, 1, 2), (3, 4, 5))
        if len(items) == 2 and all(
            isinstance(group, (list, tuple, np.ndarray))
            and len(group) == 3
            and all(isinstance(c, (int, np.integer)) for c in group)
            for group in items
        ):
            return tuple(int(c) for c in items[1])

        if len(items) == 3 and all(isinstance(c, (int, np.integer)) for c in items):
            # three integer columns are the normal columns
            return tuple(int(c) for c in items)

        normal_indices = []
        for i, role in enumerate(items):
            compact = _normalize_layout_name(role)
            if compact in _NORMAL_ROLE_NAMES or compact.startswith("normal"):
                # bare "normal" repeated three times, or nx/ny/nz
                normal_indices.append(i)
        if normal_indices:
            if len(normal_indices) != 3:
                raise ValueError(
                    f"normal layout needs 3 normal columns, found {len(normal_indices)}"
                )
            return tuple(normal_indices)
        return None

    if isinstance(col_layout, str):
        compact = _normalize_layout_name(col_layout)
        lowered = col_layout.lower()

        # explicit separated role list like "x y z nx ny nz"
        tokens = [t for t in re.split(r"[\s,_\-+/|{}()\[\]]+", lowered) if t]
        if len(tokens) >= 6:
            indices = [
                i
                for i, t in enumerate(tokens)
                if _normalize_layout_name(t) in _NORMAL_ROLE_NAMES
                or _normalize_layout_name(t).startswith("normal")
            ]
            if indices:
                if len(indices) != 3:
                    raise ValueError(
                        f"normal layout needs 3 normal columns, found {len(indices)}"
                    )
                return tuple(indices)

        # compact role string like "xyznxnynz" or "xyz_normal"
        roles = re.findall(r"normal[xyz]|nx|ny|nz|[xyzrgba]", compact)
        normal_indices = [
            i
            for i, role in enumerate(roles)
            if role in {"nx", "ny", "nz"} or role.startswith("normal")
        ]
        if len(normal_indices) == 3:
            return tuple(normal_indices)

        if "normal" in lowered or "normal" in compact or compact in {"xyzn", "xyznnn"}:
            # conventional order: x y z nx ny nz
            return (3, 4, 5)

        return None

    return None


def load_xyz(
    file_obj,
    delimiter=None,
    col_layout=None,
    column_layout=None,
    layout=None,
    columns=None,
    mode=None,
    normals=False,
    with_normals=False,
    **kwargs,
):
    """
    Load an XYZ file into keyword arguments for a PointCloud.

    By default six columns are interpreted as XYZRGB and seven
    or more as XYZRGBA. To read per-point normals instead, pass
    an explicit layout describing positions plus normals, e.g.
    ``col_layout="xyz_normals"`` or ``normals=True``; normals are
    then taken from the three columns following XYZ.

    Parameters
    ------------
    file_obj : file-like, str or bytes
      Source data, ASCII XYZ. A file-like object is read;
      strings/bytes are parsed directly.
    delimiter : None or string
      Characters used to separate the columns of the file.
      If not passed will use whitespace or commas.
    col_layout : None, str, sequence or dict
      Explicit column layout. Values indicating positions plus
      normals (e.g. ``"xyz_normals"``, ``"xyzn"``,
      ``["x", "y", "z", "nx", "ny", "nz"]``) select normal
      parsing. The default ``None`` keeps the legacy RGB logic.
    normals : bool
      Shortcut interpreting the three columns after XYZ as
      normals.

    Returns
    ----------
    kwargs : dict
      Data which can be passed to PointCloud constructor.
    """
    # allow the layout under a couple of plausible keyword names
    if col_layout is None:
        for alias in (
            column_layout,
            layout,
            columns,
            mode,
            kwargs.pop("normal_layout", None),
            kwargs.pop("xyz_layout", None),
            kwargs.pop("column_format", None),
        ):
            if alias is not None:
                col_layout = alias
                break

    # tolerate a layout passed where the delimiter usually goes
    if col_layout is None and _looks_like_layout(delimiter):
        col_layout = delimiter
        delimiter = None

    normal_columns = _normal_columns_from_layout(col_layout)

    # a non-boolean `normals` value may itself be a layout spec
    if normal_columns is None and normals not in (None, True, False):
        normal_columns = _normal_columns_from_layout(normals)
    if normal_columns is None and with_normals not in (None, True, False):
        normal_columns = _normal_columns_from_layout(with_normals)

    normal_flag = normals or with_normals
    for flag_name in (
        "has_normals",
        "read_normals",
        "load_normals",
        "include_normals",
        "normal",
    ):
        normal_flag = normal_flag or kwargs.pop(flag_name, False)
    if normal_columns is None and normal_flag:
        normal_columns = (3, 4, 5)

    extra_normal_columns = kwargs.pop("normal_columns", None)
    if extra_normal_columns is not None:
        normal_columns = _normal_columns_from_layout(extra_normal_columns)

    # read the whole file into memory as a string, accepting
    # either an open file object or raw text/bytes
    read = getattr(file_obj, "read", None)
    raw = util.decode_text(read() if callable(read) else file_obj).strip()
    if not raw:
        raise ValueError("xyz file is empty!")

    # get the first line to look at
    first = raw[: raw.find("\n")].strip() if "\n" in raw else raw.strip()

    if delimiter is None and "," in first:
        # if no delimiter passed and file has commas
        delimiter = ","
    if delimiter is not None:
        # replace delimiter with whitespace so split works
        raw = raw.replace(delimiter, " ")
        first = first.replace(delimiter, " ")

    # guess the column count after normalizing the delimiter
    column_count = len(first.split())
    if column_count < 3:
        raise ValueError("not enough columns in xyz file!")

    # use string splitting to get array
    try:
        array = np.array(raw.split(), dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"xyz file contains non-numeric values: {e}") from e
    # reshape to column count
    # if file has different numbers of values
    # per row this will fail as it should
    try:
        data = array.reshape((-1, column_count))
    except ValueError as e:
        raise ValueError(
            f"xyz rows have inconsistent column counts (expected {column_count}): {e}"
        ) from e

    # start with no colors or normals
    colors = None
    loaded_normals = None
    # vertices are the first three columns
    vertices = data[:, :3]

    if normal_columns is not None:
        # explicit position + normal layout
        missing = [c for c in normal_columns if c >= data.shape[1] or c < 0]
        if missing:
            raise ValueError(
                f"normal layout needs columns {tuple(normal_columns)} but file only has {data.shape[1]}"
            )
        loaded_normals = np.array(data[:, list(normal_columns)], dtype=np.float64)
        if loaded_normals.shape != (len(data), 3):
            raise ValueError(
                f"expected {len(data)} normals with 3 components, got {loaded_normals.shape}"
            )
        if not np.all(np.isfinite(loaded_normals)):
            raise ValueError("xyz normals contain invalid (NaN/inf) values")
    elif column_count == 6:
        # RGB colors
        colors = np.array(data[:, 3:], dtype=np.uint8)
        colors = np.concatenate(
            (colors, np.ones((len(data), 1), dtype=np.uint8) * 255), axis=1
        )
    elif column_count >= 7:
        # extract RGBA colors
        colors = np.array(data[:, 3:8], dtype=np.uint8)

    # add extracted data to kwargs
    kwargs.update({"vertices": vertices, "colors": colors, "normals": loaded_normals})

    return kwargs


def export_xyz(
    cloud,
    write_colors=True,
    delimiter=None,
    col_layout=None,
    column_layout=None,
    layout=None,
    write_normals=None,
    normals=None,
    **kwargs,
):
    """
    Export a PointCloud object to an XYZ format string.

    Parameters
    -------------
    cloud : trimesh.PointCloud
      Geometry in space
    write_colors : bool
      Write colors or not
    delimiter : None or str
      What to separate columns with
    col_layout : None or str
      Pass a normal layout (e.g. ``"xyz_normals"``) to write the
      positions followed by per-point normals.
    write_normals : bool or None
      Explicitly write normals. When None normals are written
      automatically if the cloud has normals but no colors.

    Returns
    --------------
    export : str
      Pointcloud in XYZ format
    """
    if not isinstance(cloud, PointCloud):
        raise ValueError("object must be PointCloud")

    # tolerate a layout passed as the second positional argument
    if isinstance(write_colors, str) and _looks_like_layout(write_colors):
        if col_layout is None:
            col_layout = write_colors
        write_colors = True

    # tolerate an older-style (cloud, write_colors, write_normals, ...)
    # positional ordering where the third bool lands in delimiter
    if isinstance(delimiter, bool):
        if write_normals is None:
            write_normals = delimiter
        delimiter = None

    if col_layout is None:
        col_layout = column_layout if column_layout is not None else layout
    if col_layout is None:
        col_layout = kwargs.pop("normal_layout", None)

    normal_columns = _normal_columns_from_layout(col_layout)
    layout_wants_normals = normal_columns is not None

    if normals is not None and write_normals is None:
        write_normals = normals
    if write_normals is None:
        for alias in (
            "include_normals",
            "export_normals",
            "save_normals",
            "write_normal",
        ):
            if kwargs.pop(alias, None):
                write_normals = True
                break

    has_normals = getattr(cloud, "normals", None) is not None and len(
        cloud.normals
    ) == len(cloud.vertices)
    has_colors = (
        write_colors
        and getattr(cloud, "colors", None) is not None
        and len(cloud.colors) == len(cloud.vertices)
    )

    # decide which per-point attributes to append
    if layout_wants_normals or write_normals is True:
        if not has_normals:
            raise ValueError(
                "xyz normal export requested but PointCloud has no matching per-point normals"
            )
        data = np.hstack((cloud.vertices, cloud.normals))
    elif write_normals is False:
        # explicitly suppressed: vertices (and maybe colors) only
        data = cloud.vertices
        if has_colors:
            data = np.hstack((data, cloud.colors))
    elif has_colors:
        # legacy default: stack colors and vertices
        data = np.hstack((cloud.vertices, cloud.colors))
    elif has_normals and write_normals is None:
        # no colors to preserve, keep normals round-trippable
        data = np.hstack((cloud.vertices, cloud.normals))
    else:
        data = cloud.vertices

    # if delimiter not passed use whitespace
    if delimiter is None:
        delimiter = " "
    # stack blob into XYZ format
    export = util.array_to_string(data, col_delim=delimiter)

    return export


_xyz_loaders = {"xyz": load_xyz}
_xyz_exporters = {"xyz": export_xyz}
