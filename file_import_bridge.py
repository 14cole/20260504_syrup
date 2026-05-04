"""
file_import_bridge.py
=====================

Self-contained CreoBridge that reads curves from plain text files instead
of talking to Creo.  Drop this next to creo_geo_extractor.py.

File format
-----------
One point per line, two or three numeric columns.  Columns are separated
by any combination of whitespace (spaces, tabs) or a single comma.
The third column, if present, is the Z coordinate and is dropped (we
project onto the XY plane).

Blank lines and lines beginning with '#', '!', or ';' are treated as
comments and skipped.  This means you can drop in PTC IBL-style files
and most ad-hoc point dumps with no preprocessing.

Examples
~~~~~~~~

    # leading edge of NACA 0012, chord = 1.0
    0.000  0.000
    0.025  0.0287
    0.050  0.0394
    ...

or with a Z column that gets ignored:

    -0.500  0.000  0.000
     0.500  0.000  0.000

Usage
-----
At the bottom of creo_geo_extractor.py replace _choose_initial_bridge::

    from file_import_bridge import FileImportBridge

    def _choose_initial_bridge() -> CreoBridge:
        return FileImportBridge()

The 'Pick from Creo' button on each row will instead open a file dialog.
The '# Points' spinbox now has companion behaviour controlled by a new
per-row toggle:
    - 'resample'  : interpolate along the imported polyline at uniformly-
                    spaced arc-length parameters to hit exactly N points
    - 'as-is'     : use whatever points the file contains, ignore N

The toggle is exposed via a new column in the GUI; this module supplies
the data plumbing only.  The GUI integration patch is at the bottom of
this docstring.

GUI patch summary
~~~~~~~~~~~~~~~~~
1. Add a column constant ``COL_MODE`` between ``COL_NPTS`` and ``COL_SEGTYPE``,
   bumping all subsequent constants and inserting a label/width into
   ``COLUMNS``.  Header label "Sample".
2. In ``CurveRow`` add ``resample: bool = True``.
3. In ``_append_table_row`` insert a QComboBox at COL_MODE with two items
   "Resample (N pts)" and "As-is", wire it to set ``row.resample``.
4. In ``_on_npts_changed`` and ``_on_pick_curve``, if ``row.resample`` is
   False, call ``bridge.sample_curve(curve_id, 0)`` (special signal) so
   the bridge returns the file's native points.

The FileImportBridge below already supports both modes: pass any positive
integer to resample, or pass 0 / None to get the raw file points.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Optional Qt for the file-open dialog.  We only need it on pick_curve();
# everything else is plain Python so this module is unit-testable
# without a display.
try:
    from PySide6.QtWidgets import QApplication, QFileDialog
    _HAS_QT = True
except ImportError:
    _HAS_QT = False


# ---------------------------------------------------------------------------
# CurveSample dataclass — duck-typed equivalent of the GUI's CurveSample.
# ---------------------------------------------------------------------------

@dataclass
class CurveSample:
    curve_id: str
    name: str
    points: List[Tuple[float, float]]
    closed: bool = False


# ---------------------------------------------------------------------------
# TXT loader — tolerant of comments, mixed delimiters, optional Z column.
# ---------------------------------------------------------------------------

def load_txt_points(path: str) -> List[Tuple[float, float]]:
    """
    Read a plain-text point file and return [(x, y), ...].

    Robust to:
      - Comment lines starting with #, !, or ;
      - Blank lines
      - Mixed tabs / spaces / commas as separators
      - 2-column (x y) or 3-column (x y z) layouts; z is dropped
      - UTF-8 BOM at start of file
      - Windows / Unix / Mac line endings

    Raises ValueError with a line number for malformed rows.
    """
    pts: List[Tuple[float, float]] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            if line[0] in "#!;":
                continue
            # Normalise commas to spaces, then split on any whitespace.
            cleaned = line.replace(",", " ")
            tokens = cleaned.split()
            if len(tokens) < 2:
                raise ValueError(
                    f"{path}:{lineno}: expected at least 2 numbers, got {tokens!r}"
                )
            try:
                x = float(tokens[0])
                y = float(tokens[1])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{lineno}: could not parse numbers from {tokens[:3]!r}"
                ) from exc
            pts.append((x, y))
    if len(pts) < 2:
        raise ValueError(
            f"{path}: file contained fewer than 2 valid points "
            f"(found {len(pts)})."
        )
    return pts


# ---------------------------------------------------------------------------
# Arc-length resampling — uniform parameter spacing along the polyline.
# ---------------------------------------------------------------------------

def resample_polyline(
    pts: List[Tuple[float, float]],
    n_target: int,
) -> List[Tuple[float, float]]:
    """
    Resample a polyline to exactly ``n_target`` points spaced uniformly
    in arc length.  Endpoints are preserved.

    For n_target == len(pts) the output is *not* identical to the input
    in general — it's still uniformly arc-length spaced — but the first
    and last points always match the input endpoints.
    """
    n_target = max(2, int(n_target))
    if len(pts) < 2:
        raise ValueError("Need at least 2 points to resample.")

    # Cumulative arc length at each input vertex.
    cum: List[float] = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        seg = math.hypot(dx, dy)
        cum.append(cum[-1] + seg)
    total = cum[-1]
    if total <= 0.0:
        # Degenerate: all input points coincide.  Return n_target copies
        # of the single point; downstream code will still treat it as
        # a 2-point polyline of length 0.
        return [pts[0]] * n_target

    out: List[Tuple[float, float]] = []
    j = 0
    for i in range(n_target):
        # Special-case the endpoints: use the input vertices verbatim so
        # we get bit-exact endpoint preservation (no float drift).
        if i == 0:
            out.append(pts[0])
            continue
        if i == n_target - 1:
            out.append(pts[-1])
            continue
        s = total * i / (n_target - 1)
        # Advance j until cum[j+1] >= s.
        while j < len(cum) - 2 and cum[j + 1] < s:
            j += 1
        seg_len = cum[j + 1] - cum[j]
        if seg_len <= 0.0:
            out.append(pts[j])
            continue
        f = (s - cum[j]) / seg_len
        x = pts[j][0] + f * (pts[j + 1][0] - pts[j][0])
        y = pts[j][1] + f * (pts[j + 1][1] - pts[j][1])
        out.append((x, y))
    return out


def _is_closed(pts: List[Tuple[float, float]], tol: float = 1e-9) -> bool:
    if len(pts) < 3:
        return False
    return math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < tol


# ---------------------------------------------------------------------------
# Bridge implementation — duck-types as creo_geo_extractor.CreoBridge.
# ---------------------------------------------------------------------------

class FileImportBridge:
    """
    Curve source backed by plain-text point files on disk.  No Creo
    connection, no network, no third-party packages.

    Each picked curve remembers its original on-disk points so that
    later 'resample' requests don't need to re-read the file.
    """

    def __init__(self) -> None:
        self._connected = False
        # curve_id -> {'name': str, 'raw_pts': List[(x,y)], 'path': str}
        self._curves: dict[str, dict] = {}
        self._next_idx = 0

    # ---- Bridge contract used by the GUI ------------------------------

    @property
    def display_name(self) -> str:
        return "FileImportBridge"

    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> str:
        self._connected = True
        return "File-import backend ready (no Creo required)."

    def disconnect(self) -> None:
        self._connected = False
        self._curves.clear()

    def pick_curve(self) -> Optional[CurveSample]:
        """
        Open a file dialog, load the chosen file's points, register it
        as a new curve, and return the initial sample.  The GUI's
        '# Points' spinbox controls how many points come back; pass
        n_points <= 0 in subsequent sample_curve calls to get the raw
        file contents instead.
        """
        if not self._connected:
            raise RuntimeError("Not connected.")
        path = self._open_file_dialog()
        if not path:
            return None

        try:
            raw_pts = load_txt_points(path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not load {path}: {exc}") from exc

        self._next_idx += 1
        cid = f"file_{self._next_idx}"
        # Default visible name = filename without extension.
        name = os.path.splitext(os.path.basename(path))[0] or f"curve_{self._next_idx}"
        self._curves[cid] = {"name": name, "raw_pts": raw_pts, "path": path}

        # Initial sample: 16 resampled points.  Caller can re-request
        # with the GUI's spinbox value, or with 0 for "as-is".
        initial_n = 16 if len(raw_pts) > 16 else len(raw_pts)
        return self.sample_curve(cid, initial_n)

    def sample_curve(
        self, curve_id: str, n_points: int,
    ) -> Optional[CurveSample]:
        """
        Return points for a previously-picked curve.

        n_points > 0  : arc-length resample to exactly n_points
        n_points == 0 : return the raw file's points unchanged
        n_points < 0  : same as 0 (defensive)
        """
        info = self._curves.get(curve_id)
        if info is None:
            return None

        raw = info["raw_pts"]
        if n_points is None or n_points <= 0:
            pts = list(raw)
        else:
            pts = resample_polyline(raw, n_points)

        return CurveSample(
            curve_id=curve_id,
            name=info["name"],
            points=pts,
            closed=_is_closed(pts),
        )

    # ---- Helpers ------------------------------------------------------

    def _open_file_dialog(self) -> Optional[str]:
        if not _HAS_QT:
            # Headless fallback: read FILE_IMPORT_PATH env var.  Useful
            # for unit tests and CI.
            env_path = os.environ.get("FILE_IMPORT_PATH")
            return env_path
        # Need a QApplication for QFileDialog.  Reuse if one is running.
        app = QApplication.instance()
        if app is None:
            # If the bridge is being driven from a script with no Qt loop,
            # we can't open a dialog.  Caller should supply the path.
            return None
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Select curve point file",
            "",
            "Text point files (*.txt *.dat *.pts *.ibl *.csv);;All files (*)",
        )
        return path or None

    # ---- Programmatic API for tests / scripting -----------------------

    def add_curve_from_path(self, path: str, name: Optional[str] = None) -> str:
        """
        Headless equivalent of pick_curve().  Returns the new curve_id.
        Raises ValueError on malformed input.
        """
        raw_pts = load_txt_points(path)
        self._next_idx += 1
        cid = f"file_{self._next_idx}"
        if name is None:
            name = os.path.splitext(os.path.basename(path))[0] or f"curve_{self._next_idx}"
        self._curves[cid] = {"name": name, "raw_pts": raw_pts, "path": path}
        if not self._connected:
            self._connected = True
        return cid


# ---------------------------------------------------------------------------
# Headless self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """python file_import_bridge.py"""
    import tempfile

    print("FileImportBridge self-test")
    print("-" * 40)

    sample = (
        "# NACA-ish leading edge\n"
        "0.000  0.000\n"
        "0.025, 0.0287\n"
        "0.050\t0.0394\n"
        "; legacy comment\n"
        "0.100  0.0480  0.0\n"   # 3-col, Z dropped
        "0.200  0.0578\n"
        "0.300  0.0606\n"
        "0.500  0.0529\n"
        "1.000  0.0010\n"
    )
    fd, path = tempfile.mkstemp(suffix=".txt", text=True)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(sample)

        bridge = FileImportBridge()
        bridge.connect()
        cid = bridge.add_curve_from_path(path, name="naca_test")
        print(f"  loaded curve_id = {cid}")

        raw = bridge.sample_curve(cid, 0)
        print(f"  as-is points    = {len(raw.points)}")
        assert len(raw.points) == 8, "expected 8 raw points after stripping comments"

        n_target = 5
        rs = bridge.sample_curve(cid, n_target)
        print(f"  resampled to    = {len(rs.points)}  (target {n_target})")
        assert len(rs.points) == n_target

        # Endpoints must match the file's first and last point.
        assert rs.points[0] == raw.points[0], "first point preserved"
        assert rs.points[-1] == raw.points[-1], "last point preserved"

        # Test arc-length spacing: the *arc-length* position along the
        # original polyline at each resampled point should be uniformly
        # spaced.  (Euclidean distance between consecutive samples is
        # NOT equal in general because the polyline kinks at vertices.)
        def arc_pos_along(raw_pts, q):
            """Find arc-length position of q along raw_pts (q must lie on polyline)."""
            cum_s = 0.0
            for k in range(len(raw_pts) - 1):
                ax, ay = raw_pts[k]
                bx, by = raw_pts[k + 1]
                seg = math.hypot(bx - ax, by - ay)
                if seg == 0.0:
                    continue
                # project q onto segment ab
                t = ((q[0] - ax) * (bx - ax) + (q[1] - ay) * (by - ay)) / (seg * seg)
                if -1e-9 <= t <= 1 + 1e-9:
                    proj_x = ax + t * (bx - ax)
                    proj_y = ay + t * (by - ay)
                    if math.hypot(q[0] - proj_x, q[1] - proj_y) < 1e-9:
                        return cum_s + max(0.0, min(1.0, t)) * seg
                cum_s += seg
            return cum_s
        positions = [arc_pos_along(raw.points, p) for p in rs.points]
        deltas = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        avg = sum(deltas) / len(deltas)
        spread = max(deltas) - min(deltas)
        rel = spread / avg if avg > 0 else 0.0
        print(f"  arc-length spread: {rel * 100:.4f}% of mean (lower = better)")
        assert rel < 1e-6, "resampled points should be uniformly spaced in arc length"

        # Test closed-curve detection.
        fd2, path2 = tempfile.mkstemp(suffix=".txt", text=True)
        try:
            with os.fdopen(fd2, "w") as fh:
                fh.write(
                    "1.0  0.0\n"
                    "0.0  1.0\n"
                    "-1.0 0.0\n"
                    "0.0 -1.0\n"
                    "1.0  0.0\n"
                )
            cid2 = bridge.add_curve_from_path(path2, name="closed_diamond")
            cs = bridge.sample_curve(cid2, 0)
            print(f"  closed-curve detection: {cs.closed}")
            assert cs.closed
        finally:
            os.unlink(path2)

        print("OK")
        return 0
    finally:
        os.unlink(path)


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
