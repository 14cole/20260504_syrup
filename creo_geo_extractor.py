"""
creo_geo_extractor.py
=====================

PySide6 GUI that talks to Creo 10 via win32com, samples points along sketch
curves, and exports a `.geo` file matching the format consumed by the
2D BIE/MoM RCS solver (rcs_solver.py).

File format (matches /mnt/user-data/uploads/_tmp_type4_interface.geo):

    Title: <free text>
    Segment: <name> line
    properties: <seg_type> <n_panels> <ang_deg> <ibc_flag> <ipn1> <ipn2>
    <x1> <y1> <x2> <y2>
    [more polyline rows]
    [more Segment blocks]
    IBCS:
    <flag> <R> <X> <0.0>
    Dielectrics:
    <flag> <eps_r> <eps_i> <mu_r> <mu_i>


Architecture
------------
- CreoBridge              : abstract sampler interface
- Win32ComCreoBridge      : real Creo 10 connection via pywin32 / pfcls
- MockCreoBridge          : synthesizes curves, lets the GUI run without Creo
- CurveRow                : per-row state (curve handle, color, sampled pts)
- PreviewCanvas           : custom QPainter widget showing all curves with
                            dark->light gradient + per-row color tint
- MainWindow              : table + preview + toolbar

Run
---
    pip install PySide6 pywin32           # pywin32 only needed on Windows
    python creo_geo_extractor.py


On non-Windows / no Creo running, the app boots into a Mock backend so you
can exercise the GUI end-to-end.
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# pywin32 is Windows-only; we lazy-import inside Win32ComCreoBridge.
# ---------------------------------------------------------------------------

try:
    from PySide6.QtCore import (
        QPointF,
        QRectF,
        QSize,
        Qt,
        QTimer,
        Signal,
    )
    from PySide6.QtGui import (
        QAction,
        QBrush,
        QColor,
        QFont,
        QFontDatabase,
        QIcon,
        QLinearGradient,
        QPainter,
        QPalette,
        QPen,
        QPolygonF,
    )
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QColorDialog,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QSplitter,
        QStatusBar,
        QStyle,
        QStyledItemDelegate,
        QStyleOptionButton,
        QTableWidget,
        QTableWidgetItem,
        QToolBar,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    print(
        "PySide6 is required.  Install with:  pip install PySide6\n"
        f"({exc})"
    )
    sys.exit(2)


# ===========================================================================
# Curve sampling abstraction
# ===========================================================================

@dataclass
class CurveSample:
    """Result of sampling a parametric curve at N points."""
    curve_id: str            # opaque identifier from the backend
    name: str                # human-readable label, e.g. "edge_3"
    points: List[Tuple[float, float]]  # 2D (x, y) in geometry units
    closed: bool = False     # whether the curve forms a closed loop


class CreoBridge(ABC):
    """Backend that picks curves from Creo and samples points along them."""

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def connect(self) -> str:
        """Establish connection.  Returns a status string for the user."""

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def pick_curve(self) -> Optional[CurveSample]:
        """Prompt user (in Creo) to pick a curve; returns initial sample."""

    @abstractmethod
    def sample_curve(self, curve_id: str, n_points: int) -> Optional[CurveSample]:
        """Re-sample a previously picked curve at n_points uniform parameters."""

    @property
    def display_name(self) -> str:
        return type(self).__name__


# ---------------------------------------------------------------------------
# Real Creo 10 backend via pywin32 / pfcls AsyncConnection
# ---------------------------------------------------------------------------

class Win32ComCreoBridge(CreoBridge):
    """
    Connects to a running Creo 10 session via the PTC pfcls AsyncConnection
    COM interface (delivered with ProToolkit / OTK).  Requires:

      * Creo 10 running with a sketch active
      * pywin32 installed
      * pfcls.dll registered (handled by the Creo installer)

    If any of the above is missing, .connect() raises with a clear message
    and the caller can fall back to MockCreoBridge.
    """

    # PTC publishes several ProgIDs across versions; we try each in order.
    _PROGIDS = (
        "pfcls.pfclsAsyncConnection",
        "pfcls.AsyncConnection",
        "Pfc.Application",
    )

    def __init__(self) -> None:
        self._async = None
        self._conn = None
        self._session = None
        self._curve_cache: dict[str, object] = {}  # id -> COM curve handle

    def is_connected(self) -> bool:
        return self._session is not None

    def connect(self) -> str:
        try:
            import win32com.client  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pywin32 not installed.  Install with:  pip install pywin32"
            ) from exc

        last_err: Optional[Exception] = None
        for progid in self._PROGIDS:
            try:
                self._async = win32com.client.Dispatch(progid)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                self._async = None
        if self._async is None:
            raise RuntimeError(
                "Could not Dispatch any Creo COM ProgID.  Tried: "
                + ", ".join(self._PROGIDS)
                + f".  Last error: {last_err}"
            )

        # Connect to a running Creo session.  Empty strings + -1 == attach.
        try:
            self._conn = self._async.Connect("", "", "", -1)
            self._session = self._conn.Session
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Connected to COM server but could not attach to session: {exc}.\n"
                "Make sure Creo 10 is running and a model is open."
            ) from exc
        return f"Connected to Creo session via {self._async.__class__.__name__}"

    def disconnect(self) -> None:
        try:
            if self._conn is not None:
                self._conn.Disconnect(2)
        except Exception:  # noqa: BLE001
            pass
        self._async = None
        self._conn = None
        self._session = None
        self._curve_cache.clear()

    def pick_curve(self) -> Optional[CurveSample]:
        if self._session is None:
            raise RuntimeError("Not connected to Creo.")
        # Prompt user to pick a single edge/curve.  The exact API is
        # IpfcSession.Select("edge", 1).  Different Creo releases expose
        # slightly different selectors; try the common ones.
        selection = None
        for selector in ("edge", "curve", "datum_curve"):
            try:
                selection = self._session.Select(selector, 1)
                if selection is not None and len(selection) > 0:
                    break
            except Exception:  # noqa: BLE001
                continue
        if not selection or len(selection) == 0:
            return None

        sel = selection[0]
        try:
            owner = sel.SelItem
        except Exception:  # noqa: BLE001
            owner = sel
        curve_id = f"creo_{id(owner):x}"
        try:
            name = str(getattr(owner, "Name", curve_id))
        except Exception:  # noqa: BLE001
            name = curve_id
        self._curve_cache[curve_id] = owner
        # Default to a reasonable initial point count.
        return self.sample_curve(curve_id, 16)

    def sample_curve(self, curve_id: str, n_points: int) -> Optional[CurveSample]:
        owner = self._curve_cache.get(curve_id)
        if owner is None:
            return None
        n_points = max(2, int(n_points))
        # Curve sampling: IpfcCurveDescriptor.Eval3DData(parameter).
        # Parameter range is typically [0, 1] but some descriptors use
        # the absolute curve length.  We probe both.
        try:
            descriptor = owner.GetCurveDescriptor()
        except Exception:  # noqa: BLE001
            descriptor = owner

        # Get the parameter range.
        try:
            t_lo = float(descriptor.MinParam)
            t_hi = float(descriptor.MaxParam)
        except Exception:  # noqa: BLE001
            t_lo, t_hi = 0.0, 1.0

        pts: List[Tuple[float, float]] = []
        for i in range(n_points):
            t = t_lo + (t_hi - t_lo) * (i / (n_points - 1))
            try:
                data = descriptor.Eval3DData(t)
                x, y, _z = float(data.Point[0]), float(data.Point[1]), float(data.Point[2])
            except Exception:  # noqa: BLE001
                # As a last resort, ask the curve directly.
                try:
                    data = owner.Eval3DData(t)
                    x, y, _z = float(data.Point[0]), float(data.Point[1]), float(data.Point[2])
                except Exception:  # noqa: BLE001
                    return None
            pts.append((x, y))
        # Closed if endpoints coincide within a small tolerance.
        closed = (
            len(pts) >= 3
            and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-9
        )
        try:
            name = str(getattr(owner, "Name", curve_id))
        except Exception:  # noqa: BLE001
            name = curve_id
        return CurveSample(curve_id=curve_id, name=name, points=pts, closed=closed)


# ---------------------------------------------------------------------------
# Mock backend so the GUI runs without Creo (great for design + screenshots).
# ---------------------------------------------------------------------------

class MockCreoBridge(CreoBridge):
    """Generates synthetic curves so the GUI is fully functional offline."""

    def __init__(self) -> None:
        self._connected = False
        self._next_idx = 0
        self._curves: dict[str, dict] = {}
        # A small library of canned curves the "user picks" round-robin.
        self._library: List[dict] = [
            {"name": "circle_1", "kind": "circle", "cx": 0.0, "cy": 0.0, "r": 1.0, "closed": True},
            {"name": "wing_top", "kind": "naca", "chord": 2.0, "x0": -1.0, "y0": 0.0, "thickness": 0.12, "side": +1, "closed": False},
            {"name": "wing_bot", "kind": "naca", "chord": 2.0, "x0": -1.0, "y0": 0.0, "thickness": 0.12, "side": -1, "closed": False},
            {"name": "spline_1", "kind": "spline", "ctrl": [(-1.5, -0.5), (-0.5, 0.8), (0.5, -0.4), (1.5, 0.3)], "closed": False},
            {"name": "polygon_1", "kind": "ngon", "cx": 2.5, "cy": 0.0, "r": 0.8, "n": 6, "closed": True},
        ]

    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> str:
        self._connected = True
        return "Mock backend active (Creo not connected)"

    def disconnect(self) -> None:
        self._connected = False
        self._curves.clear()

    def pick_curve(self) -> Optional[CurveSample]:
        if not self._connected:
            raise RuntimeError("Not connected (mock).")
        spec = self._library[self._next_idx % len(self._library)]
        self._next_idx += 1
        cid = f"mock_{self._next_idx}"
        self._curves[cid] = dict(spec)
        return self.sample_curve(cid, 24)

    def sample_curve(self, curve_id: str, n_points: int) -> Optional[CurveSample]:
        spec = self._curves.get(curve_id)
        if spec is None:
            return None
        n = max(2, int(n_points))
        kind = spec["kind"]
        pts: List[Tuple[float, float]] = []

        if kind == "circle":
            cx, cy, r = spec["cx"], spec["cy"], spec["r"]
            # Closed: last point coincides with first.
            for i in range(n):
                t = 2 * math.pi * i / (n - 1) if not spec["closed"] else 2 * math.pi * i / (n - 1)
                pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))

        elif kind == "naca":
            chord = spec["chord"]
            x0, y0 = spec["x0"], spec["y0"]
            t = spec["thickness"]
            side = spec["side"]
            for i in range(n):
                u = i / (n - 1)
                x = u * chord
                # NACA 00xx half-thickness (without trailing edge close-out).
                yt = 5 * t * (
                    0.2969 * math.sqrt(u)
                    - 0.1260 * u
                    - 0.3516 * u**2
                    + 0.2843 * u**3
                    - 0.1015 * u**4
                )
                pts.append((x0 + x, y0 + side * yt))

        elif kind == "spline":
            ctrl = spec["ctrl"]
            for i in range(n):
                u = (len(ctrl) - 1) * i / (n - 1)
                k = min(int(u), len(ctrl) - 2)
                f = u - k
                # Catmull-Rom-ish smooth interpolation.
                p0 = ctrl[max(0, k - 1)]
                p1 = ctrl[k]
                p2 = ctrl[k + 1]
                p3 = ctrl[min(len(ctrl) - 1, k + 2)]
                f2, f3 = f * f, f * f * f
                a = -0.5 * f3 + f2 - 0.5 * f
                b = 1.5 * f3 - 2.5 * f2 + 1.0
                c = -1.5 * f3 + 2.0 * f2 + 0.5 * f
                d = 0.5 * f3 - 0.5 * f2
                x = a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0]
                y = a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]
                pts.append((x, y))

        elif kind == "ngon":
            cx, cy, r, ns = spec["cx"], spec["cy"], spec["r"], spec["n"]
            # Sample uniformly along the polygon perimeter.
            verts = [
                (cx + r * math.cos(2 * math.pi * i / ns),
                 cy + r * math.sin(2 * math.pi * i / ns))
                for i in range(ns + 1)
            ]
            seg_lens = [
                math.hypot(verts[i + 1][0] - verts[i][0], verts[i + 1][1] - verts[i][1])
                for i in range(ns)
            ]
            total = sum(seg_lens)
            cum = [0.0]
            for L in seg_lens:
                cum.append(cum[-1] + L)
            for i in range(n):
                s = total * i / (n - 1)
                # Find which segment.
                k = 0
                while k < ns - 1 and cum[k + 1] < s:
                    k += 1
                f = (s - cum[k]) / max(seg_lens[k], 1e-12)
                x = verts[k][0] + f * (verts[k + 1][0] - verts[k][0])
                y = verts[k][1] + f * (verts[k + 1][1] - verts[k][1])
                pts.append((x, y))

        return CurveSample(
            curve_id=curve_id,
            name=spec["name"],
            points=pts,
            closed=bool(spec["closed"]),
        )


# ===========================================================================
# Per-row model
# ===========================================================================

@dataclass
class CurveRow:
    """One table row: a Creo curve plus its solver-segment metadata."""
    sample: Optional[CurveSample] = None
    n_points: int = 16
    seg_type: int = 2          # 1 sheet, 2 PEC, 3 dielectric IF, 4 dielectric+PEC, 5 diel/diel
    ibc_flag: int = 0
    ipn1: int = 0
    ipn2: int = 0
    color: QColor = field(default_factory=lambda: QColor("#4FC3F7"))


SEG_TYPE_LABELS = {
    1: "1  sheet (impedance)",
    2: "2  PEC / IBC closed",
    3: "3  dielectric interface",
    4: "4  dielectric-backed PEC",
    5: "5  dielectric/dielectric",
}


# ===========================================================================
# Preview canvas (custom paint, zero deps)
# ===========================================================================

class PreviewCanvas(QWidget):
    """Renders all curves with per-row color and dark->light point gradient."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._rows: List[CurveRow] = []
        self.setMinimumSize(400, 400)
        self.setBackgroundRole(QPalette.Base)
        self.setAutoFillBackground(True)
        # Subtle dotted-grid background.
        pal = self.palette()
        pal.setColor(QPalette.Base, QColor("#0E1116"))
        self.setPalette(pal)

    def set_rows(self, rows: Sequence[CurveRow]) -> None:
        self._rows = list(rows)
        self.update()

    # -----------------------------------------------------------------------
    # Geometry helpers
    # -----------------------------------------------------------------------

    def _bounds(self) -> Optional[Tuple[float, float, float, float]]:
        xs: List[float] = []
        ys: List[float] = []
        for row in self._rows:
            if row.sample:
                for x, y in row.sample.points:
                    xs.append(x)
                    ys.append(y)
        if not xs:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    def _world_to_screen(self) -> Tuple[Callable[[float, float], QPointF], float]:
        """Returns (transform, pixels-per-unit)."""
        b = self._bounds()
        w = self.width()
        h = self.height()
        margin = 36
        if b is None:
            cx, cy = 0.0, 0.0
            scale = 60.0
        else:
            xmin, ymin, xmax, ymax = b
            dx = max(xmax - xmin, 1e-9)
            dy = max(ymax - ymin, 1e-9)
            scale = min((w - 2 * margin) / dx, (h - 2 * margin) / dy)
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
        wx = w / 2
        wy = h / 2

        def to_screen(x: float, y: float) -> QPointF:
            return QPointF(wx + (x - cx) * scale, wy - (y - cy) * scale)

        return to_screen, scale

    # -----------------------------------------------------------------------
    # Painting
    # -----------------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._paint_grid(p)
        self._paint_axes(p)
        if not self._rows:
            self._paint_placeholder(p)
            return
        for row in self._rows:
            if row.sample and row.sample.points:
                self._paint_curve(p, row)

    def _paint_placeholder(self, p: QPainter) -> None:
        p.setPen(QColor("#3a4250"))
        font = p.font()
        font.setPointSize(11)
        font.setItalic(True)
        p.setFont(font)
        p.drawText(self.rect(), Qt.AlignCenter,
                   "No curves yet — click + to add a row, then “Pick from Creo”.")

    def _paint_grid(self, p: QPainter) -> None:
        # Dotted subgrid for that engineering-tool feel.
        p.setPen(QColor(50, 56, 64))
        step = 24
        for x in range(0, self.width(), step):
            for y in range(0, self.height(), step):
                p.drawPoint(x, y)

    def _paint_axes(self, p: QPainter) -> None:
        to_screen, _ = self._world_to_screen()
        origin = to_screen(0.0, 0.0)
        # Axes lines extending across the whole widget if origin is in view.
        if 0 <= origin.x() <= self.width():
            p.setPen(QPen(QColor(70, 78, 90), 1, Qt.DotLine))
            p.drawLine(QPointF(origin.x(), 0), QPointF(origin.x(), self.height()))
        if 0 <= origin.y() <= self.height():
            p.setPen(QPen(QColor(70, 78, 90), 1, Qt.DotLine))
            p.drawLine(QPointF(0, origin.y()), QPointF(self.width(), origin.y()))

    def _paint_curve(self, p: QPainter, row: CurveRow) -> None:
        to_screen, _ = self._world_to_screen()
        pts_world = row.sample.points
        screen_pts = [to_screen(x, y) for x, y in pts_world]

        # Polyline backbone in the row color, slightly translucent.
        base = row.color
        backbone = QColor(base.red(), base.green(), base.blue(), 110)
        p.setPen(QPen(backbone, 1.5))
        if len(screen_pts) > 1:
            poly = QPolygonF(screen_pts)
            p.drawPolyline(poly)

        # Sample points with dark->light color gradient.
        n = len(screen_pts)
        for i, sp in enumerate(screen_pts):
            t = i / max(1, n - 1)
            # Dark = 30% intensity, light = 100% intensity; mix toward white.
            dark = 0.35
            mix = dark + (1.0 - dark) * t
            r = int(min(255, base.red() * mix + 255 * (t * 0.15)))
            g = int(min(255, base.green() * mix + 255 * (t * 0.15)))
            b = int(min(255, base.blue() * mix + 255 * (t * 0.15)))
            dot_color = QColor(r, g, b)
            p.setBrush(QBrush(dot_color))
            p.setPen(QPen(QColor(0, 0, 0, 180), 0.8))
            radius = 4.5
            p.drawEllipse(sp, radius, radius)

        # Endpoint labels: "<name>_0" and "<name>_<n-1>".
        if screen_pts:
            label_pen = QPen(QColor(220, 226, 234, 220))
            p.setPen(label_pen)
            font = p.font()
            font.setPointSize(8)
            font.setBold(True)
            p.setFont(font)
            label0 = f"{row.sample.name}_0"
            labelN = f"{row.sample.name}_{n - 1}"
            p.drawText(screen_pts[0] + QPointF(7, -7), label0)
            if n > 1:
                p.drawText(screen_pts[-1] + QPointF(7, -7), labelN)


# ===========================================================================
# Color picker button (small swatch)
# ===========================================================================

class ColorButton(QPushButton):
    color_changed = Signal(QColor)

    def __init__(self, color: QColor, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(40, 26)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self._open_dialog)
        self._update_style()

    def color(self) -> QColor:
        return QColor(self._color)

    def set_color(self, color: QColor) -> None:
        self._color = QColor(color)
        self._update_style()

    def _update_style(self) -> None:
        c = self._color
        border = "#1a1d22"
        self.setStyleSheet(
            f"QPushButton {{ background: {c.name()}; border: 1px solid {border};"
            f" border-radius: 4px; }}"
            f"QPushButton:hover {{ border: 1px solid #6cb4ee; }}"
        )

    def _open_dialog(self) -> None:
        dlg = QColorDialog(self._color, self)
        dlg.setOption(QColorDialog.DontUseNativeDialog, False)
        if dlg.exec():
            new = dlg.selectedColor()
            if new.isValid():
                self._color = new
                self._update_style()
                self.color_changed.emit(self._color)


# ===========================================================================
# Main window
# ===========================================================================

DEFAULT_PALETTE = [
    "#4FC3F7", "#FF8A65", "#AED581", "#BA68C8",
    "#FFD54F", "#4DB6AC", "#F06292", "#9FA8DA",
]

COL_CURVE = 0
COL_PICK = 1
COL_NPTS = 2
COL_SEGTYPE = 3
COL_IBC = 4
COL_IPN1 = 5
COL_IPN2 = 6
COL_COLOR = 7
COL_STATUS = 8
COLUMNS = [
    ("Curve",     130),
    ("",           90),   # pick button
    ("# Points",   90),
    ("Type",      210),
    ("IBC",        70),
    ("IPN1",       70),
    ("IPN2",       70),
    ("Color",      70),
    ("Status",    160),
]


class MainWindow(QMainWindow):
    def __init__(self, bridge: CreoBridge) -> None:
        super().__init__()
        self.setWindowTitle("Creo → .geo  ·  curve sampler for 2D RCS solver")
        self.resize(1480, 820)

        self._bridge = bridge
        self._rows: List[CurveRow] = []

        self._build_ui()
        self._apply_theme()
        self._refresh_status_bar()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._make_toolbar()

        splitter = QSplitter(Qt.Horizontal)

        # ---- Left: table + add/remove + properties --------------------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Title (written to first line of .geo)")
        self.title_edit.setText("Untitled geometry")
        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Title"))
        title_row.addWidget(self.title_edit, 1)
        left_layout.addLayout(title_row)

        # Curve table.
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([c[0] for c in COLUMNS])
        for i, (_, w) in enumerate(COLUMNS):
            self.table.setColumnWidth(i, w)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.table, 1)

        # +/- buttons.
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("＋  Add curve")
        self.add_btn.clicked.connect(self._on_add_row)
        self.del_btn = QPushButton("－  Remove selected")
        self.del_btn.clicked.connect(self._on_remove_row)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.del_btn)
        btn_row.addStretch(1)
        left_layout.addLayout(btn_row)

        # Materials block.
        mat_frame = QFrame()
        mat_frame.setFrameShape(QFrame.StyledPanel)
        mat_layout = QVBoxLayout(mat_frame)
        mat_layout.setContentsMargins(10, 8, 10, 10)
        mat_layout.addWidget(self._section_label("Materials  (referenced by IBC / IPN flags)"))

        # IBCs sub-table:  flag, R, X
        self.ibc_table = QTableWidget(0, 3)
        self.ibc_table.setHorizontalHeaderLabels(["IBC flag", "R (Ω)", "X (Ω)"])
        self.ibc_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ibc_table.verticalHeader().setVisible(False)
        self.ibc_table.setMaximumHeight(120)
        ibc_btns = QHBoxLayout()
        ibc_add = QPushButton("＋ IBC")
        ibc_del = QPushButton("－")
        ibc_add.clicked.connect(self._add_ibc_row)
        ibc_del.clicked.connect(self._del_ibc_row)
        ibc_btns.addWidget(ibc_add)
        ibc_btns.addWidget(ibc_del)
        ibc_btns.addStretch(1)
        mat_layout.addWidget(self.ibc_table)
        mat_layout.addLayout(ibc_btns)

        # Dielectrics sub-table.
        self.diel_table = QTableWidget(0, 5)
        self.diel_table.setHorizontalHeaderLabels(
            ["Diel flag", "ε′", "ε″ (positive = lossy)", "μ′", "μ″"]
        )
        self.diel_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.diel_table.verticalHeader().setVisible(False)
        self.diel_table.setMaximumHeight(120)
        diel_btns = QHBoxLayout()
        diel_add = QPushButton("＋ Dielectric")
        diel_del = QPushButton("－")
        diel_add.clicked.connect(self._add_diel_row)
        diel_del.clicked.connect(self._del_diel_row)
        diel_btns.addWidget(diel_add)
        diel_btns.addWidget(diel_del)
        diel_btns.addStretch(1)
        mat_layout.addWidget(self.diel_table)
        mat_layout.addLayout(diel_btns)

        left_layout.addWidget(mat_frame)

        # ---- Right: preview canvas -----------------------------------------
        self.canvas = PreviewCanvas()
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)
        right_layout.addWidget(self._section_label("Preview  ·  dark → light gradient runs from point 0 to point N"))
        right_layout.addWidget(self.canvas, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([840, 640])

        self.setCentralWidget(splitter)

        # Status bar.
        self.setStatusBar(QStatusBar())

    def _make_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)

        self.connect_action = QAction("Connect to Creo", self)
        self.connect_action.triggered.connect(self._on_connect)
        tb.addAction(self.connect_action)

        self.disconnect_action = QAction("Disconnect", self)
        self.disconnect_action.triggered.connect(self._on_disconnect)
        tb.addAction(self.disconnect_action)

        tb.addSeparator()

        export_action = QAction("Export .geo…", self)
        export_action.triggered.connect(self._on_export)
        tb.addAction(export_action)

        tb.addSeparator()

        # Spacer.
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding,
                             spacer.sizePolicy().verticalPolicy().Preferred)
        tb.addWidget(spacer)

        self.backend_label = QLabel("  backend: —  ")
        self.backend_label.setStyleSheet("color: #8a93a4;")
        tb.addWidget(self.backend_label)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        f = lbl.font()
        f.setPointSize(9)
        f.setBold(True)
        lbl.setFont(f)
        lbl.setStyleSheet("color: #c8d0dc; letter-spacing: 0.5px;")
        return lbl

    def _apply_theme(self) -> None:
        # Dark, refined, with a single accent — feels like a tool, not a toy.
        QApplication.instance().setStyleSheet(
            """
            QMainWindow, QWidget { background-color: #15181d; color: #d8dee9; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTableWidget {
                background: #1d2128;
                color: #e8edf4;
                border: 1px solid #2a2f38;
                border-radius: 4px;
                padding: 3px 6px;
                selection-background-color: #2c5e8a;
            }
            QPushButton {
                background: #232830;
                border: 1px solid #2f3540;
                border-radius: 4px;
                padding: 5px 12px;
                color: #d8dee9;
            }
            QPushButton:hover { background: #2a3340; border: 1px solid #6cb4ee; }
            QPushButton:pressed { background: #1d2128; }
            QHeaderView::section {
                background: #1a1e25; color: #8a93a4;
                padding: 6px; border: none; border-right: 1px solid #2a2f38;
                font-weight: bold; letter-spacing: 0.5px;
            }
            QTableWidget { gridline-color: #20242c; }
            QTableWidget::item:alternate { background-color: #181c22; }
            QTableWidget::item:selected { background-color: #2c5e8a; color: white; }
            QToolBar { background: #11141a; border-bottom: 1px solid #20242c; padding: 4px; spacing: 4px; }
            QStatusBar { background: #11141a; color: #8a93a4; border-top: 1px solid #20242c; }
            QFrame[frameShape="4"] { background: #181c22; border: 1px solid #20242c; border-radius: 6px; }
            """
        )

    # -----------------------------------------------------------------------
    # Connection
    # -----------------------------------------------------------------------

    def _on_connect(self) -> None:
        try:
            msg = self._bridge.connect()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Could not connect to Creo",
                f"{exc}\n\n"
                "Falling back to mock backend so you can preview the GUI.\n"
                "Install pywin32 + start Creo 10 to use the real connector.",
            )
            self._bridge = MockCreoBridge()
            try:
                msg = self._bridge.connect()
            except Exception as exc2:  # noqa: BLE001
                QMessageBox.critical(self, "Mock backend failed", str(exc2))
                return
        self.statusBar().showMessage(msg, 5000)
        self._refresh_status_bar()

    def _on_disconnect(self) -> None:
        self._bridge.disconnect()
        self.statusBar().showMessage("Disconnected.", 3000)
        self._refresh_status_bar()

    def _refresh_status_bar(self) -> None:
        name = self._bridge.display_name
        connected = "●  connected" if self._bridge.is_connected() else "○  disconnected"
        self.backend_label.setText(f"  {name}  ·  {connected}  ")

    # -----------------------------------------------------------------------
    # Curve table
    # -----------------------------------------------------------------------

    def _on_add_row(self) -> None:
        if not self._bridge.is_connected():
            QMessageBox.information(
                self, "Not connected",
                "Connect to Creo first (toolbar → Connect to Creo).\n"
                "If Creo isn't available the GUI will offer a mock backend.",
            )
            return
        color_idx = len(self._rows) % len(DEFAULT_PALETTE)
        row = CurveRow(color=QColor(DEFAULT_PALETTE[color_idx]))
        self._rows.append(row)
        self._append_table_row(row)
        self._sync_canvas()

    def _on_remove_row(self) -> None:
        sel = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in sel:
            if 0 <= r < len(self._rows):
                self._rows.pop(r)
                self.table.removeRow(r)
        self._sync_canvas()

    def _append_table_row(self, row: CurveRow) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setRowHeight(r, 34)

        # Curve name (read-only display).
        name_item = QTableWidgetItem("(not picked)")
        name_item.setFlags(name_item.flags() ^ Qt.ItemIsEditable)
        name_item.setForeground(QBrush(QColor("#8a93a4")))
        self.table.setItem(r, COL_CURVE, name_item)

        # Pick button.
        pick_btn = QPushButton("Pick from Creo")
        pick_btn.clicked.connect(lambda _=False, rr=row: self._on_pick_curve(rr))
        self.table.setCellWidget(r, COL_PICK, pick_btn)

        # # Points spinbox.
        npts = QSpinBox()
        npts.setRange(2, 4096)
        npts.setValue(row.n_points)
        npts.valueChanged.connect(lambda v, rr=row: self._on_npts_changed(rr, v))
        self.table.setCellWidget(r, COL_NPTS, npts)

        # Seg type combo.
        combo = QComboBox()
        for k, lbl in SEG_TYPE_LABELS.items():
            combo.addItem(lbl, userData=k)
        combo.setCurrentIndex(row.seg_type - 1)
        combo.currentIndexChanged.connect(
            lambda _=0, rr=row, cc=combo: self._on_segtype_changed(rr, cc.currentData())
        )
        self.table.setCellWidget(r, COL_SEGTYPE, combo)

        # IBC / IPN1 / IPN2 spinboxes.
        for col, attr in ((COL_IBC, "ibc_flag"), (COL_IPN1, "ipn1"), (COL_IPN2, "ipn2")):
            sb = QSpinBox()
            sb.setRange(0, 999)
            sb.setValue(getattr(row, attr))
            sb.valueChanged.connect(
                lambda v, rr=row, a=attr: setattr(rr, a, int(v))
            )
            self.table.setCellWidget(r, col, sb)

        # Color button.
        cb = ColorButton(row.color)
        cb.color_changed.connect(lambda c, rr=row: self._on_color_changed(rr, c))
        # Center it.
        cell = QWidget()
        cl = QHBoxLayout(cell)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addStretch(1)
        cl.addWidget(cb)
        cl.addStretch(1)
        self.table.setCellWidget(r, COL_COLOR, cell)

        # Status text.
        st = QTableWidgetItem("—")
        st.setFlags(st.flags() ^ Qt.ItemIsEditable)
        st.setForeground(QBrush(QColor("#8a93a4")))
        self.table.setItem(r, COL_STATUS, st)

    def _row_index(self, row: CurveRow) -> int:
        try:
            return self._rows.index(row)
        except ValueError:
            return -1

    def _on_pick_curve(self, row: CurveRow) -> None:
        if not self._bridge.is_connected():
            QMessageBox.information(self, "Not connected", "Connect to Creo first.")
            return
        try:
            sample = self._bridge.pick_curve()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Pick failed", f"{exc}\n\n{traceback.format_exc()}")
            return
        if sample is None:
            self.statusBar().showMessage("No curve picked.", 3000)
            return
        row.sample = sample
        # Re-sample at the requested point count from the spinbox.
        re_sampled = self._bridge.sample_curve(sample.curve_id, row.n_points)
        if re_sampled is not None:
            row.sample = re_sampled
        self._refresh_row_display(row)
        self._sync_canvas()

    def _on_npts_changed(self, row: CurveRow, n: int) -> None:
        row.n_points = int(n)
        if row.sample is None:
            return
        new_sample = self._bridge.sample_curve(row.sample.curve_id, row.n_points)
        if new_sample is not None:
            row.sample = new_sample
        self._refresh_row_display(row)
        self._sync_canvas()

    def _on_segtype_changed(self, row: CurveRow, seg_type: int) -> None:
        row.seg_type = int(seg_type)

    def _on_color_changed(self, row: CurveRow, c: QColor) -> None:
        row.color = QColor(c)
        self._sync_canvas()

    def _refresh_row_display(self, row: CurveRow) -> None:
        r = self._row_index(row)
        if r < 0:
            return
        if row.sample is None:
            self.table.item(r, COL_CURVE).setText("(not picked)")
            self.table.item(r, COL_STATUS).setText("—")
            return
        self.table.item(r, COL_CURVE).setText(row.sample.name)
        self.table.item(r, COL_CURVE).setForeground(QBrush(QColor("#e8edf4")))
        n = len(row.sample.points)
        closed = " · closed" if row.sample.closed else ""
        self.table.item(r, COL_STATUS).setText(f"{n} pts{closed}")

    def _sync_canvas(self) -> None:
        self.canvas.set_rows(self._rows)

    # -----------------------------------------------------------------------
    # Materials sub-tables
    # -----------------------------------------------------------------------

    def _add_ibc_row(self) -> None:
        r = self.ibc_table.rowCount()
        self.ibc_table.insertRow(r)
        flag = QSpinBox(); flag.setRange(1, 999); flag.setValue(r + 1)
        rohm = QDoubleSpinBox(); rohm.setRange(-1e6, 1e6); rohm.setDecimals(4); rohm.setValue(0.0)
        xohm = QDoubleSpinBox(); xohm.setRange(-1e6, 1e6); xohm.setDecimals(4); xohm.setValue(0.0)
        self.ibc_table.setCellWidget(r, 0, flag)
        self.ibc_table.setCellWidget(r, 1, rohm)
        self.ibc_table.setCellWidget(r, 2, xohm)

    def _del_ibc_row(self) -> None:
        rows = sorted({i.row() for i in self.ibc_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.ibc_table.removeRow(r)

    def _add_diel_row(self) -> None:
        r = self.diel_table.rowCount()
        self.diel_table.insertRow(r)
        flag = QSpinBox(); flag.setRange(1, 999); flag.setValue(r + 1)
        eps_r = QDoubleSpinBox(); eps_r.setRange(0.0, 1e6); eps_r.setDecimals(4); eps_r.setValue(1.0)
        eps_i = QDoubleSpinBox(); eps_i.setRange(0.0, 1e6); eps_i.setDecimals(4); eps_i.setValue(0.0)
        mu_r  = QDoubleSpinBox(); mu_r.setRange(0.0, 1e6); mu_r.setDecimals(4); mu_r.setValue(1.0)
        mu_i  = QDoubleSpinBox(); mu_i.setRange(0.0, 1e6); mu_i.setDecimals(4); mu_i.setValue(0.0)
        self.diel_table.setCellWidget(r, 0, flag)
        self.diel_table.setCellWidget(r, 1, eps_r)
        self.diel_table.setCellWidget(r, 2, eps_i)
        self.diel_table.setCellWidget(r, 3, mu_r)
        self.diel_table.setCellWidget(r, 4, mu_i)

    def _del_diel_row(self) -> None:
        rows = sorted({i.row() for i in self.diel_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.diel_table.removeRow(r)

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def _on_export(self) -> None:
        ready_rows = [r for r in self._rows if r.sample and len(r.sample.points) >= 2]
        if not ready_rows:
            QMessageBox.information(self, "Nothing to export",
                                    "Add at least one curve and pick it from Creo first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export .geo", "geometry.geo", "Geometry files (*.geo);;All files (*)"
        )
        if not path:
            return
        try:
            text = self._format_geo(ready_rows)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", f"{exc}\n\n{traceback.format_exc()}")
            return
        self.statusBar().showMessage(f"Wrote {path}", 6000)

    def _format_geo(self, rows: List[CurveRow]) -> str:
        lines: List[str] = []
        title = self.title_edit.text().strip() or "Untitled"
        lines.append(f"Title: {title}")

        for row in rows:
            sample = row.sample
            assert sample is not None
            seg_name = sample.name.replace(" ", "_") or "segment"
            lines.append(f"Segment: {seg_name} line")
            # properties: <seg_type> <n_panels> <ang_deg> <ibc_flag> <ipn1> <ipn2>
            #   n_panels: pulse_basis convention - negative => panels-per-wavelength,
            #             positive => explicit per-primitive panel count.
            #   We default to "1 panel per polyline segment" since the points have
            #   already been sampled at the user's chosen density.
            n_panels = 1
            ang_deg = 0.0
            lines.append(
                f"properties: {row.seg_type} {n_panels} {ang_deg:.1f} "
                f"{row.ibc_flag} {row.ipn1} {row.ipn2}"
            )
            pts = sample.points
            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                lines.append(f"{x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f}")

        # IBCS section.
        lines.append("IBCS:")
        for r in range(self.ibc_table.rowCount()):
            flag = int(self.ibc_table.cellWidget(r, 0).value())
            R = float(self.ibc_table.cellWidget(r, 1).value())
            X = float(self.ibc_table.cellWidget(r, 2).value())
            lines.append(f"{flag} {R:.4f} {X:.4f} 0.0")

        # Dielectrics section.
        lines.append("Dielectrics:")
        for r in range(self.diel_table.rowCount()):
            flag = int(self.diel_table.cellWidget(r, 0).value())
            er = float(self.diel_table.cellWidget(r, 1).value())
            ei = float(self.diel_table.cellWidget(r, 2).value())
            mr = float(self.diel_table.cellWidget(r, 3).value())
            mi = float(self.diel_table.cellWidget(r, 4).value())
            lines.append(f"{flag} {er:.4f} {ei:.4f} {mr:.4f} {mi:.4f}")

        return "\n".join(lines) + "\n"


# ===========================================================================
# Bootstrap
# ===========================================================================

def _choose_initial_bridge() -> CreoBridge:
    """Try the real backend on Windows; fall back to mock everywhere else."""
    if sys.platform == "win32":
        try:
            return Win32ComCreoBridge()
        except Exception:  # noqa: BLE001
            pass
    return MockCreoBridge()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("creo_geo_extractor")
    # Prefer a refined system font where possible.
    try:
        f = QFont("Inter", 10)
        if not QFontDatabase.families().__contains__("Inter"):
            f = QFont("Segoe UI", 10) if sys.platform == "win32" else QFont("SF Pro Text", 10)
        app.setFont(f)
    except Exception:  # noqa: BLE001
        pass

    bridge = _choose_initial_bridge()
    win = MainWindow(bridge)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
