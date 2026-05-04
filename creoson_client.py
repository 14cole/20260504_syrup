"""
creoson_client.py
=================

Self-contained Creoson client.  No pip install needed — uses only the
Python standard library (urllib + json).  Drop this file into your
project area and import from it directly:

    from creoson_client import CreosonClient, CreosonBridge

Creoson is PTC's free JSON-over-HTTP listener for Creo
(https://www.simplifiedlogic.com/products/creoson).  You still need:
  1. The CreosonServer.exe / CreosonServerWithSetup.exe running locally
     (default port 9056).  This is a separate download, but it is a
     self-contained .zip — no installer, no admin rights, no Python
     packages.  Unzip it anywhere and double-click the server exe.
  2. Creo running (any way you normally launch it).
  3. In the Creoson server window, click "Start Creoson Server" and
     then "Connect to Creo".  After that, this script can talk to it.

This file provides:
  * CreosonClient          - low-level call(family, command, **kwargs)
                             plus thin convenience wrappers
  * CreosonBridge          - implements the same interface as
                             Win32ComCreoBridge in creo_geo_extractor.py,
                             so you can swap one for the other
  * a tiny __main__ test   - run `python creoson_client.py` to ping
                             the server and list selected curves


Pasting into the GUI
--------------------
At the bottom of creo_geo_extractor.py, replace::

    def _choose_initial_bridge() -> CreoBridge:
        if sys.platform == "win32":
            try:
                return Win32ComCreoBridge()
            except Exception:
                pass
        return MockCreoBridge()

with::

    from creoson_client import CreosonBridge

    def _choose_initial_bridge() -> CreoBridge:
        try:
            return CreosonBridge()
        except Exception:
            return MockCreoBridge()

That's the only change needed.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Low-level JSON-RPC client
# ---------------------------------------------------------------------------

class CreosonError(RuntimeError):
    """Raised when the Creoson server returns status.error == True."""


class CreosonClient:
    """
    Minimal Creoson client.  Speaks the same JSON envelope as creopyson.

    Every Creoson request is::

        {
          "sessionId": "<token>",
          "command":   "<family>",
          "function":  "<command>",
          "data":      { ... command-specific payload ... }
        }

    and the response is::

        {
          "status": { "error": false, "message": "" },
          "data":   { ... result ... }
        }
    """

    DEFAULT_URL = "http://localhost:9056/creoson"

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 30.0) -> None:
        self.url = url
        self.timeout = timeout
        self.session_id: Optional[str] = None

    # -- raw call ----------------------------------------------------------

    def call(self, family: str, command: str, **data: Any) -> Dict[str, Any]:
        """Send one JSON-RPC request; return the `data` dict on success."""
        payload: Dict[str, Any] = {
            "command": family,
            "function": command,
            "data": dict(data),
        }
        if self.session_id is not None:
            payload["sessionId"] = self.session_id

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.URLError as exc:
            raise CreosonError(
                f"Could not reach Creoson server at {self.url}.  "
                f"Is CreosonServer.exe running and connected to Creo?  ({exc})"
            ) from exc

        try:
            envelope = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise CreosonError(f"Malformed JSON from Creoson: {exc}") from exc

        status = envelope.get("status") or {}
        if status.get("error"):
            raise CreosonError(
                f"Creoson {family}.{command} failed: "
                f"{status.get('message') or '(no message)'}"
            )
        return envelope.get("data") or {}

    # -- session management ------------------------------------------------

    def connect(self) -> str:
        """Create a Creoson session and verify Creo is reachable."""
        data = self.call("connection", "connect")
        self.session_id = data.get("sessionId")
        if not self.session_id:
            raise CreosonError("Creoson returned no sessionId on connect.")
        # Sanity ping — fails clearly if Creoson isn't itself attached to Creo.
        self.call("connection", "is_creo_running")
        return self.session_id

    def disconnect(self) -> None:
        if self.session_id is None:
            return
        try:
            self.call("connection", "disconnect")
        except CreosonError:
            pass
        self.session_id = None

    def is_connected(self) -> bool:
        return self.session_id is not None

    # -- convenience helpers used by the GUI ------------------------------

    def get_active_model(self) -> Optional[str]:
        """Return the active model's file name, or None."""
        try:
            data = self.call("file", "get_active")
        except CreosonError:
            return None
        return data.get("file")

    def list_selected_curves(self) -> List[Dict[str, Any]]:
        """
        Return information about whatever is currently selected in Creo,
        filtered to curve / edge entities.

        Each entry has at least:
          - 'name'     : human readable label
          - 'item_id'  : opaque integer/string ID Creoson uses to
                         re-select / sample the curve
          - 'type'     : the Creoson type string ('curve', 'edge', ...)
        """
        # Creoson's `bom.get_paths` and `feature.list_selected` both touch
        # this; the most portable is the geometry-side `geometry.get_edges`
        # for selected solids, but for sketch-curves the right call is
        # `interface.export_program`-friendly `interface.list_selected`.
        # Different Creoson builds expose slightly different endpoints, so
        # we try them in order of preference.
        for family, command in (
            ("interface", "list_selected"),
            ("feature", "list_selected"),
            ("geometry", "list_selected"),
        ):
            try:
                data = self.call(family, command)
            except CreosonError:
                continue
            items = data.get("items") or data.get("features") or []
            curves = [
                it for it in items
                if str(it.get("type", "")).lower() in ("curve", "edge", "datum_curve", "sketch")
            ]
            if curves:
                return curves
        return []

    def sample_curve(
        self,
        item_id: Any,
        n_points: int,
        owner_file: Optional[str] = None,
    ) -> List[Tuple[float, float]]:
        """
        Sample (x, y) along a curve at uniformly-spaced parameters.

        Creoson's geometry-evaluation endpoint is `geometry.eval_curve`.
        It accepts:
            file       : the model file containing the curve
            curve_id   : item id from list_selected_curves
            param      : parameter in [0, 1]
        and returns:
            point      : [x, y, z]
        """
        n = max(2, int(n_points))
        pts: List[Tuple[float, float]] = []
        for i in range(n):
            t = i / (n - 1)
            data = self.call(
                "geometry",
                "eval_curve",
                file=owner_file or "",
                curve_id=item_id,
                param=t,
            )
            point = data.get("point") or []
            if len(point) < 2:
                raise CreosonError(
                    f"eval_curve returned malformed point at t={t}: {point!r}"
                )
            pts.append((float(point[0]), float(point[1])))
        return pts


# ---------------------------------------------------------------------------
# Bridge that plugs into creo_geo_extractor.py's CreoBridge ABC
# ---------------------------------------------------------------------------

# We re-define the ABC locally so this module is fully self-contained.
# It's structurally identical to CreoBridge in the GUI file; isinstance()
# checks are not used, so the duck-typed compatibility is sufficient.

@dataclass
class CurveSample:
    curve_id: str
    name: str
    points: List[Tuple[float, float]]
    closed: bool = False


class CreosonBridge:
    """Drop-in replacement for Win32ComCreoBridge."""

    def __init__(self, url: str = CreosonClient.DEFAULT_URL) -> None:
        self._client = CreosonClient(url=url)
        self._curve_index: Dict[str, Dict[str, Any]] = {}
        # ^ maps our generated curve_id -> { 'item_id', 'file', 'name' }

    # The GUI's CreoBridge contract:
    #   .is_connected(), .connect(), .disconnect(),
    #   .pick_curve(), .sample_curve(curve_id, n_points)
    #   .display_name (property)

    @property
    def display_name(self) -> str:
        return "CreosonBridge"

    def is_connected(self) -> bool:
        return self._client.is_connected()

    def connect(self) -> str:
        sid = self._client.connect()
        active = self._client.get_active_model()
        suffix = f", active model: {active}" if active else ""
        return f"Connected to Creoson (session {sid[:8]}…){suffix}"

    def disconnect(self) -> None:
        self._client.disconnect()
        self._curve_index.clear()

    def pick_curve(self) -> Optional[CurveSample]:
        """
        Workflow: user pre-selects ONE edge/curve in Creo, then clicks
        'Pick from Creo' in the GUI.  We grab the current selection.
        """
        if not self._client.is_connected():
            raise RuntimeError("Not connected to Creoson.")
        curves = self._client.list_selected_curves()
        if not curves:
            raise RuntimeError(
                "No curve currently selected in Creo.  Click an edge or "
                "sketch curve in Creo first, then click 'Pick from Creo'."
            )
        first = curves[0]
        # Build a stable id under our control so we can re-sample later.
        item_id = first.get("item_id") or first.get("id") or first.get("name")
        owner = first.get("file") or self._client.get_active_model() or ""
        name = str(first.get("name") or item_id or "curve")
        cid = f"creoson_{item_id}"
        self._curve_index[cid] = {
            "item_id": item_id,
            "file": owner,
            "name": name,
        }
        return self.sample_curve(cid, 16)

    def sample_curve(self, curve_id: str, n_points: int) -> Optional[CurveSample]:
        info = self._curve_index.get(curve_id)
        if info is None:
            return None
        try:
            pts = self._client.sample_curve(
                item_id=info["item_id"],
                n_points=n_points,
                owner_file=info.get("file"),
            )
        except CreosonError as exc:
            raise RuntimeError(str(exc)) from exc
        if len(pts) < 2:
            return None
        closed = math.hypot(
            pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]
        ) < 1e-9
        return CurveSample(
            curve_id=curve_id,
            name=info.get("name") or curve_id,
            points=pts,
            closed=closed,
        )


# ---------------------------------------------------------------------------
# CLI smoke test — run as a script to verify the server is reachable
# ---------------------------------------------------------------------------

def _cli_smoke_test() -> int:
    """python creoson_client.py  ->  ping the server, print state."""
    print(f"Probing Creoson at {CreosonClient.DEFAULT_URL} ...")
    client = CreosonClient()
    try:
        sid = client.connect()
    except CreosonError as exc:
        print(f"  FAIL: {exc}")
        print()
        print("Checklist:")
        print("  1. Is CreosonServer.exe running?")
        print("  2. In its window, did you click 'Start Creoson Server'?")
        print("  3. Did you click 'Connect to Creo' and see it succeed?")
        print("  4. Is Creo open with a model active?")
        return 1

    print(f"  connected.  sessionId = {sid}")
    active = client.get_active_model()
    print(f"  active model: {active or '(none)'}")
    try:
        curves = client.list_selected_curves()
    except CreosonError as exc:
        print(f"  could not list selection: {exc}")
        curves = []
    print(f"  selected curves: {len(curves)}")
    for c in curves[:5]:
        print(f"    - {c.get('name')!r}  type={c.get('type')!r}  id={c.get('item_id')!r}")

    client.disconnect()
    print("  disconnected.  done.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli_smoke_test())
