"""
aruco_tracker.py
================
4K ArUco Marker Tracker with Robot Path Calculation.

Requirements:  opencv-contrib-python >= 4.8  numpy

Usage:
    python aruco_tracker.py --camera 4 --dict 4X4_50
    python aruco_tracker.py --camera 4 --dict 4X4_50 --room-w 3 --room-h 4
    python aruco_tracker.py --camera 4 --dict 4X4_50 --corners corners.json

Keyboard controls (while running):
    Q  — quit
    R  — enter ROI selection mode (click 4 corners of the white wall area)
    C  — clear all path histories
    S  — save full-res screenshot to disk
    P  — print current path data to console

ROI selection:
    Click the 4 corners of the white wall area in this order:
        1. Top-left
        2. Top-right
        3. Bottom-right
        4. Bottom-left
    The view will snap to the warped 3×4 m region.
    Corners are saved to corners.json automatically.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
import time
import argparse
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path


SESSIONS_DIR = Path("sessions")   # where session data + videos are saved


# ---------------------------------------------------------------------------
# Bump Detector
# ---------------------------------------------------------------------------

class BumpDetector:
    """
    Detects collisions from position data alone — no speed required.

    Two event types:
      • "wall"  — robot centre is within WALL_MARGIN_M of any arena boundary.
      • "robot" — two robot centres are within ROBOT_DIST_M of each other.

    A COOLDOWN_S debounce window prevents the same contact from being counted
    repeatedly while two objects remain in contact.
    """
    WALL_MARGIN_M = 0.15   # m from wall edge counted as touching
    ROBOT_DIST_M  = 0.45   # m between centres counted as touching
    COOLDOWN_S    = 2.0    # seconds before the same contact counts again

    def __init__(self):
        self._wall_contact:  dict[int, float]   = {}  # mid → last wall-hit time
        self._robot_contact: dict[tuple, float] = {}  # (mid_a, mid_b) → last hit time
        self.bumps: dict[int, list] = defaultdict(list)
        self._lock = threading.Lock()

    def update_all(self, positions_m: dict, room_w: float, room_h: float, t: float) -> list:
        """
        Call once per frame with ALL visible marker positions.
        positions_m: {marker_id: (x_m, y_m)}
        Returns list of new bump events this frame.
        """
        new_bumps = []
        with self._lock:
            # ── 1. Wall collisions ─────────────────────────────────────────
            for mid, (x, y) in positions_m.items():
                near_wall = (
                    x < self.WALL_MARGIN_M or
                    x > room_w - self.WALL_MARGIN_M or
                    y < self.WALL_MARGIN_M or
                    y > room_h - self.WALL_MARGIN_M
                )
                if near_wall:
                    last = self._wall_contact.get(mid, 0.0)
                    if t - last >= self.COOLDOWN_S:
                        self._wall_contact[mid] = t
                        evt = {"t": round(t, 2), "type": "wall",
                               "pos_m": [round(x, 3), round(y, 3)]}
                        self.bumps[mid].append(evt)
                        new_bumps.append((mid, evt))
                        print(f"[bump] marker {mid} — wall  @ ({x:.2f}, {y:.2f}) m")
                else:
                    self._wall_contact.pop(mid, None)   # reset when clear

            # ── 2. Robot–robot collisions ──────────────────────────────────
            mids = list(positions_m.keys())
            for i in range(len(mids)):
                for j in range(i + 1, len(mids)):
                    a, b = mids[i], mids[j]
                    pa, pb = positions_m[a], positions_m[b]
                    dist = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
                    key  = (min(a, b), max(a, b))

                    if dist < self.ROBOT_DIST_M:
                        last = self._robot_contact.get(key, 0.0)
                        if t - last >= self.COOLDOWN_S:
                            self._robot_contact[key] = t
                            for mid, pos in [(a, pa), (b, pb)]:
                                other = b if mid == a else a
                                evt = {"t": round(t, 2), "type": "robot",
                                       "pos_m": [round(pos[0], 3), round(pos[1], 3)],
                                       "other_id": int(other),
                                       "dist_m": round(dist, 3)}
                                self.bumps[mid].append(evt)
                                new_bumps.append((mid, evt))
                            print(f"[bump] markers {a}↔{b} — robot collision  dist={dist:.2f} m")
                    else:
                        self._robot_contact.pop(key, None)   # reset when apart

        return new_bumps

    # Legacy single-marker update — kept so existing call sites don't break;
    # does nothing (all detection now happens in update_all).
    def update(self, mid: int, speed: float, t: float, pos_m=None) -> bool:
        return False

    def bump_count(self, mid: int) -> int:
        with self._lock:
            return len(self.bumps[mid])

    def total_bumps(self) -> int:
        with self._lock:
            return sum(len(v) for v in self.bumps.values())

    def export(self) -> dict:
        with self._lock:
            return {str(mid): list(events) for mid, events in self.bumps.items()}

    def clear(self, mid=None) -> None:
        with self._lock:
            if mid is not None:
                self.bumps.pop(mid, None)
                self._wall_contact.pop(mid, None)
                keys = [k for k in self._robot_contact if mid in k]
                for k in keys:
                    self._robot_contact.pop(k, None)
            else:
                self.bumps.clear()
                self._wall_contact.clear()
                self._robot_contact.clear()


# ---------------------------------------------------------------------------
# Path Tracker
# ---------------------------------------------------------------------------

class PathTracker:
    def __init__(self, max_history: int = 150, smoothing_window: int = 7):
        self.smoothing_window = smoothing_window
        self._centres:  dict[int, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._times:    dict[int, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._speeds:   dict[int, deque] = defaultdict(lambda: deque(maxlen=30))

    def update(self, marker_id: int, centre: tuple, t: float) -> None:
        prev = self._centres[marker_id]
        if prev:
            dt = t - self._times[marker_id][-1]
            if dt > 0:
                dp = np.linalg.norm(np.array(centre) - np.array(prev[-1]))
                self._speeds[marker_id].append(dp / dt)
        self._centres[marker_id].append(centre)
        self._times[marker_id].append(t)

    def smoothed_path(self, marker_id: int) -> list:
        raw = list(self._centres[marker_id])
        n   = len(raw)
        if n < 2:
            return raw
        hw  = self.smoothing_window // 2
        out = []
        for i in range(n):
            lo, hi = max(0, i - hw), min(n, i + hw + 1)
            avg = np.mean(raw[lo:hi], axis=0).astype(int)
            out.append(tuple(avg))
        return out

    def speed(self, marker_id: int) -> float:
        s = self._speeds.get(marker_id)
        return float(np.mean(list(s)[-5:])) if s else 0.0

    def heading_deg(self, marker_id: int) -> float:
        path = list(self._centres[marker_id])
        if len(path) < 2:
            return 0.0
        dp = np.array(path[-1]) - np.array(path[-2])
        return float(np.degrees(np.arctan2(-dp[1], dp[0])))

    def odometry_px(self, marker_id: int) -> float:
        path = list(self._centres[marker_id])
        return sum(
            np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
            for i in range(1, len(path))
        )

    def last_centre(self, marker_id: int):
        c = self._centres.get(marker_id)
        return c[-1] if c else None

    def export_paths(self) -> dict:
        return {mid: [list(p) for p in pts] for mid, pts in self._centres.items()}

    def clear(self, marker_id=None) -> None:
        stores = (self._centres, self._times, self._speeds)
        if marker_id is not None:
            for s in stores:
                s.pop(marker_id, None)
        else:
            for s in stores:
                s.clear()


# ---------------------------------------------------------------------------
# Perspective ROI
# ---------------------------------------------------------------------------

class PerspectiveROI:
    """
    Warps the camera frame so that the clicked white-wall quadrilateral
    fills a canonical (room_w × room_h) metre view at a fixed pixel density.

    Click order: top-left → top-right → bottom-right → bottom-left
    """

    PX_PER_METER = 200   # output resolution density

    def __init__(self, room_w: float = 4.0, room_h: float = 3.0):
        self.room_w    = room_w
        self.room_h    = room_h
        self.out_w     = int(room_w * self.PX_PER_METER)
        self.out_h     = int(room_h * self.PX_PER_METER)
        self.src_pts:  np.ndarray | None = None   # 4 clicked points in camera coords
        self.M:        np.ndarray | None = None   # 3×3 homography
        self.M_inv:    np.ndarray | None = None   # inverse (warped → camera)
        self._pending: list = []                  # points being collected

    # ── destination corners (canonical room, top-left origin) ──────────────
    @property
    def _dst_pts(self) -> np.ndarray:
        w, h = self.out_w, self.out_h
        return np.array(
            [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
        )

    # ── public API ──────────────────────────────────────────────────────────
    @property
    def active(self) -> bool:
        return self.M is not None

    @property
    def collecting(self) -> bool:
        return 0 < len(self._pending) < 4

    def start_collection(self) -> None:
        self._pending = []

    def add_point(self, x: int, y: int) -> bool:
        """Add a corner click. Returns True when all 4 collected."""
        self._pending.append([x, y])
        if len(self._pending) == 4:
            self.src_pts = np.array(self._pending, dtype=np.float32)
            self.M       = cv2.getPerspectiveTransform(self.src_pts, self._dst_pts)
            self.M_inv   = cv2.getPerspectiveTransform(self._dst_pts, self.src_pts)
            self._pending = []
            return True
        return False

    def warp(self, frame: np.ndarray) -> np.ndarray:
        if self.M is None:
            return frame
        return cv2.warpPerspective(frame, self.M, (self.out_w, self.out_h))

    def to_metres(self, px: tuple) -> tuple[float, float]:
        """Warped-image pixel → real-world metres."""
        x_m = px[0] / self.PX_PER_METER
        y_m = px[1] / self.PX_PER_METER
        return round(x_m, 3), round(y_m, 3)

    def save(self, path: str = "corners.json") -> None:
        if self.src_pts is None:
            return
        Path(path).write_text(
            json.dumps({"corners": self.src_pts.tolist(),
                        "room_w": self.room_w, "room_h": self.room_h}, indent=2)
        )
        print(f"[roi] corners saved to {path}")

    def clear(self) -> None:
        self.src_pts  = None
        self.M        = None
        self.M_inv    = None
        self._pending = []

    def load(self, path: str) -> bool:
        try:
            data         = json.loads(Path(path).read_text())
            self.room_w  = data.get("room_w", self.room_w)
            self.room_h  = data.get("room_h", self.room_h)
            self.out_w   = int(self.room_w * self.PX_PER_METER)
            self.out_h   = int(self.room_h * self.PX_PER_METER)
            self.src_pts = np.array(data["corners"], dtype=np.float32)
            self.M       = cv2.getPerspectiveTransform(self.src_pts, self._dst_pts)
            self.M_inv   = cv2.getPerspectiveTransform(self._dst_pts, self.src_pts)
            print(f"[roi] loaded corners from {path}")
            return True
        except Exception as e:
            print(f"[roi] could not load {path}: {e}")
            return False

    def draw_overlay(self, frame: np.ndarray, scale: float = 1.0) -> None:
        """Draw collected / pending corners on the raw camera preview."""
        pts = self._pending if self.collecting else (
            self.src_pts.tolist() if self.src_pts is not None else []
        )
        labels = ["TL", "TR", "BR", "BL"]
        colors = [(0, 217, 255), (255, 152, 0), (57, 255, 20), (255, 0, 200)]
        for i, pt in enumerate(pts):
            sx, sy = int(pt[0] * scale), int(pt[1] * scale)
            cv2.circle(frame, (sx, sy), 8, colors[i], -1)
            cv2.putText(frame, labels[i], (sx + 10, sy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[i], 2)
        if len(pts) >= 2 and not self.collecting:
            scaled = [(int(p[0] * scale), int(p[1] * scale)) for p in pts]
            cv2.polylines(frame, [np.array(scaled)], True, (0, 217, 255), 2)


# ---------------------------------------------------------------------------
# Red corner auto-detection
# ---------------------------------------------------------------------------

def detect_gray_surface_corners(frame: np.ndarray) -> np.ndarray | None:
    """
    Detect the 4 corners of the gray floor surface automatically.
    Segments the gray area, fits a rectangle, returns corners as
    [top-left, top-right, bottom-right, bottom-left].
    Returns None if no clear gray surface is found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Gray: low saturation, mid-to-high value
    mask = cv2.inRange(hsv, (0, 0, 60), (180, 60, 220))

    # Remove noise and fill holes
    k    = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep the largest gray blob
    fh, fw = frame.shape[:2]
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < (fw * fh) * 0.05:
        print("[roi] gray surface too small — check lighting")
        return None

    # Fit a rectangle to the blob
    rect = cv2.minAreaRect(largest)
    box  = cv2.boxPoints(rect)
    box  = np.array(box, dtype=np.float32)

    # Order: TL, TR, BR, BL
    box = box[np.argsort(box[:, 1])]
    top_two    = box[:2][np.argsort(box[:2, 0])]
    bottom_two = box[2:][np.argsort(box[2:, 0])]
    tl, tr = top_two
    bl, br = bottom_two

    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_red_corners(frame: np.ndarray) -> np.ndarray | None:
    """
    Detect 4 red markers on the white walls, fit a minimum-area rectangle
    to their centroids, and return the 4 corners ordered as:
    [top-left, top-right, bottom-right, bottom-left].
    Returns None if fewer than 4 blobs are found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps around 0/180 in HSV
    # Lower saturation threshold helps on gray/dark floor backgrounds
    mask1 = cv2.inRange(hsv, (0,   80, 50), (10,  255, 255))
    mask2 = cv2.inRange(hsv, (160, 80, 50), (180, 255, 255))
    mask  = cv2.bitwise_or(mask1, mask2)

    # Clean up noise
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area — ignore tiny specks and huge blobs
    fh, fw = frame.shape[:2]
    min_area = (fw * fh) * 0.00005
    max_area = (fw * fh) * 0.05

    centres = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            centres.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])

    if len(centres) < 4:
        print(f"[roi] found {len(centres)} red blobs, need 4")
        return None

    # If more than 4, keep the 4 outermost (one per image quadrant)
    if len(centres) > 4:
        pts = np.array(centres, dtype=np.float32)
        cx, cy = fw / 2, fh / 2
        quadrant_centres = [
            (cx * 0.5,  cy * 0.5),   # TL quadrant
            (cx * 1.5,  cy * 0.5),   # TR quadrant
            (cx * 1.5,  cy * 1.5),   # BR quadrant
            (cx * 0.5,  cy * 1.5),   # BL quadrant
        ]
        selected = []
        for qc in quadrant_centres:
            closest = min(pts, key=lambda p: (p[0]-qc[0])**2 + (p[1]-qc[1])**2)
            selected.append(closest.tolist())
        centres = selected

    # Fit a minimum-area rectangle to the 4 blob centroids
    # This enforces the rectangle constraint regardless of small detection errors
    pts = np.array(centres, dtype=np.float32)
    rect = cv2.minAreaRect(pts)          # (center, (w,h), angle)
    box  = cv2.boxPoints(rect)           # 4 corners of the fitted rectangle
    box  = np.array(box, dtype=np.float32)

    # Order corners: top-left, top-right, bottom-right, bottom-left
    # Sort by y → top two and bottom two, then sort each pair by x
    box = box[np.argsort(box[:, 1])]
    top_two    = box[:2][np.argsort(box[:2, 0])]
    bottom_two = box[2:][np.argsort(box[2:, 0])]
    tl, tr = top_two
    bl, br = bottom_two

    # Return all four corners; caller decides orientation
    return np.array([tl, tr, br, bl], dtype=np.float32)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_COLORS = [
    (0, 217, 255), (0, 152, 255), (57, 255, 20), (255, 0, 200),
    (0, 255, 185), (255, 200, 0), (180, 60, 255), (255, 80, 80),
]

def _color(mid: int) -> tuple:
    return _COLORS[mid % len(_COLORS)]

def _default_K(w: int, h: int) -> np.ndarray:
    f = max(w, h)
    return np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Main Tracker
# ---------------------------------------------------------------------------

class ArucoRobotTracker:
    def __init__(
        self,
        camera_index:     int   = 0,
        resolution:       tuple = (3840, 2160),
        fps:              int   = 30,
        aruco_dict_type:  int   = cv2.aruco.DICT_4X4_50,
        camera_matrix:    np.ndarray | None = None,
        dist_coeffs:      np.ndarray | None = None,
        marker_length:    float = 0.10,
        path_history:     int   = 150,
        display_scale:    float = 0.5,
        room_w:           float = 3.0,
        room_h:           float = 4.0,
        corners_file:     str   = "corners.json",
    ):
        self.camera_index  = camera_index
        self.resolution    = resolution
        self.fps           = fps
        self.marker_length = marker_length
        self.display_scale = display_scale
        self._K            = camera_matrix
        self._D            = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.corners_file  = corners_file

        aruco_dict  = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        params      = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        self.path_tracker   = PathTracker(max_history=path_history)
        self.bump_detector  = BumpDetector()
        self.roi            = PerspectiveROI(room_w=room_w, room_h=room_h)

        self._frame_count = 0
        self._fps_frames  = 0
        self._fps_time    = time.time()
        self._current_fps = 0.0
        self._cap         = None

        self._last_raw_corners: np.ndarray | None = None  # last 4 detected corners
        self._corner_rotation  = 0                        # 0-3, cycles with F key

        # mouse state (for ROI corner clicking on the raw preview)
        self._raw_frame_for_click = None
        self._click_scale = display_scale

        # Session / recording state
        self._session:      dict | None = None
        self._video_writer: cv2.VideoWriter | None = None
        self._state_lock    = threading.Lock()          # guards _session reads

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _open_camera(self) -> tuple[int, int]:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self.camera_index}.")

        w, h = self.resolution
        cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   2)

        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[camera] {aw}×{ah} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")

        if self._K is None:
            self._K = _default_K(aw, ah)
            print("[camera] using estimated intrinsics")

        self._cap = cap
        return aw, ah

    # ------------------------------------------------------------------
    # Mouse callback — collect ROI corners on the raw preview window
    # ------------------------------------------------------------------

    def _mouse_cb(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not self.roi.collecting:
            return
        # x, y are in preview (scaled) coords → undo scale to get camera coords
        scale = self._click_scale
        cx, cy = int(x / scale), int(y / scale)
        done = self.roi.add_point(cx, cy)
        n = len(self.roi._pending) if not done else 4
        print(f"[roi] corner {n}/4 at camera ({cx}, {cy})")
        if done:
            self.roi.save(self.corners_file)
            print("[roi] ROI set — now showing warped room view")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_path(self, frame: np.ndarray, mid: int) -> None:
        path = self.path_tracker.smoothed_path(mid)
        if len(path) < 2:
            return
        color = _color(mid)
        n = len(path)
        for i in range(1, n):
            alpha = i / n
            c     = tuple(int(ch * alpha) for ch in color)
            cv2.line(frame, path[i-1], path[i], c, max(1, int(3*alpha)), cv2.LINE_AA)

    def _draw_info_box(self, frame: np.ndarray, mid: int,
                       cx: int, cy: int, tvec=None, pos_m=None) -> None:
        color   = _color(mid)
        speed   = self.path_tracker.speed(mid)
        heading = self.path_tracker.heading_deg(mid)
        odo     = self.path_tracker.odometry_px(mid)

        lines = [
            f"ID {mid}",
            f"spd  {speed:6.1f} px/s",
            f"hdg  {heading:+6.1f}°",
            f"odo  {odo:7.0f} px",
        ]
        if pos_m is not None:
            lines.append(f"pos  {pos_m[0]:.2f} m, {pos_m[1]:.2f} m")
        elif tvec is not None:
            t = tvec.flatten()
            lines += [f"X {t[0]:+.3f} m", f"Y {t[1]:+.3f} m", f"Z {t[2]:+.3f} m"]

        lh, bw = 18, 165
        bh = len(lines) * lh + 6
        bx, by = cx + 16, cy - 14

        roi = frame[max(0, by-2): by+bh, max(0, bx-4): bx+bw]
        if roi.size:
            frame[max(0, by-2): by+bh, max(0, bx-4): bx+bw] = (roi * 0.35).astype(np.uint8)

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (bx, by + i*lh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        # heading arrow
        if len(self.path_tracker.smoothed_path(mid)) >= 2:
            rad = np.radians(heading)
            ex, ey = int(cx + 50*np.cos(rad)), int(cy - 50*np.sin(rad))
            cv2.arrowedLine(frame, (cx, cy), (ex, ey), color, 2,
                            tipLength=0.35, line_type=cv2.LINE_AA)

    def _draw_grid(self, frame: np.ndarray) -> None:
        """1 m grid lines on the warped room view."""
        h, w = frame.shape[:2]
        ppm  = self.roi.PX_PER_METER
        color = (50, 50, 50)
        for xm in range(1, int(self.roi.room_w)):
            x = xm * ppm
            cv2.line(frame, (x, 0), (x, h), color, 1)
            cv2.putText(frame, f"{xm}m", (x+3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
        for ym in range(1, int(self.roi.room_h)):
            y = ym * ppm
            cv2.line(frame, (0, y), (w, y), color, 1)
            cv2.putText(frame, f"{ym}m", (3, y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
        # border
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 217, 255), 2)
        cv2.putText(frame, f"{self.roi.room_w}m × {self.roi.room_h}m",
                    (6, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 217, 255), 1)

    def _draw_hud(self, frame, n_markers, aw, ah, roi_active) -> None:
        lines = [
            "ArUco Robot Tracker",
            f"res {aw}×{ah}   fps {self._current_fps:.1f}",
            f"markers {n_markers}   frame {self._frame_count}",
            f"ROI {'active' if roi_active else 'not set — press R'}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 20 + i*18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50 if i == 0 else 0.40,
                        (220, 220, 220) if i == 0 else
                        ((0, 217, 255) if i == 3 and roi_active else (140, 140, 140)),
                        1, cv2.LINE_AA)
        cv2.putText(frame, "Q quit   A detect   F rotate ROI   C clear   S save   P print",
                    (10, ah-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (90, 90, 90), 1, cv2.LINE_AA)

    def _apply_corners(self, corners: np.ndarray, rotation: int) -> None:
        """Apply 4 detected corners with a 0-3 rotation offset and activate ROI."""
        rotated = np.roll(corners, -rotation, axis=0)
        self.roi.clear()
        for pt in rotated:
            self.roi.add_point(int(pt[0]), int(pt[1]))

    def _draw_roi_prompt(self, frame, n_collected) -> None:
        labels = ["1-Top-left", "2-Top-right", "3-Bottom-right", "4-Bottom-left"]
        msg    = f"Click corner {n_collected+1}: {labels[n_collected]}"
        cv2.putText(frame, msg, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 217, 255), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Pose estimation
    # ------------------------------------------------------------------

    def _estimate_pose(self, corners):
        try:
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self._K, self._D)
            return rvec[0], tvec[0]
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # Session management  (called from the HTTP API thread)
    # ------------------------------------------------------------------

    def start_session(self, algorithm: str = "", robot_ids: list = None) -> dict:
        with self._state_lock:
            if self._session:
                self._close_video()   # close any previous recording

            sid = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.path_tracker.clear()
            self.bump_detector.clear()

            self._session = {
                "id":        sid,
                "algorithm": algorithm,
                "robot_ids": robot_ids or [],
                "started_at": datetime.utcnow().isoformat() + "Z",
                "t0":         time.time(),
            }
            print(f"[session] started {sid} ({algorithm})")
            return {"id": sid, "algorithm": algorithm}

    def stop_session(self) -> dict | None:
        with self._state_lock:
            if not self._session:
                return None
            sess = self._session.copy()
            sess["duration_s"] = round(time.time() - sess["t0"], 1)
            self._close_video()
            self._session = None

        # Save results outside the lock
        result = self._save_session(sess)
        print(f"[session] stopped {sess['id']} — saved to {result['dir']}")
        return result

    def _close_video(self):
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
            self._video_writer = None

    def _open_video(self, sess_dir: Path, frame_size: tuple):
        """Open a VideoWriter for this session. frame_size = (w, h)."""
        video_path = sess_dir / "video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(video_path), fourcc, 10.0, frame_size)
        if vw.isOpened():
            self._video_writer = vw
            self._session["video_path"] = str(video_path)
        else:
            print("[session] warning: could not open video writer")

    def _save_session(self, sess: dict) -> dict:
        sess_dir = SESSIONS_DIR / sess["id"]
        sess_dir.mkdir(parents=True, exist_ok=True)

        # Paths in metres
        paths_m = {}
        for mid, pts in self.path_tracker._centres.items():
            if self.roi.active:
                paths_m[str(mid)] = [list(self.roi.to_metres(p)) for p in pts]
            else:
                paths_m[str(mid)] = [[int(x), int(y)] for x, y in pts]

        data = {
            "session":    sess,
            "paths_m":    paths_m,
            "paths_px":   {str(mid): [[int(x), int(y)] for x, y in pts]
                           for mid, pts in self.path_tracker._centres.items()},
            "bumps":      self.bump_detector.export(),
            "total_bumps":self.bump_detector.total_bumps(),
            "room":       {"w": self.roi.room_w, "h": self.roi.room_h},
        }
        (sess_dir / "state.json").write_text(json.dumps(data, indent=2))
        return {"dir": str(sess_dir), "id": sess["id"], **data}

    def get_state(self) -> dict:
        """Thread-safe snapshot of current tracker state (for the HTTP API)."""
        with self._state_lock:
            sess_snap = dict(self._session) if self._session else None

        markers = {}
        for mid in set(self.path_tracker._centres.keys()):
            last = self.path_tracker.last_centre(mid)
            pos_m = list(self.roi.to_metres(last)) if (self.roi.active and last) else None
            path_m = []
            if self.roi.active:
                path_m = [list(self.roi.to_metres(p))
                          for p in self.path_tracker.smoothed_path(mid)]
            markers[str(mid)] = {
                "id":         int(mid),
                "last_pos_px": list(last) if last else None,
                "last_pos_m":  pos_m,
                "speed_px_s":  round(self.path_tracker.speed(mid), 2),
                "heading_deg": round(self.path_tracker.heading_deg(mid), 1),
                "odometry_m":  round(
                    self.path_tracker.odometry_px(mid) / max(self.roi.PX_PER_METER, 1), 2
                ) if self.roi.active else None,
                "bump_count":  self.bump_detector.bump_count(mid),
                "path_m":      path_m,
            }

        session_info = None
        if sess_snap:
            session_info = {
                "id":         sess_snap["id"],
                "algorithm":  sess_snap.get("algorithm", ""),
                "robot_ids":  sess_snap.get("robot_ids", []),
                "started_at": sess_snap.get("started_at"),
                "duration_s": round(time.time() - sess_snap["t0"], 1),
            }

        return {
            "active":      True,
            "roi_active":  self.roi.active,
            "room_w":      self.roi.room_w,
            "room_h":      self.roi.room_h,
            "fps":         round(self._current_fps, 1),
            "frame_count": self._frame_count,
            "markers":     markers,
            "session":     session_info,
            "total_bumps": self.bump_detector.total_bumps(),
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        aw, ah = self._open_camera()

        # try loading saved corners
        self.roi.load(self.corners_file)

        # preview window (raw camera, scaled)
        dw = int(aw * self.display_scale)
        dh = int(ah * self.display_scale)
        self._click_scale = self.display_scale

        raw_win  = "Raw Camera — R to set ROI"
        room_win = "Room View (warped)"

        cv2.namedWindow(raw_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(raw_win, dw, dh)
        # show a placeholder so the window is fully initialised before attaching callback
        placeholder = np.zeros((dh, dw, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Starting camera...", (dw//2 - 120, dh//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        cv2.imshow(raw_win, placeholder)
        cv2.waitKey(1)
        self._mouse_enabled = False
        try:
            cv2.setMouseCallback(raw_win, self._mouse_cb)
            self._mouse_enabled = True
        except cv2.error as e:
            print(f"[warn] mouse callback unavailable ({e})")
            print("[warn] ROI clicking disabled — use --corners to load a saved file,")
            print("[warn]   or run:  python3 aruco_tracker.py --save-frame  to capture a frame,")
            print("[warn]   then:   python3 pick_corners.py frame.png  to pick corners interactively.")

        print("[tracker] running — A to auto-detect red corners, Q to quit")

        # auto-detect red corners on the first few frames
        _auto_attempts = 0
        _auto_max      = 30   # try for ~1 second before giving up silently

        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    continue

                self._frame_count += 1
                self._fps_frames  += 1
                now = time.time()
                if now - self._fps_time >= 1.0:
                    self._current_fps = self._fps_frames / (now - self._fps_time)
                    self._fps_frames  = 0
                    self._fps_time    = now

                # ── auto-detect red floor corners on startup ──────────
                if not self.roi.active and _auto_attempts < _auto_max:
                    _auto_attempts += 1
                    corners_found = detect_red_corners(frame)
                    if corners_found is not None:
                        self._last_raw_corners = corners_found
                        self._apply_corners(corners_found, self._corner_rotation)
                        self.roi.save(self.corners_file)
                        print(f"[roi] detected red floor corners — press F to rotate if orientation is wrong")
                        _auto_attempts = _auto_max

                # ── ArUco detection always on full 4K frame ───────────
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self.detector.detectMarkers(gray)

                # ── build room view ───────────────────────────────────
                if self.roi.active:
                    work = self.roi.warp(frame)
                else:
                    work = frame

                n_markers = 0
                _frame_positions_m: dict = {}   # {mid: (x_m, y_m)} — for bump detection

                if ids is not None:
                    n_markers = len(ids)

                    for i, mid in enumerate(ids.flatten()):
                        c_orig = corners[i][0]   # corners in 4K coords

                        if self.roi.active:
                            # transform each corner point through the homography
                            pts = c_orig.reshape(-1, 1, 2).astype(np.float32)
                            pts_w = cv2.perspectiveTransform(pts, self.roi.M).reshape(-1, 2)
                            centre = (int(pts_w[:, 0].mean()), int(pts_w[:, 1].mean()))

                            # ── skip markers outside the arena bounds ──────
                            margin = 20   # px — allow a small border tolerance
                            if not (-margin <= centre[0] <= self.roi.out_w + margin and
                                    -margin <= centre[1] <= self.roi.out_h + margin):
                                continue  # marker is outside the room — ignore it

                            # draw marker outline in warped coords
                            cv2.polylines(work, [pts_w.astype(np.int32)], True, _color(mid), 2)
                            cv2.putText(work, str(mid),
                                        (centre[0] - 10, centre[1] - 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _color(mid), 2)
                        else:
                            centre = (int(c_orig[:, 0].mean()), int(c_orig[:, 1].mean()))
                            cv2.aruco.drawDetectedMarkers(work, corners[i:i+1], ids[i:i+1])

                        self.path_tracker.update(mid, centre, now)

                        # Collect position for frame-level bump detection
                        pos_m = self.roi.to_metres(centre) if self.roi.active else None
                        if pos_m is not None:
                            _frame_positions_m[mid] = pos_m

                        self._draw_path(work, mid)

                        rvec, tvec = self._estimate_pose(corners[i:i+1])
                        if rvec is not None and not self.roi.active:
                            cv2.drawFrameAxes(work, self._K, self._D,
                                              rvec, tvec, self.marker_length * 0.5)

                        self._draw_info_box(work, mid, *centre, tvec, pos_m)

                # ── bump detection (once per frame, position-based) ────
                if self.roi.active and _frame_positions_m:
                    self.bump_detector.update_all(
                        _frame_positions_m,
                        self.roi.room_w, self.roi.room_h, now,
                    )

                # ── room grid (warped view only) ──────────────────────
                if self.roi.active:
                    self._draw_grid(work)
                    cv2.imshow(room_win, work)

                    # ── video recording (active session only) ─────────
                    with self._state_lock:
                        recording = self._session is not None
                        if recording and self._video_writer is None:
                            sess_dir = SESSIONS_DIR / self._session["id"]
                            sess_dir.mkdir(parents=True, exist_ok=True)
                            h_w, w_w = work.shape[:2]
                            self._open_video(sess_dir, (w_w, h_w))
                    if recording and self._video_writer is not None:
                        self._video_writer.write(work)

                # ── raw preview ───────────────────────────────────────
                preview = cv2.resize(frame, (dw, dh))
                self.roi.draw_overlay(preview, scale=self.display_scale)

                if self.roi.collecting:
                    n = len(self.roi._pending)
                    self._draw_roi_prompt(preview, n)

                if not self.roi.active and ids is not None:
                    small_corners = [
                        (c * self.display_scale).astype(np.float32)
                        for c in corners
                    ]
                    cv2.aruco.drawDetectedMarkers(preview, small_corners, ids)

                self._draw_hud(preview, n_markers, aw, ah, self.roi.active)
                cv2.imshow(raw_win, preview)

                # ── keyboard ──────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("a"):
                    corners_found = detect_red_corners(frame)
                    if corners_found is not None:
                        self._last_raw_corners = corners_found
                        self._corner_rotation  = 0
                        self._apply_corners(corners_found, 0)
                        self.roi.save(self.corners_file)
                        print("[roi] re-detected red corners — press F to rotate if wrong")
                    else:
                        print("[roi] could not find 4 red corners — check lighting / marker colour")
                elif key == ord("f"):
                    if self._last_raw_corners is not None:
                        self._corner_rotation = (self._corner_rotation + 1) % 4
                        self._apply_corners(self._last_raw_corners, self._corner_rotation)
                        self.roi.save(self.corners_file)
                        print(f"[roi] rotation {self._corner_rotation}/3 — press F again if still wrong")
                    else:
                        print("[roi] no corners detected yet — press A first")
                elif key == ord("r"):
                    if self._mouse_enabled:
                        self.roi.start_collection()
                        print("[roi] click the 4 corners: TL → TR → BR → BL")
                    else:
                        print("[roi] mouse not available — save a frame with S, then run pick_corners.py")
                elif key == ord("c"):
                    self.path_tracker.clear()
                    self.bump_detector.clear()
                    print("[tracker] paths and bump counts cleared")
                elif key == ord("s"):
                    fname = f"screenshot_{int(now)}.png"
                    cv2.imwrite(fname, work)
                    print(f"[tracker] saved {fname}")
                elif key == ord("p"):
                    data = self.path_tracker.export_paths()
                    print(json.dumps(
                        {str(k): v[-5:] for k, v in data.items()}, indent=2))

        finally:
            self._close_video()
            self._cap.release()
            cv2.destroyAllWindows()
            print("[tracker] stopped")

    def export_paths(self) -> dict:
        return {mid: np.array(list(pts))
                for mid, pts in self.path_tracker._centres.items()}

    def save_paths_json(self, path: str = "paths.json") -> None:
        data = {str(mid): [[int(x), int(y)] for x, y in pts]
                for mid, pts in self.path_tracker._centres.items()}
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"[tracker] paths written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DICT_MAP = {
    "4X4_50":         cv2.aruco.DICT_4X4_50,
    "4X4_100":        cv2.aruco.DICT_4X4_100,
    "5X5_50":         cv2.aruco.DICT_5X5_50,
    "5X5_100":        cv2.aruco.DICT_5X5_100,
    "6X6_250":        cv2.aruco.DICT_6X6_250,
    "6X6_1000":       cv2.aruco.DICT_6X6_1000,
    "7X7_250":        cv2.aruco.DICT_7X7_250,
    "ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="4K ArUco Robot Path Tracker with perspective ROI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--camera",        type=int,   default=4)
    ap.add_argument("--width",         type=int,   default=3840)
    ap.add_argument("--height",        type=int,   default=2160)
    ap.add_argument("--fps",           type=int,   default=30)
    ap.add_argument("--dict",          type=str,   default="4X4_50", choices=list(_DICT_MAP))
    ap.add_argument("--marker-length", type=float, default=0.10)
    ap.add_argument("--history",       type=int,   default=150)
    ap.add_argument("--scale",         type=float, default=0.5)
    ap.add_argument("--calib",         type=str,   default=None)
    ap.add_argument("--room-w",        type=float, default=3.0,  help="Room width in metres")
    ap.add_argument("--room-h",        type=float, default=4.0,  help="Room height in metres")
    ap.add_argument("--corners",       type=str,   default="corners.json",
                    help="JSON file to load/save ROI corner points")
    args = ap.parse_args()

    K = D = None
    if args.calib:
        data = np.load(args.calib)
        K, D = data["camera_matrix"], data["dist_coeffs"]
        print(f"[calib] loaded from {args.calib}")

    ArucoRobotTracker(
        camera_index   = args.camera,
        resolution     = (args.width, args.height),
        fps            = args.fps,
        aruco_dict_type= _DICT_MAP[args.dict],
        camera_matrix  = K,
        dist_coeffs    = D,
        marker_length  = args.marker_length,
        path_history   = args.history,
        display_scale  = args.scale,
        room_w         = args.room_w,
        room_h         = args.room_h,
        corners_file   = args.corners,
    ).run()


if __name__ == "__main__":
    main()
