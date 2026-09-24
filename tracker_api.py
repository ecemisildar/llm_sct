"""
tracker_api.py
==============
Run this instead of aruco_tracker.py when you want dashboard integration.
Starts the ArUco tracker (OpenCV, main thread) and a FastAPI HTTP server
(port 8001, background thread) that the robot-hub dashboard talks to.

Usage:
    QT_QPA_PLATFORM=xcb python3 tracker_api.py --camera 4 --dict 4X4_50 --room-w 4 --room-h 3

API endpoints (all on http://localhost:8001):
    GET  /state                  current tracking state (live)
    POST /session/start          start a named session
    POST /session/stop           stop session, save paths + video
    GET  /sessions               list saved sessions
    GET  /sessions/{id}          saved session data
    GET  /sessions/{id}/video    download session video (mp4)
"""

import argparse
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from aruco_tracker import ArucoRobotTracker, SESSIONS_DIR, _DICT_MAP

# ---------------------------------------------------------------------------
# Shared tracker instance (set in main before uvicorn starts)
# ---------------------------------------------------------------------------

_tracker: ArucoRobotTracker | None = None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ArUco Tracker API", version="1.0")


@app.get("/state")
def get_state():
    if _tracker is None:
        return {"active": False, "markers": {}, "session": None, "total_bumps": 0}
    return _tracker.get_state()


class SessionStartBody(BaseModel):
    algorithm: str = ""
    robot_ids: list[str] = []


@app.post("/session/start")
def session_start(body: SessionStartBody):
    if _tracker is None:
        raise HTTPException(status_code=503, detail="tracker_not_running")
    result = _tracker.start_session(body.algorithm, body.robot_ids)
    return {"ok": True, **result}


@app.post("/session/stop")
def session_stop():
    if _tracker is None:
        raise HTTPException(status_code=503, detail="tracker_not_running")
    result = _tracker.stop_session()
    if result is None:
        return {"ok": True, "message": "no_active_session"}
    # Remove numpy / non-serialisable bits from result
    clean = {k: v for k, v in result.items()
             if k not in ("paths_px",) and isinstance(v, (str, int, float, list, dict, bool, type(None)))}
    return {"ok": True, "session": clean}


@app.get("/sessions")
def list_sessions():
    if not SESSIONS_DIR.exists():
        return []
    sessions = []
    for d in sorted(SESSIONS_DIR.iterdir(), reverse=True):
        state_file = d / "state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                sessions.append({
                    "id":          d.name,
                    "algorithm":   data.get("session", {}).get("algorithm", ""),
                    "started_at":  data.get("session", {}).get("started_at"),
                    "duration_s":  data.get("session", {}).get("duration_s"),
                    "total_bumps": data.get("total_bumps", 0),
                    "has_video":   (d / "video.mp4").exists(),
                })
            except Exception:
                pass
    return sessions


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    state_file = SESSIONS_DIR / session_id / "state.json"
    if not state_file.exists():
        raise HTTPException(status_code=404, detail="session_not_found")
    return JSONResponse(content=json.loads(state_file.read_text()))


@app.get("/sessions/{session_id}/video")
def get_session_video(session_id: str):
    video_path = SESSIONS_DIR / session_id / "video.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="video_not_found")
    return FileResponse(str(video_path), media_type="video/mp4",
                        filename=f"session_{session_id}.mp4")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _tracker

    ap = argparse.ArgumentParser(
        description="ArUco tracker with dashboard HTTP API (port 8001)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--camera",        type=int,   default=4)
    ap.add_argument("--width",         type=int,   default=3840)
    ap.add_argument("--height",        type=int,   default=2160)
    ap.add_argument("--fps",           type=int,   default=30)
    ap.add_argument("--dict",          type=str,   default="4X4_50", choices=list(_DICT_MAP))
    ap.add_argument("--marker-length", type=float, default=0.10)
    ap.add_argument("--history",       type=int,   default=300)
    ap.add_argument("--scale",         type=float, default=0.5)
    ap.add_argument("--calib",         type=str,   default=None)
    ap.add_argument("--room-w",        type=float, default=4.0)
    ap.add_argument("--room-h",        type=float, default=3.0)
    ap.add_argument("--corners",       type=str,   default="corners.json")
    ap.add_argument("--api-port",      type=int,   default=8001)
    args = ap.parse_args()

    K = D = None
    if args.calib:
        data = np.load(args.calib)
        K, D = data["camera_matrix"], data["dist_coeffs"]

    _tracker = ArucoRobotTracker(
        camera_index    = args.camera,
        resolution      = (args.width, args.height),
        fps             = args.fps,
        aruco_dict_type = _DICT_MAP[args.dict],
        camera_matrix   = K,
        dist_coeffs     = D,
        marker_length   = args.marker_length,
        path_history    = args.history,
        display_scale   = args.scale,
        room_w          = args.room_w,
        room_h          = args.room_h,
        corners_file    = args.corners,
    )

    # Start API server in a daemon thread so it dies when the tracker exits
    api_thread = threading.Thread(
        target=lambda: uvicorn.run(
            app, host="0.0.0.0", port=args.api_port, log_level="warning"
        ),
        daemon=True,
    )
    api_thread.start()
    print(f"[api] tracker API on http://0.0.0.0:{args.api_port}")

    # Tracker runs on the main thread (OpenCV GUI requirement)
    _tracker.run()


if __name__ == "__main__":
    main()
