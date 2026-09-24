"""
camera_calibrate.py
===================
Checkerboard-based camera calibration for the 4K tracker.

Prints a 9×6 checkerboard, hold it in front of the camera from ~20
different angles, and this script finds the intrinsics + distortion
coefficients that make ArucoRobotTracker's 3D pose estimates accurate.

Usage:
    python camera_calibrate.py                    # device 0, 4K
    python camera_calibrate.py --camera 1
    python camera_calibrate.py --square 0.025     # 25 mm squares
    python camera_calibrate.py --out my_cam.npz

Keyboard controls:
    SPACE  — capture the current frame as a calibration sample
    C      — run calibration with collected samples and save
    Q      — quit without saving
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path

CHECKERBOARD = (9, 6)   # inner corner count (columns, rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Checkerboard camera calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--camera",    type=int,   default=0)
    ap.add_argument("--width",     type=int,   default=3840)
    ap.add_argument("--height",    type=int,   default=2160)
    ap.add_argument("--square",    type=float, default=0.030,
                    help="Physical side length of one checkerboard square (metres)")
    ap.add_argument("--min-frames",type=int,   default=20,
                    help="Minimum number of good frames before calibration")
    ap.add_argument("--out",       type=str,   default="calib.npz",
                    help="Output file path")
    ap.add_argument("--scale",     type=float, default=0.4,
                    help="Preview scale")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          30)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[calib] camera {aw}×{ah}")

    # 3-D points in the checkerboard's own coordinate system
    obj_pts_template = np.zeros(
        (CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32
    )
    obj_pts_template[:, :2] = (
        np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
        * args.square
    )

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    dw, dh   = int(aw * args.scale), int(ah * args.scale)

    win = "Calibration — SPACE capture  C calibrate  Q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, dw, dh)

    print(f"[calib] hold the checkerboard and press SPACE to capture frames.")
    print(f"[calib] need at least {args.min_frames} good frames, then press C.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)

        n = len(obj_points)
        status_color = (57, 255, 20) if found else (0, 120, 255)
        cv2.putText(display, f"Frames captured: {n}/{args.min_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(display, "SPACE=capture  C=calibrate  Q=quit",
                    (10, ah - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (160, 160, 160), 1)

        cv2.imshow(win, cv2.resize(display, (dw, dh)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        elif key == ord(" ") and found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(obj_pts_template)
            img_points.append(corners2)
            print(f"[calib] captured frame {len(obj_points)}")
            # brief flash to confirm
            cv2.imshow(win, cv2.resize(frame * 0, (dw, dh)))
            cv2.waitKey(100)

        elif key == ord("c"):
            if len(obj_points) < args.min_frames:
                print(f"[calib] need {args.min_frames} frames, have {len(obj_points)}")
                continue

            print("[calib] running calibration …")
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, (aw, ah), None, None
            )
            print(f"[calib] RMS reprojection error: {rms:.4f} px")
            print(f"[calib] camera matrix:\n{camera_matrix}")
            print(f"[calib] distortion:  {dist_coeffs.ravel()}")

            out_path = Path(args.out)
            np.savez(
                out_path,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                rms=np.array([rms]),
                image_size=np.array([aw, ah]),
            )
            print(f"[calib] saved to {out_path}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
