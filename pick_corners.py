"""
pick_corners.py
===============
Click the 4 corners of the white wall area on a saved image.
Uses tkinter (no Qt/OpenCV GUI needed).

Usage:
    # 1. Press S in the tracker to save a screenshot, then:
    python3 pick_corners.py screenshot_<timestamp>.png

    # Custom room size or output file:
    python3 pick_corners.py frame.png --room-w 3 --room-h 4 --out corners.json

Click order:  Top-left → Top-right → Bottom-right → Bottom-left
Right-click to undo last point. Close the window when done.
"""

import tkinter as tk
from tkinter import messagebox
import json
import argparse
from pathlib import Path

try:
    from PIL import Image, ImageTk
except ImportError:
    import subprocess, sys
    print("Installing Pillow ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
    from PIL import Image, ImageTk


LABELS = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
COLORS = ["#00D9FF", "#FF9800", "#39FF14", "#FF00C8"]
MAX_DIM = 1200


def pick_corners(image_path: str, out_path: str, room_w: float, room_h: float) -> None:
    img = Image.open(image_path)
    orig_w, orig_h = img.size
    scale = min(MAX_DIM / orig_w, MAX_DIM / orig_h, 1.0)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)
    img_disp = img.resize((disp_w, disp_h), Image.LANCZOS)

    points = []   # in original image coords

    root = tk.Tk()
    root.title("Pick Corners — click TL → TR → BR → BL")
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=disp_w, height=disp_h, cursor="crosshair")
    canvas.pack()

    status = tk.Label(root, text=f"Click corner 1/4: {LABELS[0]}",
                      font=("Helvetica", 13), fg=COLORS[0], bg="#111")
    status.pack(fill="x")

    tk_img = ImageTk.PhotoImage(img_disp)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

    def redraw():
        canvas.delete("overlay")
        for i, pt in enumerate(points):
            sx, sy = int(pt[0] * scale), int(pt[1] * scale)
            r = 8
            canvas.create_oval(sx-r, sy-r, sx+r, sy+r,
                                fill=COLORS[i], outline="white", width=2, tags="overlay")
            canvas.create_text(sx+14, sy-8, text=LABELS[i],
                                fill=COLORS[i], font=("Helvetica", 11, "bold"), tags="overlay")
        if len(points) == 4:
            coords = []
            for pt in points:
                coords += [int(pt[0]*scale), int(pt[1]*scale)]
            coords += [int(points[0][0]*scale), int(points[0][1]*scale)]
            canvas.create_line(coords, fill="#00D9FF", width=2, tags="overlay")

    def on_click(event):
        if len(points) >= 4:
            return
        orig_x = int(event.x / scale)
        orig_y = int(event.y / scale)
        points.append((orig_x, orig_y))
        idx = len(points)
        print(f"  corner {idx}/4 — {LABELS[idx-1]}: ({orig_x}, {orig_y})")
        redraw()
        if idx < 4:
            status.config(text=f"Click corner {idx+1}/4: {LABELS[idx]}",
                          fg=COLORS[idx])
        else:
            status.config(text="All 4 corners set — close window to save, right-click to undo",
                          fg="#39FF14")

    def on_right_click(event):
        if points:
            points.pop()
            redraw()
            idx = len(points)
            status.config(text=f"Click corner {idx+1}/4: {LABELS[idx]}",
                          fg=COLORS[idx])

    def on_close():
        if len(points) != 4:
            if not messagebox.askyesno("Quit", "Only {}/4 corners set. Quit without saving?".format(len(points))):
                return
            root.destroy()
            return
        data = {
            "corners": [[int(x), int(y)] for x, y in points],
            "room_w": room_w,
            "room_h": room_h,
        }
        Path(out_path).write_text(json.dumps(data, indent=2))
        print(f"\nSaved to {out_path}")
        print("Now run:  python3 aruco_tracker.py --camera 4 --dict 4X4_50")
        root.destroy()

    canvas.bind("<Button-1>", on_click)
    canvas.bind("<Button-3>", on_right_click)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick 4 ROI corners on a saved image (tkinter)")
    ap.add_argument("image",    type=str,   help="Path to screenshot PNG")
    ap.add_argument("--out",    type=str,   default="corners.json")
    ap.add_argument("--room-w", type=float, default=3.0)
    ap.add_argument("--room-h", type=float, default=4.0)
    args = ap.parse_args()
    pick_corners(args.image, args.out, args.room_w, args.room_h)


if __name__ == "__main__":
    main()
