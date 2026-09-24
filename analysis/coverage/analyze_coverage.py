#!/usr/bin/env python3
"""ArUco trajectories with the simulation's exact circular grid-cell calculation."""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

os.environ.setdefault('OPENCV_FFMPEG_THREADS', '1')
import cv2
import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[2]
WORKSPACE = EXPERIMENT.parents[1]
SIMULATION = WORKSPACE / 'src/llm_sct/evaluation/evaluation/coverage_counter.py'
OUTPUT = Path(__file__).resolve().parent
ROBOT_IDS = (0, 1, 2)
VERSION = 3


def simulation_geometry():
    """Load only pure geometry from source, without importing ROS dependencies."""
    source = SIMULATION.read_text()
    tree = ast.parse(source)
    circle = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'circle_intersects_cell')
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CoverageCounter')
    footprint = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_footprint_cells')
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == 'DEFAULT_ROBOT_FOOTPRINT_RADIUS'}
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    values = {}
    for n in ast.walk(init):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute):
            key = n.targets[0].attr
            if key in ('grid_size', 'pose_interval_sec'):
                values[key] = ast.literal_eval(n.value)
    namespace = {'math': math}
    exec(compile(ast.Module(body=[circle, footprint], type_ignores=[]), str(SIMULATION), 'exec'), namespace)
    return namespace['_footprint_cells'], constants['DEFAULT_ROBOT_FOOTPRINT_RADIUS'], values, hashlib.sha256(source.encode()).hexdigest()


FOOTPRINT, RADIUS, SETTINGS, SIM_SHA = simulation_geometry()
GRID = SETTINGS['grid_size']
INTERVAL = SETTINGS['pose_interval_sec']
CALIBRATION = json.loads((EXPERIMENT / 'analysis/cropping/arena_calibration.json').read_text())
SOURCE_TO_CROP = np.array(CALIBRATION['source_to_cropped_homography'], dtype=np.float64)
CROP_TO_SOURCE = np.linalg.inv(SOURCE_TO_CROP)
WIDTH = CALIBRATION['arena_width_m']
HEIGHT = CALIBRATION['arena_height_m']
NX, NY = int(WIDTH / GRID), int(HEIGHT / GRID)
COUNTER = SimpleNamespace(env_min_x=0., env_min_y=0., grid_size=GRID,
                          num_cells_x=NX, num_cells_y=NY, robot_footprint_radius=RADIUS)


def write_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


class Tracker:
    def __init__(self):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
        self.last = {}
        self.last_search = -10.
        self.rejected = 0

    def detect(self, image, origin=(0, 0), scales=(1., 1.5, 2.), wanted=ROBOT_IDS):
        found = {}
        if image.size == 0:
            return found
        for scale in scales:
            work = image if scale == 1 else cv2.resize(image, None, fx=scale, fy=scale,
                                                       interpolation=cv2.INTER_CUBIC)
            corners, ids, _ = self.detector.detectMarkers(work)
            if ids is not None:
                for quad, marker in zip(corners, ids.flatten()):
                    marker = int(marker)
                    if marker in wanted and marker not in found:
                        found[marker] = quad[0] / scale + np.array(origin)
            if all(marker in found for marker in wanted):
                break
        return found

    def update(self, gray, t, recover=None):
        height, width = gray.shape
        found = {}
        for marker, (previous_t, quad) in self.last.items():
            if t - previous_t > 1.:
                continue
            x, y = quad.mean(axis=0)
            margin = 75
            x0, y0 = max(0, int(x) - margin), max(0, int(y) - margin)
            x1, y1 = min(width, int(x) + margin), min(height, int(y) + margin)
            if x1 <= x0 or y1 <= y0:
                continue
            found.update(self.detect(gray[y0:y1, x0:x1], (x0, y0), wanted=(marker,)))
        missing = set(ROBOT_IDS) - set(found)
        overdue = any(marker not in self.last or t - self.last[marker][0] >= 1. for marker in missing)
        if t - self.last_search >= 2. or (overdue and t - self.last_search >= .8):
            found.update(self.detect(gray, scales=(1., 1.5), wanted=tuple(missing) or ROBOT_IDS))
            self.last_search = t
        if recover is not None and set(ROBOT_IDS) - set(found):
            found.update(recover(set(ROBOT_IDS) - set(found), t, self))
        accepted = {}
        for marker, quad in found.items():
            center = quad.mean(axis=0)
            if marker in self.last:
                previous_t, previous = self.last[marker]
                # Generous motion gate prevents decoded IDs from teleporting to another object.
                distance_m = np.linalg.norm((center - previous.mean(axis=0)) * np.array([WIDTH / width, HEIGHT / height]))
                if distance_m > 2. * (t - previous_t) + .15:
                    self.rejected += 1
                    continue
            accepted[marker] = quad
            self.last[marker] = (t, quad)
        return accepted


def analyze(video):
    cv2.setNumThreads(1)
    video = Path(video)
    task = video.parent.parent.name
    directory = OUTPUT / task / video.stem
    directory.mkdir(parents=True, exist_ok=True)
    stat = video.stat()
    summary_path = directory / 'summary.json'
    if summary_path.exists():
        cached = json.loads(summary_path.read_text())
        if (cached.get('analysis_version') == VERSION and cached.get('simulation_sha256') == SIM_SHA
                and cached.get('video_size') == stat.st_size and cached.get('video_mtime_ns') == stat.st_mtime_ns):
            print('Retained:', task, video.name, flush=True)
            return cached
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open {video}')
    tracker = Tracker()
    original = video.parent.parent / video.name
    original_stat = original.stat()
    original_capture = cv2.VideoCapture(str(original))
    original_time = -1.
    fallback_counts = {marker: 0 for marker in ROBOT_IDS}

    def recover(missing, t, current_tracker):
        nonlocal original_time
        # Advance sequentially: seeking for every lost marker repeatedly decodes a whole GOP.
        while original_time + 1. / 60. < t:
            if not original_capture.grab():
                return {}
            original_time = original_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.
        ok, raw = original_capture.retrieve()
        if not ok or abs(original_time - t) > .1:
            return {}
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        recovered = {}
        for marker in sorted(missing):
            previous = current_tracker.last.get(marker)
            if previous is not None and t - previous[0] <= 1.:
                center = cv2.perspectiveTransform(previous[1][None, :, :].astype(np.float32), CROP_TO_SOURCE)[0].mean(axis=0)
                x, y = center
                x0, y0 = max(0, int(x)-120), max(0, int(y)-120)
                x1, y1 = min(gray.shape[1], int(x)+120), min(gray.shape[0], int(y)+120)
                if x1 > x0 and y1 > y0:
                    recovered.update(current_tracker.detect(gray[y0:y1, x0:x1], (x0,y0), wanted=(marker,)))
        still_missing = set(missing) - set(recovered)
        if still_missing:
            recovered.update(current_tracker.detect(gray, scales=(1.,1.5), wanted=tuple(still_missing)))
        mapped = {}
        for marker, quad in recovered.items():
            mapped[marker] = cv2.perspectiveTransform(quad[None, :, :].astype(np.float32), SOURCE_TO_CROP)[0]
            fallback_counts[marker] += 1
        return mapped
    paths, observations, visits, timeseries = [], [], [], []
    visited, by_robot = set(), {marker: set() for marker in ROBOT_IDS}
    counts = {marker: 0 for marker in ROBOT_IDS}
    longest_gaps = {marker: 0. for marker in ROBOT_IDS}
    last_seen = {marker: 0. for marker in ROBOT_IDS}
    first_seen = {marker: None for marker in ROBOT_IDS}
    next_sample, next_metric = 0., 0.
    sample_count = frame_count = 0
    last_time, last_frame = 0., None
    preview = None
    while capture.grab():
        frame_count += 1
        t = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.
        if frame_count > 1 and t + 1e-8 < last_time:
            raise RuntimeError(f'Nonmonotonic timestamps: {video}')
        last_time = t
        if t + 1e-8 < next_sample:
            continue
        ok, frame = capture.retrieve()
        if not ok:
            raise RuntimeError(f'Cannot decode frame {frame_count} of {video}')
        if frame.shape[:2] != (600, 800):
            raise RuntimeError(f'Unexpected crop dimensions: {video}')
        last_frame = frame
        while next_sample <= t + 1e-8:
            next_sample += INTERVAL
        # Match the simulation's metric timer: emit previous observations before future poses.
        while next_metric + 1e-8 < t:
            timeseries.append((round(next_metric, 6), 100. * len(visited) / (NX * NY)))
            next_metric += .5
        found = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), t, recover)
        sample_count += 1
        observations.append((round(t, 6), *[int(marker in found) for marker in ROBOT_IDS]))
        for marker, quad in sorted(found.items()):
            x_px, y_px = quad.mean(axis=0)
            x, y = float(x_px) * WIDTH / 800, HEIGHT - float(y_px) * HEIGHT / 600
            if not (-RADIUS <= x <= WIDTH + RADIUS and -RADIUS <= y <= HEIGHT + RADIUS):
                continue
            counts[marker] += 1
            longest_gaps[marker] = max(longest_gaps[marker], t - last_seen[marker])
            last_seen[marker] = t
            if first_seen[marker] is None:
                first_seen[marker] = t
            seconds = int(t)
            nanos = int(round((t - seconds) * 1e9))
            paths.append((seconds, nanos, f'robot_{marker}', round(x, 6), round(y, 6), round(t, 6)))
            for ix, iy in FOOTPRINT(COUNTER, x, y):
                cell = ix * NY + iy
                by_robot[marker].add(cell)
                if cell not in visited:
                    visited.add(cell)
                    visits.append((seconds, nanos, f'robot_{marker}', cell, ix * GRID, iy * GRID,
                                   (ix + .5) * GRID, (iy + .5) * GRID))
        if next_metric <= t + 1e-8:
            timeseries.append((round(next_metric, 6), 100. * len(visited) / (NX * NY)))
            next_metric += .5
        if preview is None and len(found) == 3:
            preview = frame.copy()
            cv2.aruco.drawDetectedMarkers(preview, [found[m][None, :, :] for m in sorted(found)],
                                         np.array(sorted(found), dtype=np.int32)[:, None])
    capture.release()
    original_capture.release()
    if not sample_count or last_frame is None:
        raise RuntimeError(f'No samples in {video}')
    duration = last_time + 1. / 30.
    for marker in ROBOT_IDS:
        longest_gaps[marker] = max(longest_gaps[marker], duration - last_seen[marker])
    timeseries.append((round(duration, 6), 100. * len(visited) / (NX * NY)))
    assert all(a[1] <= b[1] for a, b in zip(timeseries, timeseries[1:]))
    assert all(0 <= cov <= 100 for _, cov in timeseries)
    assert len(visited) == len(visits)
    write_csv(directory / 'coverage_paths.csv', ['stamp_sec', 'stamp_nsec', 'robot', 'x', 'y', 'elapsed_s'], paths)
    write_csv(directory / 'coverage_visited_cells.csv',
              ['stamp_sec', 'stamp_nsec', 'robot', 'cell_index', 'cell_min_x', 'cell_min_y', 'cell_center_x', 'cell_center_y'], visits)
    write_csv(directory / 'coverage_timeseries.csv', ['time_s', 'coverage_pct'], timeseries)
    write_csv(directory / 'tracking_observations.csv', ['time_s', 'robot_0_detected', 'robot_1_detected', 'robot_2_detected'], observations)
    write_csv(directory / 'robot_summary.csv', ['robot', 'detected_samples', 'detection_pct', 'longest_missing_gap_s', 'coverage_pct'],
              [(f'robot_{m}', counts[m], 100. * counts[m] / sample_count, longest_gaps[m],
                100. * len(by_robot[m]) / (NX * NY)) for m in ROBOT_IDS])
    cv2.imwrite(str(directory / 'detection_preview.jpg'), preview if preview is not None else last_frame)
    cv2.imwrite(str(directory / 'arena_background.jpg'), last_frame)
    t100 = next((t for t, cov in timeseries if cov >= 100.), None)
    condition = 'llm' if '_llm' in video.stem or '_lm' in video.stem else 'baseline'
    summary = dict(task=task, condition=condition, video=str(video), analysis_version=VERSION,
                   simulation_source=str(SIMULATION), simulation_sha256=SIM_SHA,
                   video_size=stat.st_size, video_mtime_ns=stat.st_mtime_ns, frame_count=frame_count,
                   duration_s=duration, analyzed_samples=sample_count, pose_interval_s=INTERVAL,
                   grid_size_m=GRID, robot_footprint_radius_m=RADIUS, arena_area_m2=WIDTH * HEIGHT,
                   free_cells=NX * NY, blocked_cells=[], visited_cells=len(visited),
                   final_coverage_pct=100. * len(visited) / (NX * NY), time_to_100_pct_s=t100,
                   detection_pct={str(m): 100. * counts[m] / sample_count for m in ROBOT_IDS},
                   longest_missing_gap_s={str(m): longest_gaps[m] for m in ROBOT_IDS},
                   first_detection_s={str(m): first_seen[m] for m in ROBOT_IDS},
                   rejected_jump_detections=tracker.rejected,
                   original_video=str(original), original_size=original_stat.st_size,
                   original_mtime_ns=original_stat.st_mtime_ns,
                   original_fallback_detections={str(m): fallback_counts[m] for m in ROBOT_IDS},
                   coordinates='metres; origin bottom left; ArUco marker centre approximates robot centre',
                   missing_detections='No coverage is invented or interpolated during tracking gaps.',
                   obstacle_policy='All 12 arena floor cells are counted; no surveyed obstacle mask available.')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    after = video.stat()
    assert (stat.st_size, stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    original_after = original.stat()
    assert (original_stat.st_size, original_stat.st_mtime_ns) == (original_after.st_size, original_after.st_mtime_ns)
    print(f'Completed: {task}/{video.name}: {summary["final_coverage_pct"]:.2f}%; '
          f'detection={[round(summary["detection_pct"][str(m)], 1) for m in ROBOT_IDS]}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()
    videos = sorted((EXPERIMENT / 'videos').glob('*/cropped/*.mp4'))
    videos = videos[args.offset:]
    if args.limit:
        videos = videos[:args.limit]
    start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        summaries = list(pool.map(analyze, map(str, videos)))
    saved = [json.loads(p.read_text()) for p in sorted(OUTPUT.glob('*/*/summary.json'))]
    (OUTPUT / 'batch_manifest.json').write_text(json.dumps(saved, indent=2) + '\n')
    print(f'Analyzed {len(summaries)} videos in {time.monotonic() - start:.1f}s.', flush=True)


if __name__ == '__main__':
    main()
