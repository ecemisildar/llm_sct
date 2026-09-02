#!/usr/bin/env python3
import os
import threading
import time
import math
import csv
import shutil
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
from rclpy.node import Node

from tf2_msgs.msg import TFMessage
from std_msgs.msg import Bool, Int32, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory


DEFAULT_ROBOT_FOOTPRINT_RADIUS = 0.36


def circle_intersects_cell(
    center_x, center_y, radius, cell_min_x, cell_min_y, cell_max_x, cell_max_y
):
    """Return whether a closed circle touches an axis-aligned grid cell."""
    closest_x = min(max(center_x, cell_min_x), cell_max_x)
    closest_y = min(max(center_y, cell_min_y), cell_max_y)
    return (
        (center_x - closest_x) ** 2 + (center_y - closest_y) ** 2
        <= radius ** 2
    )


class CoverageCounter(Node):
    """
    - Tracks visited grid cells from Gazebo poses (TFMessage)
    - Saves results into a timestamped run folder
    - Logs paths + visited cells to CSV for offline analysis
    - Copies run metadata (active YAML + prompt text) into the same folder
    - Does not render or save plots (handled by offline scripts)
    """

    def __init__(self):
        super().__init__("coverage_counter")

        # fixed settings (no args)
        results_dir_default = Path.home() / "sct_ws" / "src" / "llm_sct" / "new_results"
        self.results_root = Path(
            str(self.declare_parameter("results_dir", str(results_dir_default)).value)
        )
        run_id_param = str(self.declare_parameter("run_id", "").value).strip()
        self.run_id = run_id_param or time.strftime("run_%Y%m%d_%H%M%S")
        self.metadata_yaml_path_raw = str(
            self.declare_parameter("metadata_yaml_path", "").value
        ).strip()
        self.metadata_yaml_path = Path(self.metadata_yaml_path_raw) if self.metadata_yaml_path_raw else None
        self.prompt_file_path_raw = str(
            self.declare_parameter("prompt_file_path", "").value
        ).strip()
        self.prompt_file_path = Path(self.prompt_file_path_raw) if self.prompt_file_path_raw else None
        self.prompt_text = str(self.declare_parameter("prompt_text", "").value)
        self.mission = str(
            self.declare_parameter("mission", "unknown").value
        ).strip().casefold()
        self.run_duration = float(self.declare_parameter("run_duration", 200.0).value)
        self.shutdown_on_task_complete = bool(
            self.declare_parameter("shutdown_on_task_complete", False).value
        )
        self.wait_for_all_task_completion = bool(
            self.declare_parameter(
                "wait_for_all_task_completion", False
            ).value
        )
        self.task_progress_on_complete = int(
            self.declare_parameter("task_progress_on_complete", 3).value
        )
        configured_color_order = str(
            self.declare_parameter(
                "target_color_order", "red,green,blue"
            ).value
        )
        self.target_color_order = tuple(
            color.strip().casefold()
            for color in configured_color_order.split(",")
            if color.strip()
        )
        self.total_task_targets = max(1, len(self.target_color_order))
        if self.mission != "complex_task":
            self.task_progress_on_complete = self.total_task_targets
        else:
            self.total_task_targets = self.task_progress_on_complete
        self.flush_interval_sec = float(self.declare_parameter("flush_interval_sec", 3.0).value)
        self.flush_max_rows = int(self.declare_parameter("flush_max_rows", 2000).value)
        launch_parameter_names = (
            "sim_world",
            "headless",
            "auto_start",
            "run_duration",
            "total_robots",
            "random_seed",
            "auto_start_supervisor",
            "results_dir",
            "metadata_yaml_path",
        )
        self.launch_arguments = {
            name: str(self.declare_parameter(f"launch_{name}", "").value)
            for name in launch_parameter_names
        }

        self.obstacle_occupancy_threshold = 0.4
        self.circle_obstacle_occupancy_threshold = 0.05
        self.robot_footprint_radius = max(
            0.0,
            float(
                self.declare_parameter(
                    "robot_footprint_radius", DEFAULT_ROBOT_FOOTPRINT_RADIUS
                ).value
            ),
        )
        sim_world = self.launch_arguments.get("sim_world", "").strip()
        self.world_sdf = (
            Path(sim_world)
            if sim_world
            else Path(get_package_share_directory("leo_exploration"))
            / "worlds"
            / "random_world.sdf"
        )

        yaml_group = self.metadata_yaml_path.stem if self.metadata_yaml_path else "unknown_yaml"
        robot_count = self.launch_arguments.get("total_robots", "").strip() or "unknown"
        robot_group = f"robots_{robot_count}"
        self.results_dir = self.results_root / yaml_group / robot_group / self.run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.coverage_csv_path = self.results_dir / "coverage_timeseries.csv"
        self.paths_csv_path = self.results_dir / "coverage_paths.csv"
        self.visited_cells_csv_path = self.results_dir / "coverage_visited_cells.csv"
        self.task_result_path = self.results_dir / "task_result.csv"
        self.task_progress_path = self.results_dir / "task_progress.csv"
        self.status_path = self.results_dir / "SAVE_STATUS.txt"
        self.error_path = self.results_dir / "SAVE_ERROR.txt"
        self.prompt_txt_path = self.results_dir / "prompt.txt"

        self._saved_ok = False
        self._saving_now = False
        self._wall_start = time.time()
        total_robots_raw = self.launch_arguments.get("total_robots", "").strip()
        try:
            self.total_robots = max(1, int(total_robots_raw))
        except ValueError:
            self.total_robots = 1
        self.completed_robots = set()
        self.robot_completion_times = {
            robot_index: None for robot_index in range(self.total_robots)
        }
        self.robot_task_progress = {
            robot_index: 0 for robot_index in range(self.total_robots)
        }
        self.robot_color_reached_times = {
            robot_index: {"red": None, "green": None, "blue": None}
            for robot_index in range(self.total_robots)
        }
        self.team_completed_colors = set()
        self.team_delivered_boxes = {color: 0 for color in ("red", "green", "blue")}
        self.delivery_color_robot = {
            color: None for color in ("red", "green", "blue")
        }
        self.delivery_color_times = {
            color: None for color in ("red", "green", "blue")
        }
        self.task_completion_duration = None
        self._nominal_timeout_reported = False
        # grid
        self.env_min = -5
        self.env_max = 5
        self.grid_size = 1.0
        self.num_cells_y = int((self.env_max - self.env_min) / self.grid_size)
        self.cells = [
            (
                self.env_min + ix * self.grid_size,
                self.env_min + iy * self.grid_size,
            )
            for ix in range(self.num_cells_y)
            for iy in range(self.num_cells_y)
        ]
        self.visited = set()
        self.blocked = self._compute_blocked_cells()

        # metrics

        self.start_time = self.get_clock().now()
        self.coverage_history = []
        self._ensure_paths_csv_header()
        self._ensure_cells_csv_header()
        self._path_rows = []
        self._cell_rows = []

        # --- subscriptions ---
        pose_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            TFMessage,
            "/world/random_world/dynamic_pose/info",
            self.pose_callback,
            pose_qos,
        )
        self.task_complete_subscriptions = [
            self.create_subscription(
                Bool,
                f"/robot_{robot_index}/task_complete",
                lambda msg, index=robot_index: self._task_complete_callback(index, msg),
                10,
            )
            for robot_index in range(self.total_robots)
        ]
        self.task_progress_subscriptions = [
            self.create_subscription(
                Int32,
                f"/robot_{robot_index}/task_progress",
                lambda msg, index=robot_index: self._task_progress_callback(index, msg),
                10,
            )
            for robot_index in range(self.total_robots)
        ]
        self.task_progress_event_subscriptions = [
            self.create_subscription(
                String,
                f"/robot_{robot_index}/task_progress_event",
                lambda msg, index=robot_index: self._task_progress_event_callback(index, msg),
                10,
            )
            for robot_index in range(self.total_robots)
        ]

        self.pose_interval_sec = 0.2
        self._last_pose_time = self.get_clock().now()

        # timers
        self.timer_plot = self.create_timer(0.5, self._on_metrics_timer)
        self.timer_timeout = self.create_timer(1.0, self._on_timeout_timer)
        self.timer_flush = self.create_timer(self.flush_interval_sec, self._flush_buffers)

        self._write_status(
            "Node started\n"
            f"run_id: {self.run_id}\n"
            f"results_dir: {self.results_dir}\n"
            f"metadata_yaml: {self.metadata_yaml_path}\n"
            f"prompt_file: {self.prompt_file_path}\n"
            "launch_arguments:\n"
            + "".join(
                f"  {name}: {value}\n"
                for name, value in self.launch_arguments.items()
            )
        )
        self._save_run_metadata()
        self._write_task_result()
        self._write_task_progress()

    def pose_callback(self, msg: TFMessage):
        now = self.get_clock().now()
        if (now - self._last_pose_time).nanoseconds < int(self.pose_interval_sec * 1e9):
            return
        self._last_pose_time = now

        for t in msg.transforms:
            name = t.child_frame_id
            if not name.startswith("robot_"):
                continue
            if "/" in name:
                continue

            x = t.transform.translation.x
            y = t.transform.translation.y
            self._append_path_row(name, x, y)

            for ix, iy in self._footprint_cells(x, y):
                idx = ix * self.num_cells_y + iy
                if idx not in self.visited and idx not in self.blocked:
                    cx = self.env_min + ix * self.grid_size
                    cy = self.env_min + iy * self.grid_size
                    self.visited.add(idx)
                    self._append_visited_cell_row(name, idx, cx, cy)

    def _footprint_cells(self, x, y):
        """Yield grid cells touched by the rover's circular XY footprint."""
        radius = self.robot_footprint_radius
        min_ix = math.floor((x - radius - self.env_min) / self.grid_size)
        max_ix = math.floor((x + radius - self.env_min) / self.grid_size)
        min_iy = math.floor((y - radius - self.env_min) / self.grid_size)
        max_iy = math.floor((y + radius - self.env_min) / self.grid_size)

        for ix in range(max(0, min_ix), min(self.num_cells_y - 1, max_ix) + 1):
            cell_min_x = self.env_min + ix * self.grid_size
            cell_max_x = cell_min_x + self.grid_size
            for iy in range(max(0, min_iy), min(self.num_cells_y - 1, max_iy) + 1):
                cell_min_y = self.env_min + iy * self.grid_size
                cell_max_y = cell_min_y + self.grid_size
                if circle_intersects_cell(
                    x,
                    y,
                    radius,
                    cell_min_x,
                    cell_min_y,
                    cell_max_x,
                    cell_max_y,
                ):
                    yield ix, iy

    # timers
    def _task_progress_callback(self, robot_index: int, msg: Int32):
        if self.mission in {"delivery", "complex_task"}:
            self._write_task_progress()
            self._write_task_result()
            return
        progress = min(
            self.total_task_targets, max(0, int(msg.data))
        )
        self.robot_task_progress[robot_index] = max(
            self.robot_task_progress[robot_index], progress
        )
        self._write_task_progress()
        self._write_task_result()

    def _task_progress_event_callback(self, robot_index: int, msg: String):
        color = msg.data.strip().lower()
        if color not in ("red", "green", "blue"):
            return
        if self.mission == "complex_task":
            if sum(self.team_delivered_boxes.values()) >= self.task_progress_on_complete:
                return
            elapsed = time.time() - self._wall_start
            self.team_delivered_boxes[color] += 1
            self.robot_task_progress[robot_index] += 1
            if self.robot_color_reached_times[robot_index][color] is None:
                self.robot_color_reached_times[robot_index][color] = elapsed
            self._write_status(
                f"robot_{robot_index} delivered {color} box "
                f"{self.team_delivered_boxes[color]} at {elapsed:.3f}s.\n"
            )
            if sum(self.team_delivered_boxes.values()) >= self.task_progress_on_complete:
                self._complete_delivery_task(elapsed)
            self._write_task_progress()
            self._write_task_result()
            return
        if self.mission == "delivery":
            if color not in self.team_completed_colors:
                elapsed = time.time() - self._wall_start
                self.team_completed_colors.add(color)
                self.delivery_color_robot[color] = robot_index
                self.delivery_color_times[color] = elapsed
                self.robot_task_progress[robot_index] += 1
                self.robot_color_reached_times[robot_index][color] = elapsed
                self._write_status(
                    f"robot_{robot_index} delivered {color} at "
                    f"{elapsed:.3f}s.\n"
                )
                if len(self.team_completed_colors) == len(
                    self.delivery_color_times
                ):
                    self._complete_delivery_task(elapsed)
            self._write_task_progress()
            self._write_task_result()
            return
        reached_times = self.robot_color_reached_times[robot_index]
        if reached_times[color] is None:
            reached_times[color] = time.time() - self._wall_start
        self._write_task_progress()

    def _task_complete_callback(self, robot_index: int, msg: Bool):
        if self.mission in {"delivery", "complex_task"}:
            completed = (
                sum(self.team_delivered_boxes.values())
                if self.mission == "complex_task"
                else len(self.team_completed_colors)
            )
            required = (
                self.task_progress_on_complete
                if self.mission == "complex_task"
                else len(self.delivery_color_times)
            )
            if msg.data and completed >= required:
                self._complete_delivery_task(
                    time.time() - self._wall_start
                )
            return
        if not msg.data or robot_index in self.completed_robots:
            return
        self.completed_robots.add(robot_index)
        self.robot_completion_times[robot_index] = time.time() - self._wall_start
        self.robot_task_progress[robot_index] = self.task_progress_on_complete
        self._write_status(f"robot_{robot_index} completed the patrolling task.\n")
        if len(self.completed_robots) == self.total_robots:
            self.task_completion_duration = time.time() - self._wall_start
            self._write_status(
                "Task success: true\n"
                f"Task duration: {self.task_completion_duration:.3f}s\n"
            )
            self._write_task_result()
            if self.shutdown_on_task_complete:
                self.shutdown_and_save()

    def _complete_delivery_task(self, elapsed: float):
        if self.task_completion_duration is not None:
            return
        self.task_completion_duration = elapsed
        self.completed_robots = set(range(self.total_robots))
        for robot_index in range(self.total_robots):
            self.robot_completion_times[robot_index] = elapsed
        self._write_status(
            "Delivery task success: true\n"
            f"Task duration: {elapsed:.3f}s\n"
        )
        self._write_task_result()
        self._write_task_progress()
        if self.shutdown_on_task_complete:
            self.shutdown_and_save()

    def _on_metrics_timer(self):
        if self._saving_now:
            return
        self._update_metrics()

    def _on_timeout_timer(self):
        elapsed = time.time() - self._wall_start
        if elapsed >= self.run_duration:
            if (
                self.wait_for_all_task_completion
                and len(self.completed_robots) < self.total_robots
            ):
                if not self._nominal_timeout_reported:
                    self._nominal_timeout_reported = True
                    self._write_status(
                        f"Nominal duration reached ({self.run_duration}s), "
                        "but robots remain unfinished; continuing simulation.\n"
                    )
                return
            self._write_status(f"Timeout reached ({self.run_duration}s). Saving + shutdown.\n")
            self.shutdown_and_save()

    def _update_metrics(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        free_cells = max(1, (len(self.cells) - len(self.blocked)))
        coverage_pct = (len(self.visited) / free_cells) * 100.0
        self.coverage_history.append((elapsed, coverage_pct))

    # saving
    def shutdown_and_save(self):
        self._saving_now = True
        try:
            try:
                self.timer_plot.cancel()
                self.timer_timeout.cancel()
            except Exception:
                pass

            self._update_metrics()
            self.save_all_artifacts()
        finally:
            self._saving_now = False
            # Launch shutdown is triggered by this process exiting. Some ROS 2
            # middleware configurations can block indefinitely inside
            # rclpy.shutdown(), leaving Gazebo alive after all artifacts were
            # already saved. Arm a short post-save fallback that guarantees the
            # process exit and therefore the launch OnProcessExit event.
            threading.Timer(0.5, lambda: os._exit(0)).start()
            rclpy.shutdown()

    def save_all_artifacts(self):
        if self._saved_ok:
            return
        try:
            self._write_status("Saving started...\n")

            self._flush_buffers(force=True)
            self._write_timeseries_csvs()
            self._write_task_result()
            self._write_task_progress()
            if self.task_completion_duration is None:
                self._write_status(
                    "Task success: false\n"
                    f"Task duration: {time.time() - self._wall_start:.3f}s\n"
                )

            self._saved_ok = True
            self._write_status("Saving OK\n")
        except Exception:
            self.error_path.write_text(traceback.format_exc(), encoding="utf-8")
            self._write_status("Saving FAILED (see SAVE_ERROR.txt)\n")

    def _write_timeseries_csvs(self):
        with self.coverage_csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "coverage_pct"])
            for t, cov in self.coverage_history:
                w.writerow([f"{t:.3f}", f"{cov:.3f}"])

    def _write_task_result(self):
        if self.mission in {"delivery", "complex_task"}:
            if self.mission == "complex_task":
                completed_targets = sum(self.team_delivered_boxes.values())
                total_targets = self.task_progress_on_complete
                success = completed_targets >= total_targets
                contributing_robots = sum(
                    progress > 0 for progress in self.robot_task_progress.values()
                )
            else:
                completed_targets = len(self.team_completed_colors)
                total_targets = len(self.delivery_color_times)
                success = completed_targets == total_targets
                contributing_robots = len(
                    {
                        robot
                        for robot in self.delivery_color_robot.values()
                        if robot is not None
                    }
                )
        else:
            completed_targets = sum(self.robot_task_progress.values())
            total_targets = self.total_robots * self.total_task_targets
            success = len(self.completed_robots) == self.total_robots
            contributing_robots = len(self.completed_robots)
        duration = (
            self.task_completion_duration
            if self.task_completion_duration is not None
            else time.time() - self._wall_start
        )
        with self.task_result_path.open("w", newline="") as f:
            writer = csv.writer(f)
            progress_pct = 100.0 * completed_targets / total_targets
            writer.writerow(
                [
                    "success",
                    "duration_s",
                    "completed_robots",
                    "total_robots",
                    "completed_targets",
                    "total_targets",
                    "progress_pct",
                ]
            )
            writer.writerow(
                [
                    str(success).lower(),
                    f"{duration:.3f}",
                    contributing_robots,
                    self.total_robots,
                    completed_targets,
                    total_targets,
                    f"{progress_pct:.3f}",
                ]
            )

    def _write_task_progress(self):
        if self.mission == "complex_task":
            self._write_complex_task_progress()
            return
        if self.mission == "delivery":
            self._write_delivery_progress()
            return
        with self.task_progress_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "robot",
                    "completed_colors",
                    "total_colors",
                    "progress_pct",
                    "complete",
                    "completion_s",
                    "red_reached_s",
                    "green_reached_s",
                    "blue_reached_s",
                ]
            )
            for robot_index in range(self.total_robots):
                completed = self.robot_task_progress[robot_index]
                reached_times = self.robot_color_reached_times[robot_index]
                writer.writerow(
                    [
                        f"robot_{robot_index}",
                        completed,
                        self.total_task_targets,
                        f"{100.0 * completed / self.total_task_targets:.3f}",
                        str(completed == self.total_task_targets).lower(),
                        self._format_optional_time(
                            self.robot_completion_times[robot_index]
                        ),
                        self._format_optional_time(reached_times["red"]),
                        self._format_optional_time(reached_times["green"]),
                        self._format_optional_time(reached_times["blue"]),
                    ]
                )

    def _write_delivery_progress(self):
        with self.task_progress_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "robot",
                    "delivered_colors",
                    "delivered_count",
                    "team_total_colors",
                    "contribution_pct",
                    "team_complete",
                    "completion_s",
                    "red_delivered_s",
                    "green_delivered_s",
                    "blue_delivered_s",
                ]
            )
            team_complete = (
                len(self.team_completed_colors) == len(self.delivery_color_times)
            )
            for robot_index in range(self.total_robots):
                colors = [
                    color
                    for color, owner in self.delivery_color_robot.items()
                    if owner == robot_index
                ]
                reached_times = self.robot_color_reached_times[robot_index]
                writer.writerow(
                    [
                        f"robot_{robot_index}",
                        "|".join(colors),
                        len(colors),
                        len(self.delivery_color_times),
                        f"{100.0 * len(colors) / len(self.delivery_color_times):.3f}",
                        str(team_complete).lower(),
                        self._format_optional_time(
                            self.task_completion_duration
                        ),
                        self._format_optional_time(reached_times["red"]),
                        self._format_optional_time(reached_times["green"]),
                        self._format_optional_time(reached_times["blue"]),
                    ]
                )

    def _write_complex_task_progress(self):
        total_delivered = sum(self.team_delivered_boxes.values())
        team_complete = total_delivered >= self.task_progress_on_complete
        with self.task_progress_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "robot", "delivered_boxes", "team_delivered_boxes",
                    "team_total_boxes", "contribution_pct", "team_complete",
                    "completion_s", "red_boxes", "green_boxes", "blue_boxes",
                ]
            )
            for robot_index in range(self.total_robots):
                delivered = self.robot_task_progress[robot_index]
                writer.writerow(
                    [
                        f"robot_{robot_index}", delivered, total_delivered,
                        self.task_progress_on_complete,
                        f"{100.0 * delivered / self.task_progress_on_complete:.3f}",
                        str(team_complete).lower(),
                        self._format_optional_time(self.task_completion_duration),
                        self.team_delivered_boxes["red"],
                        self.team_delivered_boxes["green"],
                        self.team_delivered_boxes["blue"],
                    ]
                )

    @staticmethod
    def _format_optional_time(value):
        return "" if value is None else f"{value:.3f}"

    def _ensure_paths_csv_header(self):
        if not self.paths_csv_path.exists():
            with self.paths_csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "stamp_sec", "stamp_nsec",
                    "elapsed_s",
                    "robot",
                    "x", "y"
                ])

    def _ensure_cells_csv_header(self):
        if not self.visited_cells_csv_path.exists():
            with self.visited_cells_csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "stamp_sec", "stamp_nsec",
                    "robot",
                    "cell_index",
                    "cell_min_x", "cell_min_y",
                    "cell_center_x", "cell_center_y"
                ])

    def _append_path_row(self, robot: str, x: float, y: float):
        try:
            robot_index = int(robot.removeprefix("robot_"))
        except ValueError:
            robot_index = -1
        if robot_index in self.completed_robots:
            return
        stamp = self.get_clock().now().to_msg()
        elapsed_s = time.time() - self._wall_start
        self._path_rows.append([
            int(stamp.sec), int(stamp.nanosec),
            f"{elapsed_s:.3f}",
            robot,
            f"{x:.3f}", f"{y:.3f}"
        ])
        if len(self._path_rows) >= self.flush_max_rows:
            self._flush_buffers()

    def _append_visited_cell_row(self, robot: str, idx: int, cx: float, cy: float):
        stamp = self.get_clock().now().to_msg()
        center_x = cx + (self.grid_size * 0.5)
        center_y = cy + (self.grid_size * 0.5)
        self._cell_rows.append([
            int(stamp.sec), int(stamp.nanosec),
            robot,
            int(idx),
            f"{cx:.3f}", f"{cy:.3f}",
            f"{center_x:.3f}", f"{center_y:.3f}"
        ])
        if len(self._cell_rows) >= self.flush_max_rows:
            self._flush_buffers()

    def _flush_buffers(self, force: bool = False):
        if not self._path_rows and not self._cell_rows:
            return
        if not force and self._saving_now:
            return
        if self._path_rows:
            with self.paths_csv_path.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerows(self._path_rows)
            self._path_rows.clear()
        if self._cell_rows:
            with self.visited_cells_csv_path.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerows(self._cell_rows)
            self._cell_rows.clear()

    def _write_status(self, text: str):
        with self.status_path.open("a", encoding="utf-8") as f:
            f.write(text)

    def _save_run_metadata(self):
        prompt_source = self.prompt_file_path
        if prompt_source is None and self.metadata_yaml_path is not None:
            sibling_prompt = self.metadata_yaml_path.with_suffix(".prompt.txt")
            if sibling_prompt.exists():
                prompt_source = sibling_prompt

        if prompt_source is not None and prompt_source.exists():
            prompt_contents = prompt_source.read_text(encoding="utf-8")
        else:
            if prompt_source is not None and not prompt_source.exists():
                self._write_status(
                    f"WARNING: Prompt file not found: {prompt_source}\n"
                )
            prompt_contents = self.prompt_text
        self.prompt_txt_path.write_text(prompt_contents, encoding="utf-8")
        if self.metadata_yaml_path is None:
            self._write_status("WARNING: No metadata_yaml_path provided.\n")
            return
        if not self.metadata_yaml_path.exists():
            self._write_status(f"WARNING: YAML not found: {self.metadata_yaml_path}\n")
            return
        try:
            shutil.copy2(self.metadata_yaml_path, self.results_dir / self.metadata_yaml_path.name)
        except Exception as exc:
            self._write_status(f"WARNING: Failed to copy YAML: {exc}\n")

    # obstacles
    def _compute_blocked_cells(self):
        obstacles = self._load_obstacle_rectangles()
        if not obstacles:
            return set()

        blocked = set()
        cell_area = self.grid_size * self.grid_size
        for idx, (cx, cy) in enumerate(self.cells):
            cell_min_x = cx
            cell_min_y = cy
            cell_max_x = cx + self.grid_size
            cell_max_y = cy + self.grid_size
            for obs in obstacles:
                overlap = self._overlap_area_obstacle_cell(
                    obs, cell_min_x, cell_min_y, cell_max_x, cell_max_y
                )
                threshold = (
                    self.circle_obstacle_occupancy_threshold
                    if obs[0] == "circle"
                    else self.obstacle_occupancy_threshold
                )
                if (overlap / cell_area) >= threshold:
                    blocked.add(idx)
                    break
        return blocked

    def _load_obstacle_rectangles(self):
        if not self.world_sdf.exists():
            self._write_status(f"WARNING: SDF not found: {self.world_sdf}\n")
            return []
        try:
            root = ET.parse(self.world_sdf).getroot()
        except ET.ParseError as exc:
            self._write_status(f"WARNING: Failed to parse SDF: {exc}\n")
            return []
        world = root.find("world")
        if world is None:
            return []
        obstacles = []
        for model in world.findall("model"):
            name = model.get("name", "")
            if name == "ground_plane" or name.endswith("_wall"):
                continue
            model_pose = self._parse_pose(model.findtext("pose"))
            for link in model.findall("link"):
                link_pose = self._parse_pose(link.findtext("pose"))
                for collision in link.findall("collision"):
                    collision_pose = self._parse_pose(collision.findtext("pose"))
                    x = model_pose[0] + link_pose[0] + collision_pose[0]
                    y = model_pose[1] + link_pose[1] + collision_pose[1]
                    yaw = model_pose[5] + link_pose[5] + collision_pose[5]

                    size_text = collision.findtext("geometry/box/size")
                    if size_text:
                        sx, sy, _ = (float(v) for v in size_text.split())
                        obstacles.append(("box", x, y, sx, sy, yaw))
                        continue
                    radius_text = collision.findtext("geometry/cylinder/radius")
                    if radius_text:
                        obstacles.append(("circle", x, y, float(radius_text)))
        return obstacles

    def _parse_pose(self, pose_text):
        if not pose_text:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        vals = [float(v) for v in pose_text.split()]
        while len(vals) < 6:
            vals.append(0.0)
        return tuple(vals[:6])

    def _rect_corners(self, cx, cy, sx, sy, yaw):
        hx = sx / 2.0
        hy = sy / 2.0
        c = math.cos(yaw)
        s = math.sin(yaw)
        corners = []
        for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            x = cx + (dx * c - dy * s)
            y = cy + (dx * s + dy * c)
            corners.append((x, y))
        return corners

    def _clip_polygon(self, poly, edge_fn, intersect_fn):
        if not poly:
            return []
        output = []
        prev = poly[-1]
        prev_inside = edge_fn(prev)
        for curr in poly:
            curr_inside = edge_fn(curr)
            if curr_inside:
                if not prev_inside:
                    output.append(intersect_fn(prev, curr))
                output.append(curr)
            elif prev_inside:
                output.append(intersect_fn(prev, curr))
            prev, prev_inside = curr, curr_inside
        return output

    def _polygon_area(self, poly):
        if len(poly) < 3:
            return 0.0
        area = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            area += x1 * y2 - x2 * y1
        return abs(area) * 0.5

    def _overlap_area_obstacle_cell(self, obstacle, min_x, min_y, max_x, max_y):
        if obstacle[0] == "circle":
            return self._overlap_area_circle_cell(obstacle, min_x, min_y, max_x, max_y)
        return self._overlap_area_rect_cell(obstacle, min_x, min_y, max_x, max_y)

    def _overlap_area_rect_cell(self, obstacle, min_x, min_y, max_x, max_y):
        _, cx, cy, sx, sy, yaw = obstacle
        poly = self._rect_corners(cx, cy, sx, sy, yaw)

        def clip_left(p):   return p[0] >= min_x
        def clip_right(p):  return p[0] <= max_x
        def clip_bottom(p): return p[1] >= min_y
        def clip_top(p):    return p[1] <= max_y

        def intersect_x(p1, p2, xk):
            x1, y1 = p1
            x2, y2 = p2
            if x1 == x2:
                return (xk, y1)
            t = (xk - x1) / (x2 - x1)
            return (xk, y1 + t * (y2 - y1))

        def intersect_y(p1, p2, yk):
            x1, y1 = p1
            x2, y2 = p2
            if y1 == y2:
                return (x1, yk)
            t = (yk - y1) / (y2 - y1)
            return (x1 + t * (x2 - x1), yk)

        poly = self._clip_polygon(poly, clip_left,   lambda a, b: intersect_x(a, b, min_x))
        poly = self._clip_polygon(poly, clip_right,  lambda a, b: intersect_x(a, b, max_x))
        poly = self._clip_polygon(poly, clip_bottom, lambda a, b: intersect_y(a, b, min_y))
        poly = self._clip_polygon(poly, clip_top,    lambda a, b: intersect_y(a, b, max_y))
        return self._polygon_area(poly)

    def _overlap_area_circle_cell(self, obstacle, min_x, min_y, max_x, max_y):
        _, cx, cy, radius = obstacle
        samples_per_axis = 10
        inside = 0
        step_x = (max_x - min_x) / samples_per_axis
        step_y = (max_y - min_y) / samples_per_axis
        radius_sq = radius * radius
        for ix in range(samples_per_axis):
            px = min_x + (ix + 0.5) * step_x
            for iy in range(samples_per_axis):
                py = min_y + (iy + 0.5) * step_y
                if (px - cx) ** 2 + (py - cy) ** 2 <= radius_sq:
                    inside += 1
        return (inside / (samples_per_axis * samples_per_axis)) * (
            (max_x - min_x) * (max_y - min_y)
        )


def main(args=None):
    rclpy.init(args=args)
    node = CoverageCounter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._write_status("KeyboardInterrupt received. Saving + shutdown.\n")
        node.shutdown_and_save()
    except BaseException:
        node.error_path.write_text(traceback.format_exc(), encoding="utf-8")
        node._write_status("BaseException received. Attempting save in finally.\n")
    finally:
        try:
            if not node._saved_ok:
                node.save_all_artifacts()
        except Exception:
            node.error_path.write_text(traceback.format_exc(), encoding="utf-8")
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
