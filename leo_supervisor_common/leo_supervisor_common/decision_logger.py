import csv
import math
from pathlib import Path
from typing import Mapping, Sequence


class SctDecisionLogger:
    """Write the decision and event-summary CSV files used by every task."""

    BASE_COLUMNS = (
        "elapsed_s",
        "robot",
        "selected_event",
        "uncontrollable_events",
        "raw_zone",
    )

    def __init__(
        self,
        *,
        enabled: bool,
        robot: str,
        results_dir: str,
        yaml_path: str,
        total_robots: int,
        run_id: str,
        snapshot_columns: Sequence[str] = (),
    ):
        self.enabled = enabled
        self.robot = robot
        self.snapshot_columns = tuple(snapshot_columns)
        self.event_counts = {}
        self.decision_path = None
        self.summary_path = None

        if not enabled:
            return

        log_dir = self._run_log_dir(
            results_dir, yaml_path, total_robots, run_id
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        self.decision_path = log_dir / f"selected_events_{robot}.csv"
        self.summary_path = log_dir / f"event_percentages_{robot}.csv"

        if not self.decision_path.exists():
            with self.decision_path.open("w", newline="") as output:
                csv.writer(output).writerow(
                    [*self.BASE_COLUMNS, *self.snapshot_columns]
                )

    @staticmethod
    def _run_log_dir(
        results_dir: str,
        yaml_path: str,
        total_robots: int,
        run_id: str,
    ) -> Path:
        if not results_dir:
            return Path.cwd()
        yaml_group = Path(yaml_path).stem if yaml_path else "unknown_yaml"
        group_dir = Path(results_dir) / yaml_group / f"robots_{total_robots}"
        return group_dir / run_id if run_id else group_dir

    @staticmethod
    def _format_value(value):
        if isinstance(value, float):
            return f"{value:.6f}" if math.isfinite(value) else "inf"
        return value

    def log(
        self,
        selected_event: str,
        ce_exists: bool,
        uncontrollable_events: Sequence[str],
        snapshot: Mapping,
    ):
        if not self.enabled or self.decision_path is None:
            return

        if ce_exists:
            self.event_counts[selected_event] = (
                self.event_counts.get(selected_event, 0) + 1
            )

        row = [
            f"{snapshot['elapsed_s']:.3f}",
            self.robot,
            selected_event,
            "|".join(uncontrollable_events) or "none",
            snapshot["raw_zone"],
        ]
        row.extend(
            self._format_value(snapshot[column])
            for column in self.snapshot_columns
        )
        with self.decision_path.open("a", newline="") as output:
            csv.writer(output).writerow(row)

    def write_summary(self):
        if not self.enabled or self.summary_path is None:
            return

        total = sum(self.event_counts.values())
        with self.summary_path.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["robot", "selected_event", "count", "percentage"])
            for event, count in sorted(self.event_counts.items()):
                percentage = (100.0 * count / total) if total else 0.0
                writer.writerow(
                    [self.robot, event, count, f"{percentage:.3f}"]
                )
            writer.writerow(
                [
                    self.robot,
                    "TOTAL",
                    total,
                    "100.000" if total else "0.000",
                ]
            )
