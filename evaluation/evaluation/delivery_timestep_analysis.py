"""Reconstruct delivery steps using ROS timestamps matched to saved event CSVs."""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analyze_prompt_sensitivity import selected_event_paths, write_csv

ZONE = {"EV_drop_zone_red": "red", "EV_drop_zone_a": "red",
        "EV_drop_zone_blue": "blue", "EV_drop_zone_b": "blue"}
SELECTED = re.compile(r"\[([0-9]+\.[0-9]+)\] \[(robot_\d+)\.robot_supervisor\]: Selected controllable event: (\S+)")


def analyze_timesteps(runs, output: Path, log_root: Path):
    saved = {}
    wanted = set()
    for run in runs:
        entries = []
        for path in sorted(selected_event_paths(Path(run["run"]))):
            with path.open(newline="") as stream:
                records = list(csv.DictReader(stream))
            if not records:
                continue
            robot = records[0]["robot"]
            sequence = tuple(r["selected_event"] for r in records)
            key = (robot, sequence[:12])
            wanted.add(key)
            entries.append((robot, sequence, records, key))
        saved[run["run"]] = entries
    logs = defaultdict(list)
    for path in log_root.rglob("python3*.log"):
        records = []
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = SELECTED.search(line)
                if match:
                    records.append((float(match[1]), match[2], match[3]))
        if records:
            key = (records[0][1], tuple(r[2] for r in records[:12]))
            if key in wanted:
                logs[key].append((path, records))

    robot_steps = []
    for run in runs:
        for robot, sequence, records, key in saved[run["run"]]:
            red = blue = 0
            for record in records:
                zone = ZONE.get(record["selected_event"])
                if not zone:
                    continue
                red += zone == "red"
                blue += zone == "blue"
                robot_steps.append({**{k: run[k] for k in ("prompt", "generation", "collision_control", "seed", "run")},
                                    "robot": robot, "robot_elapsed_s": record["elapsed_s"],
                                    "zone": zone, "robot_red_boxes": red, "robot_blue_boxes": blue})
    write_csv(output / "delivery_zone_robot_timesteps.csv", robot_steps)
    steps, summaries, trajectories = [], [], {}
    for run in runs:
        entries = saved[run["run"]]
        drops, origins, missing = [], [], []
        for robot, sequence, records, key in entries:
            candidates = [(path, events) for path, events in logs[key]
                          if tuple(e[2] for e in events[:len(sequence)]) == sequence[:len(events)]
                          and not any(e in ZONE for e in sequence[len(events):])]
            if len(candidates) != 1:
                if not any(e in ZONE for e in sequence):
                    continue
                missing.append(robot)
                continue
            path, events = candidates[0]
            origins.append(events[0][0] - float(records[0]["elapsed_s"]))
            for index, record in enumerate(records):
                if record["selected_event"] in ZONE:
                    drops.append((events[index][0], robot, ZONE[record["selected_event"]],
                                  record["elapsed_s"], str(path)))
        verified = not missing and bool(entries)
        prefix = {k: run[k] for k in ("prompt", "generation", "collision_control", "seed", "supervisor", "run")}
        if not verified:
            summaries.append({**prefix, "timing_verified": False,
                              "missing_robot_logs": ";".join(missing) or "missing event CSVs",
                              "red_boxes": run["red_boxes"], "blue_boxes": run["blue_boxes"],
                              "max_abs_difference": "", "balance_always_within_1": "",
                              "first_violation_elapsed_s": "",
                              "delivery_count_matches_events": run["delivery_count_matches_events"]})
            continue
        origin = min(origins)
        red = blue = max_difference = 0
        first_violation = ""
        trajectory = [(0., 0, 0, 0)]
        steps.append({**prefix, "timestep": 0, "elapsed_s": 0., "unix_time_s": origin,
                      "robot": "", "zone": "", "red_boxes": 0, "blue_boxes": 0,
                      "red_minus_blue": 0, "within_plus_minus_1": True,
                      "robot_elapsed_s": "", "source_log": ""})
        for index, (timestamp, robot, zone, robot_elapsed, source) in enumerate(sorted(drops), 1):
            red += zone == "red"
            blue += zone == "blue"
            difference = red - blue
            max_difference = max(max_difference, abs(difference))
            elapsed = timestamp - origin
            if abs(difference) > 1 and first_violation == "":
                first_violation = elapsed
            steps.append({**prefix, "timestep": index, "elapsed_s": elapsed,
                          "unix_time_s": timestamp, "robot": robot, "zone": zone,
                          "red_boxes": red, "blue_boxes": blue, "red_minus_blue": difference,
                          "within_plus_minus_1": abs(difference) <= 1,
                          "robot_elapsed_s": robot_elapsed, "source_log": source})
            trajectory.append((elapsed, red, blue, difference))
        trajectory.append((max(float(run["duration_s"]), trajectory[-1][0]), red, blue, red-blue))
        trajectories[run["run"]] = trajectory
        summaries.append({**prefix, "timing_verified": True, "missing_robot_logs": "",
                          "red_boxes": red, "blue_boxes": blue,
                          "max_abs_difference": max_difference,
                          "balance_always_within_1": max_difference <= 1,
                          "first_violation_elapsed_s": first_violation,
                          "delivery_count_matches_events": run["delivery_count_matches_events"]})
    write_csv(output / "delivery_zone_timesteps.csv", steps)
    write_csv(output / "delivery_balance_per_run.csv", summaries)
    # Every run gets a cell. The two rows of plots show zone counts and their difference.
    with PdfPages(output / "delivery_zone_timesteps.pdf") as pdf:
        groups = defaultdict(list)
        for run in runs:
            groups[(run["prompt"], run["collision_control"], run["generation"])].append(run)
        for (prompt, mode, generation), group in sorted(groups.items()):
            for start in range(0, len(group), 5):
                chunk = group[start:start+5]
                fig, axes = plt.subplots(2, len(chunk), figsize=(4*len(chunk), 6), squeeze=False)
                for column, run in enumerate(chunk):
                    top, bottom = axes[:, column]
                    trajectory = trajectories.get(run["run"])
                    top.set_title(f"Seed {run['seed']}", fontsize=12)
                    if trajectory:
                        t, r, b, d = zip(*trajectory)
                        top.step(t, r, where="post", color="firebrick", label="Red")
                        top.step(t, b, where="post", color="royalblue", label="Blue")
                        bottom.axhspan(-1, 1, color="green", alpha=.12)
                        bottom.step(t, d, where="post", color="black")
                        bottom.axhline(1, ls="--", color="green", lw=.7)
                        bottom.axhline(-1, ls="--", color="green", lw=.7)
                        limit = max(2, max(abs(x) for x in d))
                        bottom.set_ylim(-limit-.3, limit+.3)
                        bottom.set_yticks(range(-limit, limit+1))
                    else:
                        top.text(.5, .5, "Shared timing unavailable", transform=top.transAxes,
                                 ha="center", fontsize=10)
                    top.set_ylim(0, 6.3)
                    top.set_yticks(range(7))
                    bottom.set_xlabel("Seconds", fontsize=11)
                    for axis in (top, bottom):
                        axis.tick_params(labelsize=9)
                        axis.spines[["top", "right"]].set_visible(False)
                axes[0, 0].set_ylabel("Delivered boxes", fontsize=11)
                axes[1, 0].set_ylabel("Red − blue", fontsize=11)
                if axes[0, 0].lines:
                    axes[0, 0].legend(fontsize=9)
                fig.suptitle(f"P{prompt} · {mode} · Generation {generation}", fontsize=14)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    verified = [r for r in summaries if r["timing_verified"]]
    violations = sum(not r["balance_always_within_1"] for r in verified)
    print(f"Shared ROS timing verified for {len(verified)}/{len(runs)} runs; {violations} exceed ±1")
