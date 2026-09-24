#!/usr/bin/env python3
"""Burn timestamped selected-event CSV logs into a sped-up run video."""

import argparse
import csv
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


COLORS = ("&H004FC3F7", "&H0066D98B", "&H00F0A35E")


def ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, hundredths = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{hundredths:02d}"


def ass_text(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def read_events(path: Path) -> tuple[str, list[tuple[float, str]]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return path.stem.removeprefix("selected_events_"), []
    robot = rows[0].get("robot", path.stem.removeprefix("selected_events_"))
    events = [
        (float(row["elapsed_s"]), row.get("selected_event", "none") or "none")
        for row in rows
    ]
    return robot, events


def creation_time(path: Path) -> float:
    """Return filesystem birth time (the run artifacts are created together)."""
    result = subprocess.run(
        ["stat", "--format=%w", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if value == "-":
        raise RuntimeError(f"filesystem creation time is unavailable for {path}")
    date_and_fraction, timezone = value.rsplit(" ", 1)
    if "." in date_and_fraction:
        date, fraction = date_and_fraction.split(".", 1)
        date_and_fraction = f"{date}.{fraction[:6]}"
    return datetime.strptime(
        f"{date_and_fraction} {timezone}", "%Y-%m-%d %H:%M:%S.%f %z"
    ).timestamp()


def align_events(
    events: list[tuple[float, str]], offset: float
) -> list[tuple[float, str]]:
    """Map supervisor elapsed time onto video time and clip pre-video events."""
    shifted = [(elapsed - offset, event) for elapsed, event in events]
    latest_before_video = None
    visible = []
    for timestamp, event in shifted:
        if timestamp <= 0:
            latest_before_video = event
        else:
            visible.append((timestamp, event))
    if latest_before_video is not None:
        visible.insert(0, (0.0, latest_before_video))
    return visible


def write_ass(
    path: Path, video: Path, csv_paths: list[Path], duration: float
) -> None:
    styles = []
    dialogues = []
    for index, csv_path in enumerate(csv_paths):
        robot, events = read_events(csv_path)
        offset = creation_time(video) - creation_time(csv_path)
        events = align_events(events, offset)
        print(f"{robot}: applying {offset:.3f} s supervisor-to-video offset")
        style = f"Robot{index}"
        color = COLORS[index % len(COLORS)]
        styles.append(
            f"Style: {style},DejaVu Sans,30,{color},&H00000000,&H00000000,"
            f"&H90000000,1,0,0,0,100,100,0,0,1,3,1,7,24,{24 + index * 42},24,1"
        )
        if events and events[0][0] > 0:
            events.insert(0, (0.0, "waiting"))
        for event_index, (start, event) in enumerate(events):
            end = events[event_index + 1][0] if event_index + 1 < len(events) else duration
            if end <= start:
                continue
            label = ass_text(f"{robot}: {event}")
            dialogues.append(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{label}"
            )

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
"""
    events_header = """

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    path.write_text(header + "\n".join(styles) + events_header + "\n".join(dialogues) + "\n")


def probe_duration(video: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be greater than zero")

    video = args.video.resolve()
    output = (args.output or video.with_name(f"{video.stem}_annotated_{args.speed:g}x.mp4")).resolve()
    csv_paths = sorted(video.parent.glob("selected_events_robot_*.csv"))
    if not csv_paths:
        parser.error(f"no selected_events_robot_*.csv files found beside {video}")

    with tempfile.TemporaryDirectory(prefix="annotated-video-") as temp_dir:
        subtitle_path = Path(temp_dir) / "selected_events.ass"
        write_ass(subtitle_path, video, csv_paths, probe_duration(video))
        escaped_subtitle = str(subtitle_path).replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video),
                "-vf", f"subtitles='{escaped_subtitle}',setpts=PTS/{args.speed}",
                "-an", "-r", "30", "-c:v", "libx264", "-preset", "medium",
                "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(output),
            ],
            check=True,
        )
    print(output)


if __name__ == "__main__":
    main()
