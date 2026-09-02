"""Launch the nine-box delivery task without changing the original defaults."""

import copy
import os
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetLaunchConfiguration,
)
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


_TEMP_COMPLEX_WORLDS = []


def _create_nine_box_world(context):
    """Clone each compact pickup box twice before Gazebo starts."""
    package_share = get_package_share_directory("leo_delivery")
    source = os.path.join(package_share, "worlds", "delivery_world.sdf")
    tree = ET.parse(source)
    world = tree.getroot().find("world")
    if world is None:
        raise RuntimeError(f"No <world> element found in {source}")

    for color in ("red", "green", "blue"):
        model_name = f"{color}_delivery_box"
        original = next(
            (model for model in world.findall("model") if model.get("name") == model_name),
            None,
        )
        if original is None:
            raise RuntimeError(f"Pickup model {model_name!r} not found in {source}")
        pose = original.find("pose")
        if pose is None or not pose.text:
            raise RuntimeError(f"Pickup model {model_name!r} has no pose")
        values = pose.text.split()
        center_x = float(values[0])
        for box_index, x_offset in ((2, -0.35), (3, 0.35)):
            clone = copy.deepcopy(original)
            clone.set("name", f"{model_name}_{box_index}")
            clone_pose = clone.find("pose")
            clone_values = list(values)
            clone_values[0] = str(center_x + x_offset)
            clone_pose.text = " ".join(clone_values)
            world.append(clone)

    handle = tempfile.NamedTemporaryFile(
        prefix="leo_complex_nine_boxes_", suffix=".sdf", delete=False
    )
    handle.close()
    tree.write(handle.name, encoding="unicode", xml_declaration=True)
    _TEMP_COMPLEX_WORLDS.append(handle.name)
    return [SetLaunchConfiguration("complex_sim_world", handle.name)]


def _cleanup_complex_worlds(*_args, **_kwargs):
    for path in _TEMP_COMPLEX_WORLDS:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    _TEMP_COMPLEX_WORLDS.clear()
    return []


def generate_launch_description():
    package_share = get_package_share_directory("leo_delivery")
    launch_path = os.path.join(
        package_share, "launch", "leo_gz.launch.py"
    )
    supervisor_yaml_path = DeclareLaunchArgument(
        "supervisor_yaml_path",
        default_value=os.path.join(package_share, "config", "supervisor.yaml"),
        description="Path to the baseline or LLM-generated supervisor YAML",
    )
    results_dir = DeclareLaunchArgument(
        "results_dir",
        default_value=(
            "/home/ecem/sct_ws/src/llm_sct/new_results/llm/"
            "results_complex_task"
        ),
        description="Directory for complex-task run artifacts",
    )
    return LaunchDescription(
        [
            supervisor_yaml_path,
            results_dir,
            OpaqueFunction(function=_create_nine_box_world),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(launch_path),
                launch_arguments={
                    "supervisor_executable": "complex_delivery_supervisor",
                    "task_progress_on_complete": "9",
                    "evaluation_mission": "complex_task",
                    "metadata_yaml_path": LaunchConfiguration(
                        "supervisor_yaml_path"
                    ),
                    "results_dir": LaunchConfiguration("results_dir"),
                    "sim_world": LaunchConfiguration("complex_sim_world"),
                    # Three 0.25 m boxes centered at x offsets -0.35, 0, +0.35.
                    "object_cluster_half_width": "0.475",
                }.items(),
            ),
            RegisterEventHandler(
                OnShutdown(
                    on_shutdown=[OpaqueFunction(function=_cleanup_complex_worlds)]
                )
            ),
        ]
    )
