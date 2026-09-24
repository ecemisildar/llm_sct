"""Launch three robots, six generic boxes, and red/blue delivery zones."""

import copy
import importlib.util
import math
import os
import random
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetLaunchConfiguration,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


_TEMP_WORLDS = []


def _create_balanced_world(context):
    package_share = get_package_share_directory("leo_delivery")
    source = os.path.join(package_share, "worlds", "delivery_arena_6x8.sdf")
    tree = ET.parse(source)
    world = tree.getroot().find("world")
    if world is None:
        raise RuntimeError(f"No world element in {source}")
    models = {model.get("name"): model for model in world.findall("model")}
    template = copy.deepcopy(models["green_delivery_box"])
    for model in list(world.findall("model")):
        name = model.get("name", "")
        if name == "green_box" or name.endswith("_delivery_box"):
            world.remove(model)
    helper_path = os.path.join(
        get_package_share_directory("leo_gz_bringup"),
        "launch", "world_randomization.py",
    )
    spec = importlib.util.spec_from_file_location("balanced_randomization", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    seed_text = LaunchConfiguration("random_seed").perform(context)
    used_seed = helper.randomize_task_models(world, "balanced_delivery", seed_text)
    rng = random.Random(used_seed ^ 0xB41A6CED)
    zones = [helper._xy(models[name]) for name in ("red_box", "blue_box")]
    positions = []
    for _index in range(6):
        for _attempt in range(10000):
            candidate = (rng.uniform(-2.25, 2.25), rng.uniform(-3.25, 3.25))
            if any(math.hypot(candidate[0] - x, candidate[1] - y) < 1.5
                   for x, y in zones + positions):
                continue
            positions.append(candidate)
            break
        else:
            raise RuntimeError("Could not place three separated delivery boxes")
    for index, (x, y) in enumerate(positions):
        box = copy.deepcopy(template)
        box.set("name", f"generic_delivery_box_{index}")
        box.find("pose").text = f"{x:.6f} {y:.6f} 0.125 0 0 0"
        world.append(box)
    if "top_view_camera" not in models:
        camera_source = os.path.join(package_share, "worlds", "delivery_world.sdf")
        camera_world = ET.parse(camera_source).getroot().find("world")
        camera = next(
            (model for model in camera_world.findall("model")
             if model.get("name") == "top_view_camera"),
            None,
        )
        if camera is None:
            raise RuntimeError(f"No top_view_camera model in {camera_source}")
        world.append(copy.deepcopy(camera))
    handle = tempfile.NamedTemporaryFile(
        prefix="leo_balanced_delivery_", suffix=".sdf", delete=False
    )
    handle.close()
    tree.write(handle.name, encoding="unicode", xml_declaration=True)
    _TEMP_WORLDS.append(handle.name)
    return [SetLaunchConfiguration("balanced_world", handle.name)]


def generate_launch_description():
    package_share = get_package_share_directory("leo_delivery")
    base_launch = os.path.join(package_share, "launch", "leo_gz.launch.py")
    default_yaml = os.path.join(
        package_share, "config", "balanced_delivery", "supervisor.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="true"),
            DeclareLaunchArgument("overhead_camera", default_value="true"),
            DeclareLaunchArgument("record_video", default_value="false"),
            DeclareLaunchArgument("run_duration", default_value="180.0"),
            DeclareLaunchArgument("random_seed", default_value="12345"),
            DeclareLaunchArgument(
                "results_dir",
                default_value=os.path.join(
                    os.path.expanduser("~"), "sct_ws", "src", "llm_sct",
                    "RESULTS_LAST", "delivery",
                ),
            ),
            DeclareLaunchArgument(
                "supervisor_yaml_path", default_value=default_yaml
            ),
            OpaqueFunction(function=_create_balanced_world),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments={
                    "sim_world": LaunchConfiguration("balanced_world"),
                    "headless": LaunchConfiguration("headless"),
                    "overhead_camera": LaunchConfiguration("overhead_camera"),
                    "record_video": LaunchConfiguration("record_video"),
                    "run_duration": LaunchConfiguration("run_duration"),
                    "random_seed": LaunchConfiguration("random_seed"),
                    "results_dir": LaunchConfiguration("results_dir"),
                    "total_robots": "3",
                    "supervisor_executable": "balanced_delivery_supervisor",
                    "metadata_yaml_path": LaunchConfiguration(
                        "supervisor_yaml_path"
                    ),
                    "evaluation_mission": "balanced_delivery",
                    "task_progress_on_complete": "6",
                    "object_cluster_half_width": "0.50",
                    "target_detector_executable": "color_detector",
                }.items(),
            ),
        ]
    )
