# Copyright 2023 Fictionlab sp. z o.o.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import os
import sys
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_prefix, get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown, TimerAction, SetEnvironmentVariable, SetLaunchConfiguration
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, OpaqueFunction
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


_TEMP_WORLD_PATHS = []


def _configure_overhead_camera(context):
    world_path = LaunchConfiguration("sim_world").perform(context)
    enabled = LaunchConfiguration("overhead_camera").perform(context).lower() in (
        "1", "true", "yes", "on"
    )
    if enabled:
        return [SetLaunchConfiguration("effective_sim_world", world_path)]

    tree = ET.parse(world_path)
    world = tree.getroot().find("world")
    if world is None:
        raise RuntimeError(f"No <world> element found in {world_path}")
    for model in list(world.findall("model")):
        if model.get("name") == "top_view_camera":
            world.remove(model)

    handle = tempfile.NamedTemporaryFile(
        prefix="leo_delivery_no_overhead_camera_", suffix=".sdf", delete=False
    )
    handle.close()
    tree.write(handle.name, encoding="unicode", xml_declaration=True)
    _TEMP_WORLD_PATHS.append(handle.name)
    return [SetLaunchConfiguration("effective_sim_world", handle.name)]


def _cleanup_temp_worlds(*_args, **_kwargs):
    """Remove only temporary files created by this launch instance."""
    for path in _TEMP_WORLD_PATHS:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    _TEMP_WORLD_PATHS.clear()
    return []


def generate_launch_description():
    # Setup project paths
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_project_gazebo = get_package_share_directory("leo_delivery")
    evaluation_python_path = os.path.join(
        get_package_prefix("evaluation"),
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
    )
    python_path = os.pathsep.join(
        path
        for path in (evaluation_python_path, os.environ.get("PYTHONPATH", ""))
        if path
    )
    existing_ign_path = os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
    model_path = os.path.join(pkg_project_gazebo, "models")
    ign_resource_path = os.pathsep.join(
        [p for p in [pkg_project_gazebo, model_path, existing_ign_path] if p]
    )

    sim_world = DeclareLaunchArgument(
        "sim_world",
        default_value=os.path.join(pkg_project_gazebo, "worlds", "delivery_world.sdf"),
        description="Path to the Gazebo world file",
    )
    headless = DeclareLaunchArgument(
        "headless",
        default_value="true",
        description="Run Gazebo headless (no GUI)",
    )
    overhead_camera = DeclareLaunchArgument(
        "overhead_camera",
        default_value="false",
        description="Enable the fixed overhead camera sensor",
    )
    auto_start = DeclareLaunchArgument(
        "auto_start",
        default_value="true",
        description="Start physics immediately (run without manual play)",
    )
    run_duration = DeclareLaunchArgument(
        "run_duration",
        default_value="600.0",
        description="Seconds before shutting down the launch",
    )
    total_robots = DeclareLaunchArgument(
        "total_robots",
        default_value="3",
        description="Number of robots to spawn at safe, separated positions",
    )
    random_seed = DeclareLaunchArgument(
        "random_seed",
        default_value="auto",
        description="Base seed for per-robot supervisor random choices. Use 'auto' for a fresh seed each run.",
    )
    record_video = DeclareLaunchArgument("record_video", default_value="false")
    supervisor_executable = DeclareLaunchArgument(
        "supervisor_executable",
        default_value="robot_supervisor",
        description="Delivery supervisor executable; the default preserves the original task",
    )
    task_progress_on_complete = DeclareLaunchArgument(
        "task_progress_on_complete",
        default_value="3",
        description="Delivered-box progress value that completes the run",
    )
    object_cluster_half_width = DeclareLaunchArgument(
        "object_cluster_half_width",
        default_value="0.125",
        description="Half-width in metres of the delivery pickup cluster",
    )
    evaluation_mission = DeclareLaunchArgument(
        "evaluation_mission",
        default_value="delivery",
        description="Mission label used by the evaluation node",
    )
    
    results_dir = DeclareLaunchArgument(
        "results_dir",
        default_value=os.path.join(
            os.path.expanduser("~"),
            "sct_ws",
            "src",
            "llm_sct",
            "new_results",
            "baseline",
            "results_delivery",
        ),
        description="Directory to write run artifacts",
    )
    metadata_yaml_path = DeclareLaunchArgument(
        "metadata_yaml_path",
        default_value=os.path.join(
            pkg_project_gazebo,
            "config",
            "supervisor.yaml",
        ),
        description="YAML file to copy into each run folder",
    )

    # Setup to launch the simulator and Gazebo world
    gz_args = PythonExpression([
        "'",
        LaunchConfiguration("effective_sim_world"),
        "' + (' -s --headless-rendering' if '",
        LaunchConfiguration("headless"),
        "' == 'true' else '') + (' -r' if '",
        LaunchConfiguration("auto_start"),
        "' == 'true' else '')",
    ])
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    spawn_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_project_gazebo, "launch", "spawn_multi_robots.launch.py")
        ),
        launch_arguments={
            "sim_world": LaunchConfiguration("effective_sim_world"),
            "headless": LaunchConfiguration("headless"),
            "auto_start": LaunchConfiguration("auto_start"),
            "run_duration": LaunchConfiguration("run_duration"),
            "total_robots": LaunchConfiguration("total_robots"),
            "random_seed": LaunchConfiguration("random_seed"),
            "record_video": LaunchConfiguration("record_video"),
            "results_dir": LaunchConfiguration("results_dir"),
            "metadata_yaml_path": LaunchConfiguration("metadata_yaml_path"),
            "supervisor_executable": LaunchConfiguration("supervisor_executable"),
            "task_progress_on_complete": LaunchConfiguration("task_progress_on_complete"),
            "evaluation_mission": LaunchConfiguration("evaluation_mission"),
            "object_cluster_half_width": LaunchConfiguration(
                "object_cluster_half_width"
            ),
        }.items(),
    )

    # Bridge ROS topics and Gazebo messages for establishing communication
    topic_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="clock_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
        ],
        parameters=[
            {
                "qos_overrides./tf_static.publisher.durability": "transient_local",
            }
        ],
        output="screen",
    )
    return LaunchDescription(
        [
            SetEnvironmentVariable(
                name="IGN_GAZEBO_RESOURCE_PATH",
                value=ign_resource_path,
            ),
            SetEnvironmentVariable(name="PYTHONPATH", value=python_path),
            sim_world,
            headless,
            overhead_camera,
            auto_start,
            run_duration,
            total_robots,
            random_seed,
            record_video,
            supervisor_executable,
            task_progress_on_complete,
            object_cluster_half_width,
            evaluation_mission,
            results_dir,
            metadata_yaml_path,
            OpaqueFunction(function=_configure_overhead_camera),
            gz_sim,
            spawn_robot,
            topic_bridge,
            RegisterEventHandler(
                OnShutdown(
                    on_shutdown=[OpaqueFunction(function=_cleanup_temp_worlds)],
                )
            ),
        ]
    )
