"""Launch image processing and one SCT supervisor on a physical LEO rover."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


MISSIONS = {"exploration", "patrolling", "delivery"}


def _nodes(context):
    mission = LaunchConfiguration("mission").perform(context).strip().lower()
    if mission not in MISSIONS:
        raise ValueError(f"mission must be one of {sorted(MISSIONS)}, got {mission!r}")

    namespace = LaunchConfiguration("robot_ns").perform(context).strip("/")
    package_share = Path(get_package_share_directory("leo_real_experiments"))
    config = package_share / "config" / "real_rover.yaml"
    supervisor_yaml = LaunchConfiguration("supervisor_yaml_path").perform(context).strip()
    if not supervisor_yaml:
        best_yaml = package_share / "BEST_YAMLS" / f"{mission}_fixed.yaml"
        if best_yaml.is_file():
            supervisor_yaml = str(best_yaml)
    enable_supervisor = LaunchConfiguration("enable_supervisor").perform(context).lower() in {
        "1", "true", "yes", "on"
    }
    supervisor_parameters = [str(config)]
    supervisor_parameters.append({"enabled": enable_supervisor})
    if supervisor_yaml:
        supervisor_parameters.append({"supervisor_yaml_path": supervisor_yaml})

    common = {"package": "leo_real_experiments", "namespace": namespace, "output": "screen"}
    nodes = [
        Node(executable="image_processor", name="image_processor", parameters=[str(config)], **common),
        Node(executable="color_detector", name="color_detector", parameters=[str(config)], **common),
        Node(executable="stuck_recovery", name="stuck_recovery", parameters=[str(config)], **common),
        Node(
            executable=f"{mission}_supervisor",
            name="robot_supervisor",
            parameters=supervisor_parameters,
            **common,
        ),
    ]
    if mission == "delivery":
        nodes.append(
            Node(executable="delivery_shape_detector", name="delivery_shape_detector", **common)
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("mission", default_value="exploration"),
            DeclareLaunchArgument("robot_ns", default_value=""),
            DeclareLaunchArgument("enable_supervisor", default_value="true"),
            DeclareLaunchArgument("supervisor_yaml_path", default_value=""),
            OpaqueFunction(function=_nodes),
        ]
    )
