"""Launch the standard Leo hardware stack in this rover's namespace."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_ns = LaunchConfiguration("robot_ns")
    leo_bringup = PathJoinSubstitution(
        [FindPackageShare("leo_bringup"), "launch", "leo_bringup.launch.xml"]
    )
    realsense_bringup = PathJoinSubstitution(
        [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ns", default_value=""),
            GroupAction(
                [
                    PushRosNamespace(robot_ns),
                    IncludeLaunchDescription(AnyLaunchDescriptionSource(leo_bringup)),
                    IncludeLaunchDescription(
                        AnyLaunchDescriptionSource(realsense_bringup),
                        launch_arguments={
                            "camera_namespace": "",
                            "camera_name": "camera",
                            "enable_color": "true",
                            "enable_depth": "true",
                            "align_depth.enable": "true",
                        }.items(),
                    ),
                ]
            ),
        ]
    )
