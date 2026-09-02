"""Run delivery and patrolling supervisors together in one Gazebo world."""

import importlib.util
import math
import os
import random
import subprocess
import sys
import time

import xacro

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    Shutdown,
    TimerAction,
)
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _load_spawn_helpers():
    helper_path = os.path.join(
        get_package_share_directory("leo_gz_bringup"),
        "launch",
        "multi_robot_mission.py",
    )
    spec = importlib.util.spec_from_file_location("leo_spawn_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _kill_simulator_processes(*_args, **_kwargs):
    for signal in ("-INT", "-TERM", "-KILL"):
        for pattern in (r"gz sim", r"ign gazebo", r"parameter_bridge"):
            subprocess.run(
                ["pkill", signal, "-f", pattern],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        time.sleep(0.3)
    return []


def generate_launch_description():
    helpers = _load_spawn_helpers()
    ros_gz_sim = get_package_share_directory("ros_gz_sim")
    bringup_share = get_package_share_directory("leo_gz_bringup")
    delivery_share = get_package_share_directory("leo_delivery")
    patrolling_share = get_package_share_directory("leo_patrolling")
    leo_description = get_package_share_directory("leo_description")
    run_id = time.strftime("mixed_run_%Y%m%d_%H%M%S")

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
    resource_path = os.pathsep.join(
        path
        for path in (
            delivery_share,
            os.path.join(delivery_share, "models"),
            os.environ.get("IGN_GAZEBO_RESOURCE_PATH", ""),
        )
        if path
    )

    arguments = [
        DeclareLaunchArgument(
            "sim_world",
            default_value=os.path.join(delivery_share, "worlds", "delivery_world.sdf"),
            description="World containing both delivery and RGB patrolling targets",
        ),
        DeclareLaunchArgument("headless", default_value="true"),
        DeclareLaunchArgument("auto_start", default_value="true"),
        DeclareLaunchArgument("run_duration", default_value="600.0"),
        DeclareLaunchArgument("delivery_robots", default_value="3"),
        DeclareLaunchArgument("patrolling_robots", default_value="3"),
        DeclareLaunchArgument("random_seed", default_value="12345"),
        DeclareLaunchArgument("forward_probability", default_value="0.90"),
        DeclareLaunchArgument("target_color_order", default_value="red,green,blue"),
        DeclareLaunchArgument(
            "delivery_yaml",
            default_value=os.path.join(delivery_share, "config", "sup_delivery.yaml"),
        ),
        DeclareLaunchArgument(
            "patrolling_yaml",
            default_value=os.path.join(
                patrolling_share, "config", "sup_patrolling.yaml"
            ),
        ),
        DeclareLaunchArgument(
            "results_dir",
            default_value=os.path.join(
                os.path.expanduser("~"),
                "sct_ws",
                "src",
                "llm_sct",
                "new_results",
                "mixed",
            ),
        ),
    ]

    gz_args = PythonExpression(
        [
            "'",
            LaunchConfiguration("sim_world"),
            "' + (' -s' if '",
            LaunchConfiguration("headless"),
            "' == 'true' else '') + (' -r' if '",
            LaunchConfiguration("auto_start"),
            "' == 'true' else '')",
        ]
    )
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="clock_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock"],
        output="screen",
    )

    def create_robots(context):
        delivery_count = int(LaunchConfiguration("delivery_robots").perform(context))
        patrolling_count = int(
            LaunchConfiguration("patrolling_robots").perform(context)
        )
        if delivery_count < 0 or patrolling_count < 0:
            raise RuntimeError("Robot counts cannot be negative")
        total_robots = delivery_count + patrolling_count
        if total_robots < 1:
            raise RuntimeError("At least one robot must be requested")

        world_path = LaunchConfiguration("sim_world").perform(context)
        seed = helpers._resolve_seed(
            LaunchConfiguration("random_seed").perform(context), "random_seed"
        )
        slots = helpers._build_robot_spawn_slots(
            total_robots, world_path, random.Random(seed ^ 0x5A17C0DE)
        )
        print(
            f"[mixed_missions] delivery={delivery_count} "
            f"patrolling={patrolling_count} seed={seed}",
            flush=True,
        )

        robots = []
        for index, slot in enumerate(slots):
            is_delivery = index < delivery_count
            robots.append(
                {
                    "ns": f"robot_{index}",
                    "x": slot["x"],
                    "y": slot["y"],
                    "yaw": slot["yaw"],
                    "package": "leo_delivery" if is_delivery else "leo_patrolling",
                    "yaml": LaunchConfiguration(
                        "delivery_yaml" if is_delivery else "patrolling_yaml"
                    ),
                    "detector": (
                        "delivery_shape_detector" if is_delivery else "color_detector"
                    ),
                    "uses_color_order": not is_delivery,
                }
            )

        bridge_arguments = []
        for robot in robots:
            ns = robot["ns"]
            bridge_arguments.extend(
                [
                    f"/{ns}/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
                    f"/{ns}/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
                    f"/{ns}/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
                    f"/{ns}/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model",
                    f"/{ns}/depth_camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
                    f"/{ns}/depth_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
                    f"/world/random_world/model/{ns}/link/{ns}/base_footprint/"
                    "sensor/contact_sensor/contact"
                    "@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts",
                ]
            )
        bridge_arguments.extend(
            [
                "/world/random_world/dynamic_pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
                "/world/random_world/remove@ros_gz_interfaces/srv/DeleteEntity",
                "/world/random_world/set_pose@ros_gz_interfaces/srv/SetEntityPose",
            ]
        )
        nodes = [
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="all_robots_bridge",
                arguments=bridge_arguments,
                remappings=[(f"/{robot['ns']}/tf", "/tf") for robot in robots],
                output="screen",
            )
        ]

        xacro_file = os.path.join(leo_description, "urdf", "leo_sim.urdf.xacro")
        for robot in robots:
            ns = robot["ns"]
            description = xacro.process_file(
                xacro_file, mappings={"robot_ns": ns}
            ).toxml().replace(
                "<frame_id>odom</frame_id>",
                f"<frame_id>{ns}/odom</frame_id>",
            )
            state_publisher = Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                namespace=ns,
                parameters=[{"use_sim_time": True, "robot_description": description}],
                remappings=[
                    ("joint_states", f"/{ns}/joint_states"),
                    ("tf", "/tf"),
                    ("tf_static", "/tf_static"),
                ],
                output="screen",
            )
            spawn = Node(
                package="ros_gz_sim",
                executable="create",
                namespace=ns,
                arguments=[
                    "-name",
                    ns,
                    "-x",
                    str(robot["x"]),
                    "-y",
                    str(robot["y"]),
                    "-z",
                    "0.1",
                    "-Y",
                    str(robot["yaw"]),
                    "-topic",
                    f"/{ns}/robot_description",
                ],
                output="screen",
            )
            spawn_tf = Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"{ns}_spawn_offset_tf",
                arguments=[
                    str(robot["x"]),
                    str(robot["y"]),
                    "0.0",
                    str(robot["yaw"]),
                    "0.0",
                    "0.0",
                    "odom",
                    f"{ns}/odom",
                ],
                output="screen",
            )

            supervisor_parameters = [
                {"motion_hold_duration": 0.2},
                {"supervisor_yaml_path": robot["yaml"]},
                {"random_seed": seed},
                {"run_id": run_id},
                {"results_dir": LaunchConfiguration("results_dir")},
                {"total_robots": total_robots},
                {
                    "forward_probability": ParameterValue(
                        LaunchConfiguration("forward_probability"), value_type=float
                    )
                },
            ]
            recovery_parameters = [
                {"enabled": True},
                {"timeout_s": 5.0},
                {"displacement_m": 0.10},
                {"escape_turn_rad": 3.0 * math.pi / 4.0},
                {"escape_angular_z": 1.0},
                {"reverse_linear_x": -0.20},
                {"reverse_duration_s": 0.8},
                {"forward_linear_x": 0.20},
                {"forward_duration_s": 0.8},
                {"cooldown_s": 3.0},
            ]
            if robot["uses_color_order"]:
                color_order = {"target_color_order": LaunchConfiguration("target_color_order")}
                supervisor_parameters.append(color_order)
                recovery_parameters.append(color_order)

            supervisor = Node(
                package=robot["package"],
                executable="robot_supervisor",
                name="robot_supervisor",
                namespace=ns,
                parameters=supervisor_parameters,
                remappings=[("cmd_vel", "cmd_vel_supervisor")],
                output="screen",
            )
            recovery = Node(
                package="leo_image_processing",
                executable="stuck_recovery",
                name="stuck_recovery",
                namespace=ns,
                parameters=recovery_parameters,
                output="screen",
            )
            image_processor = Node(
                package="leo_image_processing",
                executable="image_processor",
                name="image_processor",
                namespace=ns,
                parameters=[
                    {"use_sim_time": True},
                    {"depth_topic": f"/{ns}/depth_camera/depth_image"},
                    {"obstacle_threshold": 0.70},
                ],
                output="screen",
            )
            detector = Node(
                package="leo_image_processing",
                executable=robot["detector"],
                name=robot["detector"],
                namespace=ns,
                parameters=[
                    {"use_sim_time": True},
                    {"rgb_topic": f"/{ns}/depth_camera/image"},
                    {"robot_name": ns},
                    {"reached_distance": 1.0},
                ],
                output="screen",
            )
            rgb_bridge = Node(
                package="ros_gz_image",
                executable="image_bridge",
                name=f"{ns}_rgb_image_bridge",
                arguments=[f"/{ns}/depth_camera/image"],
                output="screen",
            )
            nodes.extend(
                [
                    state_publisher,
                    spawn,
                    spawn_tf,
                    rgb_bridge,
                    RegisterEventHandler(
                        OnProcessExit(
                            target_action=spawn,
                            on_exit=[
                                supervisor,
                                recovery,
                                TimerAction(
                                    period=2.0,
                                    actions=[image_processor, detector],
                                ),
                            ],
                        )
                    ),
                ]
            )
        return nodes

    return LaunchDescription(
        [
            SetEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", resource_path),
            SetEnvironmentVariable("PYTHONPATH", python_path),
            *arguments,
            gazebo,
            clock_bridge,
            OpaqueFunction(function=create_robots),
            TimerAction(
                period=LaunchConfiguration("run_duration"),
                actions=[Shutdown(reason="run_duration reached")],
            ),
            RegisterEventHandler(
                OnShutdown(
                    on_shutdown=[OpaqueFunction(function=_kill_simulator_processes)]
                )
            ),
        ]
    )
