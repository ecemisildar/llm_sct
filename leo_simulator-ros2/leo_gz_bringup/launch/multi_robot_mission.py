import math
import os
import random
import sys
import time

import xacro

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    Shutdown,
    TimerAction,
)
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue



def _build_robot_spawn_slots(total_robots: int, world_path: str):
    del world_path
    if total_robots == 1:
        return [{"x": 0.0, "y": 0.0, "yaw": 0.0}]

    minimum_spacing = 0.8
    radius = max(
        0.75,
        minimum_spacing / (2.0 * math.sin(math.pi / total_robots)),
    )
    if radius > 3.5:
        raise RuntimeError(
            f"Cannot place {total_robots} robots on the central ring with "
            f"{minimum_spacing:.1f} m spacing inside the world."
        )

    return [
        {
            "x": radius * math.cos(angle),
            "y": radius * math.sin(angle),
            "yaw": angle,
        }
        for angle in (
            2.0 * math.pi * index / total_robots
            for index in range(total_robots)
        )
    ]


def _resolve_seed(value: str, name: str) -> int:
    text = str(value).strip().lower()
    if text in {"", "auto", "random"}:
        return random.SystemRandom().randint(1, 2_147_483_647)
    try:
        return int(text)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid {name} '{value}'. Use an integer seed or 'auto'."
        ) from exc


def generate_multi_robot_launch(
    mission_package: str,
    *,
    enable_color_detector: bool,
    enable_color_order: bool = False,
    shutdown_on_task_complete: bool = False,
    wait_for_all_task_completion: bool = False,
    task_progress_on_complete: int = 1,
):

    leo_description = get_package_share_directory("leo_description")
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
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

    random_seed_arg = DeclareLaunchArgument(
        "random_seed",
        default_value="auto",
        description="Base seed for per-robot supervisor random choices. Use 'auto' for a fresh seed each run.",
    )

    evaluation_parameters = [
        {"run_id": run_id},
        {
            "run_duration": ParameterValue(
                LaunchConfiguration("run_duration"), value_type=float
            )
        },
        {"results_dir": LaunchConfiguration("results_dir")},
        {"metadata_yaml_path": LaunchConfiguration("metadata_yaml_path")},
    ]
    if shutdown_on_task_complete:
        evaluation_parameters.extend(
            [
                {"shutdown_on_task_complete": True},
                {"task_progress_on_complete": task_progress_on_complete},
                {
                    "wait_for_all_task_completion":
                    wait_for_all_task_completion
                },
            ]
        )
    evaluation_parameters.append(
        {
            f"launch_{name}": ParameterValue(
                LaunchConfiguration(name), value_type=str
            )
            for name in (
                "sim_world",
                "headless",
                "auto_start",
                "run_duration",
                "total_robots",
                "random_seed",
                "results_dir",
                "metadata_yaml_path",
            )
        }
    )
    if enable_color_order:
        evaluation_parameters.append(
            {
                "launch_target_color_order": ParameterValue(
                    LaunchConfiguration("target_color_order"), value_type=str
                )
            }
        )

    plot_node = Node(
            package="evaluation",
            executable="coverage_counter",
            name="coverage_counter",
            parameters=evaluation_parameters,
            output="screen"
    )
 

    # --- Function to create all robot nodes ---
    def create_all_robot_nodes(context):
        total_robots_value = int(LaunchConfiguration("total_robots").perform(context))
        total_robots = max(1, total_robots_value)
        world_path = LaunchConfiguration("sim_world").perform(context)
        random_seed = _resolve_seed(LaunchConfiguration("random_seed").perform(context), "random_seed")

        print(
            f"[spawn_multi_robots] run_id={run_id} random_seed={random_seed} ",
            flush=True,
        )
        base_slots = _build_robot_spawn_slots(total_robots, world_path)
        robots = []
        for i, slot in enumerate(base_slots):
            robots.append({
                "ns": f"robot_{i}",
                "x": slot["x"],
                "y": slot["y"],
                "yaw": slot["yaw"],
            })

        nodes = []

        # --- One bridge for all robots ---
        bridge_args = []
        for robot in robots:
            ns = robot["ns"]
            bridge_args += [
                f"/{ns}/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
                f"/{ns}/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
                f"/{ns}/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
                f"/{ns}/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model",
                f"/{ns}/depth_camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
                f"/{ns}/depth_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
                f"/world/random_world/model/{ns}/link/{ns}/base_footprint/sensor/contact_sensor/contact"
                f"@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts",
            ]

        bridge_args += [
            "/world/random_world/dynamic_pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
            "/world/random_world/remove@ros_gz_interfaces/srv/DeleteEntity",
            "/world/random_world/set_pose@ros_gz_interfaces/srv/SetEntityPose",
        ]

        bridge_node = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="all_robots_bridge",
            arguments=bridge_args,
            parameters=[{"qos_overrides./tf_static.publisher.durability": "transient_local"}],
            remappings=[
                *[(f"/{robot['ns']}/tf", "/tf") for robot in robots],
            ],
            output="screen"
        )
        nodes.append(bridge_node)

        # --- Create each robot ---
        for robot in robots:
            ns = robot["ns"]
            x = robot["x"]
            y = robot["y"]
            yaw = robot["yaw"]

            # URDF with per-robot namespace mapping
            xacro_file = os.path.join(leo_description, 'urdf', 'leo_sim.urdf.xacro')
            doc = xacro.process_file(xacro_file, mappings={"robot_ns": ns})
            robot_description = doc.toxml().replace(
                "<frame_id>odom</frame_id>",
                f"<frame_id>{ns}/odom</frame_id>",
            )

            # State publisher (per robot)
            state_pub = Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                namespace=ns,
                parameters=[{
                    "use_sim_time": True,
                    "robot_description": robot_description
                }],
                remappings=[
                    ("joint_states", f"/{ns}/joint_states"),
                    ("tf", "/tf"),
                    ("tf_static", "/tf_static"),
                ],
                output="screen"
            )

            # Spawn robot in Gazebo
            spawn_node = Node(
                package="ros_gz_sim",
                executable="create",
                namespace=ns,
                arguments=[
                    "-name", ns,
                    "-x", str(x),
                    "-y", str(y),
                    "-z", "0.1",
                    "-Y", str(yaw),
                    "-topic", f"/{ns}/robot_description"
                ],
                output="screen"
            )

            spawn_offset_tf = Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"{ns}_spawn_offset_tf",
                arguments=[
                    str(x),
                    str(y),
                    "0.0",
                    str(yaw),
                    "0.0",
                    "0.0",
                    "odom",
                    f"{ns}/odom",
                ],
                output="screen",
            )

            behavior_parameters = [
                {"motion_hold_duration": 1.0},
                {"supervisor_yaml_path": LaunchConfiguration("metadata_yaml_path")},
                {"random_seed": random_seed},
                {"run_id": run_id},
                {"results_dir": LaunchConfiguration("results_dir")},
                {"total_robots": LaunchConfiguration("total_robots")},
            ]
            if enable_color_order:
                behavior_parameters.append(
                    {"target_color_order": LaunchConfiguration("target_color_order")}
                )

            behavior_node = Node(
                package=mission_package,
                executable="robot_supervisor",
                name="robot_supervisor",
                namespace=ns,
                parameters=behavior_parameters,
                remappings=[("cmd_vel", "cmd_vel_supervisor")],
                output="screen",
            )

            image_processor_node = Node(
                package="leo_image_processing",
                executable="image_processor",
                name="image_processor",
                namespace=ns,
                parameters=[
                    {"use_sim_time": True},
                    {"depth_topic": f"/{ns}/depth_camera/depth_image"},
                    {"obstacle_threshold": 0.70},
                ],
                output="screen"
            )

            rgb_image_bridge = Node(
                package="ros_gz_image",
                executable="image_bridge",
                name=f"{ns}_rgb_image_bridge",
                arguments=[f"/{ns}/depth_camera/image"],
                output="screen",
            )

            stuck_recovery_node = Node(
                package="leo_image_processing",
                executable="stuck_recovery",
                name="stuck_recovery",
                namespace=ns,
                parameters=[
                    {"enabled": True},
                    {"timeout_s": 5.0},
                    {"displacement_m": 0.10},
                    {"escape_turn_rad": 2.356194490192345},
                    {"escape_angular_z": 1.0},
                    {"reverse_linear_x": -0.20},
                    {"reverse_duration_s": 0.8},
                    {"forward_linear_x": 0.20},
                    {"forward_duration_s": 0.8},
                    {"cooldown_s": 3.0},
                ],
                output="screen",
            )

            delayed_nodes = [image_processor_node]
            if enable_color_detector:
                delayed_nodes.append(
                    Node(
                        package="leo_image_processing",
                        executable="color_detector",
                        name="color_detector",
                        namespace=ns,
                        parameters=[
                            {"use_sim_time": True},
                            {"rgb_topic": f"/{ns}/depth_camera/image"},
                            {"robot_name": ns},
                            {"reached_distance": 1.0},
                        ],
                        output="screen",
                    )
                )

            nodes += [
                state_pub,
                spawn_node,
                spawn_offset_tf,
                rgb_image_bridge,
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=spawn_node,
                        on_exit=[
                            behavior_node,
                            stuck_recovery_node,
                            TimerAction(
                                period=2.0,
                                actions=delayed_nodes,
                            ),
                        ],
                    )
                ),
            ]

        return nodes

    actions = [
        SetEnvironmentVariable(name="PYTHONPATH", value=python_path),
        random_seed_arg,
        RegisterEventHandler(
            OnShutdown(
                on_shutdown=[
                    ExecuteProcess(
                        cmd=["bash", "-lc", "pkill -f parameter_bridge || true"],
                        output="screen",
                    )
                ]
            )
        ),
    ]
    if shutdown_on_task_complete:
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=plot_node,
                    on_exit=[Shutdown(reason="all robots reached a goal")],
                )
            )
        )
    if not wait_for_all_task_completion:
        actions.append(
            TimerAction(
                period=LaunchConfiguration("run_duration"),
                actions=[Shutdown(reason="run_duration reached")],
            )
        )
    actions.extend([
        plot_node,
        Node(
            package="evaluation",
            executable="bump_counter",
            name="bump_counter",
            parameters=[
                {"global_mode": True},
                {"run_id": run_id},
                {"results_dir": LaunchConfiguration("results_dir")},
                {"metadata_yaml_path": LaunchConfiguration("metadata_yaml_path")},
                {"total_robots": LaunchConfiguration("total_robots")},
                {"save_depth_pre_collision_images": True},
                {"depth_history_frames": 5},
                {"depth_topic_template": "/{robot}/depth_camera/depth_image"},
            ],
            output="screen",
        ),
        OpaqueFunction(function=create_all_robot_nodes),
    ])
    return LaunchDescription(actions)
