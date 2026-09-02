import math
import os
import random
import sys
import time
import xml.etree.ElementTree as ET

import xacro

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    Shutdown,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue



def _pose_xy(element):
    pose = element.find("pose")
    if pose is None or not pose.text:
        return 0.0, 0.0
    values = pose.text.split()
    return (
        float(values[0]) if len(values) > 0 else 0.0,
        float(values[1]) if len(values) > 1 else 0.0,
    )


def _world_obstacle_circles(world_path: str):
    """Approximate static collision geometry with conservative XY circles."""
    try:
        root = ET.parse(world_path).getroot()
    except (ET.ParseError, OSError, ValueError):
        return []

    obstacles = []
    for model in root.findall(".//world/model"):
        name = model.get("name", "")
        if name in {"ground_plane", "north_wall", "south_wall", "east_wall", "west_wall"}:
            continue
        model_x, model_y = _pose_xy(model)
        radius = 0.0
        for collision in model.findall(".//collision"):
            local_x, local_y = _pose_xy(collision)
            geometry = collision.find("geometry")
            if geometry is None:
                continue
            box_size = geometry.findtext("box/size")
            cylinder_radius = geometry.findtext("cylinder/radius")
            if box_size:
                values = [float(value) for value in box_size.split()]
                if len(values) >= 2:
                    geometry_radius = math.hypot(values[0], values[1]) / 2.0
                else:
                    continue
            elif cylinder_radius:
                geometry_radius = float(cylinder_radius)
            else:
                continue
            radius = max(
                radius,
                math.hypot(local_x, local_y) + geometry_radius,
            )
        if radius > 0.0:
            obstacles.append((model_x, model_y, radius))
    return obstacles


def _build_robot_spawn_slots(
    total_robots: int,
    world_path: str,
    rng: random.Random,
):
    minimum_spacing = 1.0
    obstacle_clearance = 0.55
    spawn_limit = 4.35
    obstacles = _world_obstacle_circles(world_path)
    slots = []

    for _ in range(10000):
        if len(slots) == total_robots:
            return slots
        x = rng.uniform(-spawn_limit, spawn_limit)
        y = rng.uniform(-spawn_limit, spawn_limit)
        if any(
            math.hypot(x - slot["x"], y - slot["y"]) < minimum_spacing
            for slot in slots
        ):
            continue
        if any(
            math.hypot(x - obstacle_x, y - obstacle_y)
            < obstacle_radius + obstacle_clearance
            for obstacle_x, obstacle_y, obstacle_radius in obstacles
        ):
            continue
        slots.append(
            {
                "x": x,
                "y": y,
                "yaw": rng.uniform(-math.pi, math.pi),
            }
        )

    raise RuntimeError(
        f"Could not find {total_robots} random collision-free spawn positions "
        f"in {world_path}."
    )


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
    supervisor_executable="robot_supervisor",
    evaluation_mission=None,
    enable_color_order: bool = False,
    shutdown_on_task_complete: bool = False,
    wait_for_all_task_completion: bool = False,
    task_progress_on_complete: int = 1,
):

    leo_description = get_package_share_directory("leo_description")
    run_id = f'{time.strftime("run_%Y%m%d_%H%M%S")}_{time.time_ns() % 1_000_000_000:09d}'
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
        default_value="12345",
        description="Base seed for reproducible per-robot choices; use 'auto' for a fresh seed.",
    )
    forward_probability_arg = DeclareLaunchArgument(
        "forward_probability",
        default_value="0.90",
        description="Probability of choosing forward when multiple motion requests are enabled.",
    )
    record_video_arg = DeclareLaunchArgument(
        "record_video",
        default_value="false",
        description="Record the overhead camera to an MP4 in the run directory.",
    )
    object_cluster_half_width_arg = DeclareLaunchArgument(
        "object_cluster_half_width",
        default_value="0.125",
        description="Half-width in metres of a delivery pickup cluster.",
    )

    evaluation_parameters = [
        {"run_id": run_id},
        {"mission": evaluation_mission or mission_package.removeprefix("leo_")},
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
                {
                    "task_progress_on_complete": ParameterValue(
                        task_progress_on_complete, value_type=int
                    )
                },
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
                "target_color_order": ParameterValue(
                    LaunchConfiguration("target_color_order"), value_type=str
                ),
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
        spawn_rng = random.Random(random_seed ^ 0x5A17C0DE)
        base_slots = _build_robot_spawn_slots(
            total_robots,
            world_path,
            spawn_rng,
        )
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
            "/world/random_world/create@ros_gz_interfaces/srv/SpawnEntity",
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
                {"motion_hold_duration": 0.2},
                {"supervisor_yaml_path": LaunchConfiguration("metadata_yaml_path")},
                {"random_seed": random_seed},
                {"run_id": run_id},
                {"results_dir": LaunchConfiguration("results_dir")},
                {"total_robots": LaunchConfiguration("total_robots")},
                {
                    "forward_probability": ParameterValue(
                        LaunchConfiguration("forward_probability"),
                        value_type=float,
                    )
                },
            ]
            if shutdown_on_task_complete:
                # The evaluation node saves the final completion timestamp and
                # exits first. Its OnProcessExit handler then shuts down Gazebo.
                # Keep the final robot stopped in place until that shutdown.
                behavior_parameters.append({"remove_completed_robot": False})
            if enable_color_order:
                behavior_parameters.append(
                    {"target_color_order": LaunchConfiguration("target_color_order")}
                )

            behavior_node = Node(
                package=mission_package,
                executable=supervisor_executable,
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

            stuck_recovery_parameters = [
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
            ]
            if enable_color_order:
                stuck_recovery_parameters.append(
                    {"target_color_order": LaunchConfiguration("target_color_order")}
                )

            stuck_recovery_node = Node(
                package="leo_image_processing",
                executable="stuck_recovery",
                name="stuck_recovery",
                namespace=ns,
                parameters=stuck_recovery_parameters,
                output="screen",
            )

            delayed_nodes = [image_processor_node]
            if enable_color_detector:
                delayed_nodes.append(
                    Node(
                        package="leo_image_processing",
                        executable=(
                            "delivery_shape_detector"
                            if mission_package == "leo_delivery"
                            else "color_detector"
                        ),
                        name=(
                            "delivery_shape_detector"
                            if mission_package == "leo_delivery"
                            else "color_detector"
                        ),
                        namespace=ns,
                        parameters=[
                            {"use_sim_time": True},
                            {"rgb_topic": f"/{ns}/depth_camera/image"},
                            {"robot_name": ns},
                            {"reached_distance": 1.0},
                            *(
                                [
                                    {
                                        "object_cluster_half_width":
                                        ParameterValue(
                                            LaunchConfiguration(
                                                "object_cluster_half_width"
                                            ),
                                            value_type=float,
                                        )
                                    }
                                ]
                                if mission_package == "leo_delivery"
                                else []
                            ),
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
        forward_probability_arg,
        record_video_arg,
        object_cluster_half_width_arg,
    ]
    if shutdown_on_task_complete:
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=plot_node,
                    on_exit=[
                        Shutdown(
                            reason=(
                                "task completion time saved; "
                                "shutting down Gazebo"
                            )
                        )
                    ],
                )
            )
        )
    if not wait_for_all_task_completion and not shutdown_on_task_complete:
        actions.append(
            TimerAction(
                period=LaunchConfiguration("run_duration"),
                actions=[Shutdown(reason="run_duration reached")],
            )
        )
    actions.extend([
        Node(
            package="ros_gz_image",
            executable="image_bridge",
            name="top_view_image_bridge",
            arguments=["/top_view_camera/image"],
            output="screen",
        ),
        Node(
            package="evaluation",
            executable="simulation_video_recorder",
            name="simulation_video_recorder",
            parameters=[
                {"run_id": run_id},
                {"results_dir": LaunchConfiguration("results_dir")},
                {"metadata_yaml_path": LaunchConfiguration("metadata_yaml_path")},
                {"total_robots": LaunchConfiguration("total_robots")},
                {"image_topic": "/top_view_camera/image"},
                {"fps": 15.0},
                {
                    "enabled": ParameterValue(
                        LaunchConfiguration("record_video"), value_type=bool
                    )
                },
            ],
            output="screen",
        ),
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
                {"save_depth_pre_collision_images": False},
                {"depth_history_frames": 5},
                {"depth_topic_template": "/{robot}/depth_camera/depth_image"},
            ],
            output="screen",
        ),
        OpaqueFunction(function=create_all_robot_nodes),
    ])
    return LaunchDescription(actions)
