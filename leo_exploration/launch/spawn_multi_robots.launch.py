import math
import os
import random
import time
import xml.etree.ElementTree as ET

import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
    TimerAction,
)
from launch.event_handlers import OnShutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _parse_pose(pose_text: str | None):
    if not pose_text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    vals = [float(v) for v in pose_text.split()]
    while len(vals) < 6:
        vals.append(0.0)
    return tuple(vals[:6])


def _load_world_obstacles(world_path: str):
    root = ET.parse(world_path).getroot()
    world = root.find("world")
    if world is None:
        return []

    obstacles = []
    for model in world.findall("model"):
        name = model.get("name", "")
        if name == "ground_plane":
            continue
        model_pose = _parse_pose(model.findtext("pose"))
        for link in model.findall("link"):
            link_pose = _parse_pose(link.findtext("pose"))
            for collision in link.findall("collision"):
                collision_pose = _parse_pose(collision.findtext("pose"))
                cx = model_pose[0] + link_pose[0] + collision_pose[0]
                cy = model_pose[1] + link_pose[1] + collision_pose[1]
                yaw = model_pose[5] + link_pose[5] + collision_pose[5]

                box_size = collision.findtext("geometry/box/size")
                cyl_radius = collision.findtext("geometry/cylinder/radius")
                if box_size:
                    sx, sy, _ = (float(v) for v in box_size.split())
                    obstacles.append(("box", cx, cy, sx, sy, yaw))
                elif cyl_radius:
                    obstacles.append(("cylinder", cx, cy, float(cyl_radius)))
    return obstacles


def _point_in_rotated_box(px: float, py: float, cx: float, cy: float, sx: float, sy: float, yaw: float, margin: float):
    dx = px - cx
    dy = py - cy
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    lx = dx * c - dy * s
    ly = dx * s + dy * c
    return abs(lx) <= (sx * 0.5 + margin) and abs(ly) <= (sy * 0.5 + margin)


def _point_is_free(px: float, py: float, obstacles, wall_margin: float, obstacle_margin: float):
    if abs(px) > (5.0 - wall_margin) or abs(py) > (5.0 - wall_margin):
        return False
    for obstacle in obstacles:
        if obstacle[0] == "box":
            _, cx, cy, sx, sy, yaw = obstacle
            if _point_in_rotated_box(px, py, cx, cy, sx, sy, yaw, obstacle_margin):
                return False
        else:
            _, cx, cy, radius = obstacle
            if math.hypot(px - cx, py - cy) <= (radius + obstacle_margin):
                return False
    return True


def _build_robot_spawn_slots(total_robots: int, world_path: str):
    obstacles = _load_world_obstacles(world_path)
    candidate_x = [-3.25, -1.75, -0.25, 1.25, 2.75, 3.5]
    candidate_y = [-3.25, -1.75, -0.25, 1.25, 2.75, 3.5]
    min_robot_spacing = 1.25
    wall_margin = 0.6
    obstacle_margin = 0.55

    candidates = []
    for y in candidate_y:
        for x in candidate_x:
            if _point_is_free(x, y, obstacles, wall_margin, obstacle_margin):
                # Prefer far-apart outer slots first so robots start dispersed.
                candidates.append((x, y))

    if not candidates:
        raise RuntimeError(f"No safe spawn slots found in {world_path}.")

    robots = []
    # Greedy farthest-point sampling:
    # 1) start with the outermost free slot
    # 2) repeatedly add the slot with the largest distance to the current set
    first_x, first_y = max(candidates, key=lambda p: abs(p[0]) + abs(p[1]))
    robots.append({"x": first_x, "y": first_y, "yaw": 0.0})

    remaining = [p for p in candidates if p != (first_x, first_y)]
    while remaining and len(robots) < total_robots:
        feasible = []
        for x, y in remaining:
            distances = [math.hypot(x - robot["x"], y - robot["y"]) for robot in robots]
            min_dist = min(distances)
            if min_dist >= min_robot_spacing:
                feasible.append((min_dist, abs(x) + abs(y), x, y))
        if not feasible:
            break
        # Prefer the point with the largest minimum distance to existing robots.
        # Break ties toward outer slots.
        _, _, x, y = max(feasible, key=lambda item: (item[0], item[1]))
        robots.append({"x": x, "y": y, "yaw": 0.0})
        remaining = [p for p in remaining if p != (x, y)]

    if len(robots) < total_robots:
        raise RuntimeError(
            f"Only found {len(robots)} safe spawn slots in {world_path}, need {total_robots}."
        )
    return robots


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


def generate_launch_description():

    leo_description = get_package_share_directory("leo_description")
    run_id = time.strftime("run_%Y%m%d_%H%M%S")

    auto_start_supervisor = LaunchConfiguration("auto_start_supervisor")
    auto_start_supervisor_arg = DeclareLaunchArgument(
        "auto_start_supervisor",
        default_value="true",
        description="Enable robot_supervisor_3_movements on launch",
    )
    random_seed_arg = DeclareLaunchArgument(
        "random_seed",
        default_value="auto",
        description="Base seed for per-robot supervisor random choices. Use 'auto' for a fresh seed each run.",
    )

    plot_node = Node(
            package="leo_exploration",
            executable="coverage_counter",
            name="coverage_counter",
            parameters=[
                {"run_id": run_id},
                {
                    "run_duration": ParameterValue(
                        LaunchConfiguration("run_duration"), value_type=float
                    )
                },
                {"results_dir": LaunchConfiguration("results_dir")},
                {"metadata_yaml_path": LaunchConfiguration("metadata_yaml_path")},
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
                        
                        "auto_start_supervisor",
                        "results_dir",
                        "metadata_yaml_path",
                    )
                },
            ],
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
                f"/{ns}/depth_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image",
                f"/{ns}/depth_camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
                f"/{ns}/depth_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
                f"/world/random_world/model/{ns}/link/{ns}/base_footprint/sensor/contact_sensor/contact"
                f"@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts",
            ]

        bridge_args += [
            "/world/random_world/dynamic_pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
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

            behavior_node = Node(
                package="leo_exploration",
                executable="robot_supervisor",
                name="robot_supervisor",
                namespace=ns,
                parameters=[
                    {"enabled": auto_start_supervisor},
                    {"motion_hold_duration": 1.0},
                    {"supervisor_yaml_path": LaunchConfiguration("metadata_yaml_path")},
                    {"random_seed": random_seed},
                    {"run_id": run_id},
                    {"results_dir": LaunchConfiguration("results_dir")},
                    {"total_robots": LaunchConfiguration("total_robots")},
                ],
                # output="screen",
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

            nodes += [
                state_pub,
                spawn_node,
                spawn_offset_tf,
                behavior_node,
                TimerAction(
                    period=2.0,
                    actions=[image_processor_node],
                ),
            ]

        return nodes

    return LaunchDescription([
        auto_start_supervisor_arg,
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
        TimerAction(
            period=LaunchConfiguration("run_duration"),
            actions=[Shutdown(reason="run_duration reached")],
        ),
        plot_node,
        Node(
            package="leo_exploration",
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
        OpaqueFunction(function=create_all_robot_nodes)
    ])
