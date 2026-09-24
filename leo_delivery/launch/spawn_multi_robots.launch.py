import importlib.util
import os

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    shared_path = os.path.join(
        get_package_share_directory("leo_gz_bringup"),
        "launch",
        "multi_robot_mission.py",
    )
    spec = importlib.util.spec_from_file_location("leo_multi_robot_mission", shared_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_multi_robot_launch(
        "leo_delivery",
        enable_color_detector=True,
        supervisor_executable=LaunchConfiguration("supervisor_executable"),
        evaluation_mission=LaunchConfiguration("evaluation_mission"),
        shutdown_on_task_complete=True,
        task_progress_on_complete=LaunchConfiguration("task_progress_on_complete"),
        target_detector_executable=LaunchConfiguration(
            "target_detector_executable"
        ),
    )
