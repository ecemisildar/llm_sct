import importlib.util
import os

from ament_index_python.packages import get_package_share_directory


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
        "leo_patrolling",
        enable_color_detector=True,
    )
