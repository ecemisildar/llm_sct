from glob import glob

from setuptools import find_packages, setup


package_name = "leo_real_experiments"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/BEST_YAMLS", glob("BEST_YAMLS/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ecem",
    maintainer_email="ecem@todo.todo",
    description="Image processing and SCT supervisors for LEO real-robot experiments",
    license="TODO: License declaration",
    entry_points={
        "console_scripts": [
            "image_processor = leo_real_experiments.image_processor:main",
            "color_detector = leo_real_experiments.color_detector:main",
            "delivery_shape_detector = leo_real_experiments.delivery_shape_detector:main",
            "stuck_recovery = leo_real_experiments.stuck_recovery:main",
            "exploration_supervisor = leo_real_experiments.exploration_supervisor:main",
            "patrolling_supervisor = leo_real_experiments.patrolling_supervisor:main",
            "delivery_supervisor = leo_real_experiments.delivery_supervisor:main",
        ],
    },
)
