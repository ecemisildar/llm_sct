from setuptools import find_packages, setup


package_name = "leo_supervisor_common"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ecem",
    maintainer_email="ecem@todo.todo",
    description="Shared utilities for LEO task supervisors",
    license="TODO: License declaration",
)
