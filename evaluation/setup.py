from setuptools import find_packages, setup


package_name = "evaluation"


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
    description="Shared evaluation nodes and offline analysis tools.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "coverage_counter = evaluation.coverage_counter:main",
            "bump_counter = evaluation.bump_counter:main",
            "analyze_run_from_logs = evaluation.analyze_run_from_logs:main",
            "aggregate_runs = evaluation.aggregate_runs:main",
        ],
    },
)
