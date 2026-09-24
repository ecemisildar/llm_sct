from glob import glob

from setuptools import find_packages, setup

package_name = 'leo_delivery'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test', 'evaluation', 'evaluation.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/balanced_delivery',
            glob('../automata/baseline_automata/balanced_delivery/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/worlds', glob('../worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ecem',
    maintainer_email='ecem@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_supervisor = leo_delivery.robot_supervisor:main',
            'complex_delivery_supervisor = leo_delivery.complex_delivery_supervisor:main',
            'balanced_delivery_supervisor = leo_delivery.balanced_delivery_supervisor:main',
        ],
    },
)
