from setuptools import setup

package_name = 'leo_image_processing'

setup(
    name=package_name,
    version='0.0.0',
    py_modules=['leo_image_processing', 'color_detector'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'image_processor = leo_image_processing:main',
            'color_detector = color_detector:main',
        ],
    },
)
