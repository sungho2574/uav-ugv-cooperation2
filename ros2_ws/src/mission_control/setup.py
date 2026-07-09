from setuptools import find_packages, setup

package_name = 'mission_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'shapely', 'PyYAML', 'numpy'],
    zip_safe=True,
    maintainer='sungho',
    maintainer_email='sungho2574@gmail.com',
    description='Central mission state machine: zone decomposition, coverage planning, and crazyswarm2 command dispatch',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_node = mission_control.control_node:main',
        ],
    },
)
