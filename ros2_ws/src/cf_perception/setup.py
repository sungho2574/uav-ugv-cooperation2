from setuptools import find_packages, setup

package_name = 'cf_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'PyYAML', 'numpy', 'opencv-python'],
    zip_safe=True,
    maintainer='sungho',
    maintainer_email='sungho2574@gmail.com',
    description='Crazyflie telemetry relay + ArUco marker detection, sim and real variants',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_perception_node = cf_perception.sim_perception_node:main',
            'real_perception_node = cf_perception.real_perception_node:main',
        ],
    },
)
