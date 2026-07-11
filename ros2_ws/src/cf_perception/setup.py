import os
from glob import glob

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
        # Without this, config/camera_intrinsics.yaml never lands in
        # install/cf_perception/share/cf_perception/config/, and
        # real_perception_node's camera_intrinsics_path (set by
        # real.launch.py to that installed path) fails with FileNotFoundError.
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Same reasoning for the YOLO .onnx weights -- real.launch.py resolves
        # the default yolo_weights_path from this installed share/ location
        # (see DEFAULT_YOLO_WEIGHTS_FILENAME there), not the source tree.
        (os.path.join('share', package_name, 'weights'), glob('cf_perception/weights/*.onnx')),
    ],
    # `ultralytics` runs the YOLO .onnx graph with the same pre/postprocessing
    # it was trained/exported with (see yolo_detector.py) -- cv2.dnn was tried
    # first but can't import YOLO11's C2PSA attention blocks, and a hand-rolled
    # onnxruntime postprocessing pass after that risked its own format-
    # assumption bugs.
    install_requires=['setuptools', 'PyYAML', 'numpy', 'opencv-python', 'ultralytics'],
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
