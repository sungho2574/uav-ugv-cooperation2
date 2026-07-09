from glob import glob

from setuptools import find_packages, setup

package_name = 'gcs_dashboard'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/templates', glob('gcs_dashboard/templates/*')),
        ('share/' + package_name + '/static', glob('gcs_dashboard/static/*')),
    ],
    install_requires=['setuptools', 'flask', 'numpy', 'opencv-python', 'PyYAML'],
    zip_safe=True,
    maintainer='sungho',
    maintainer_email='sungho2574@gmail.com',
    description='Flask + Three.js ground control station dashboard',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gcs_node = gcs_dashboard.gcs_node:main',
        ],
    },
)
