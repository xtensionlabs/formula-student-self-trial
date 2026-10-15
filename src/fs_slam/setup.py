from setuptools import setup, find_packages

# Plain ament_python package; console_scripts become 'ros2 run fs_slam <name>'.
setup(
    name='fs_slam', version='0.1.0', packages=find_packages(exclude=['test']),
    data_files=[('share/ament_index/resource_index/packages', ['resource/fs_slam']),
                ('share/fs_slam', ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    entry_points={'console_scripts': ['slam_node = fs_slam.slam_node:main', 'loop_closure_node = fs_slam.loop_closure_node:main', ]},
)
