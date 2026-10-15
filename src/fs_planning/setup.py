from setuptools import setup, find_packages

# Plain ament_python package; console_scripts become 'ros2 run fs_planning <name>'.
setup(
    name='fs_planning', version='0.1.0', packages=find_packages(exclude=['test']),
    data_files=[('share/ament_index/resource_index/packages', ['resource/fs_planning']),
                ('share/fs_planning', ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    entry_points={'console_scripts': ['path_planner_node = fs_planning.path_planner_node:main', ]},
)
