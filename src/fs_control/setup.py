from setuptools import setup, find_packages

# Plain ament_python package; console_scripts become 'ros2 run fs_control <name>'.
setup(
    name='fs_control', version='0.1.0', packages=find_packages(exclude=['test']),
    data_files=[('share/ament_index/resource_index/packages', ['resource/fs_control']),
                ('share/fs_control', ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    entry_points={'console_scripts': ['mpc_controller_node = fs_control.mpc_controller_node:main', 'ann_controller_node = fs_control.ann_controller_node:main', ]},
)
