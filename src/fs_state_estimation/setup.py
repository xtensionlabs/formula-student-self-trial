from setuptools import setup, find_packages

# Plain ament_python package; console_scripts become 'ros2 run fs_state_estimation <name>'.
setup(
    name='fs_state_estimation', version='0.1.0', packages=find_packages(exclude=['test']),
    data_files=[('share/ament_index/resource_index/packages', ['resource/fs_state_estimation']),
                ('share/fs_state_estimation', ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    entry_points={'console_scripts': ['state_estimator_node = fs_state_estimation.state_estimator_node:main', ]},
)
