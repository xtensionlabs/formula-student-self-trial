import glob

from setuptools import setup, find_packages

# Launch, config and RViz files are installed so `ros2 launch fs_bringup autonomous.launch.py` finds them.
setup(
    name='fs_bringup', version='0.1.0', packages=find_packages(exclude=['test']),
    data_files=[('share/ament_index/resource_index/packages', ['resource/fs_bringup']),
                ('share/fs_bringup', ['package.xml']),
                ('share/fs_bringup/launch', glob.glob('launch/*.py')),
                ('share/fs_bringup/config', glob.glob('config/*.yaml')),
                ('share/fs_bringup/rviz', glob.glob('rviz/*.rviz'))],
    install_requires=['setuptools'], zip_safe=True,
    entry_points={'console_scripts': ['viz_node = fs_bringup.viz_node:main']},
)
