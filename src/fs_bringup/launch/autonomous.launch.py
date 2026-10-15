"""Start the whole autonomous stack: ros2 launch fs_bringup autonomous.launch.py [rviz:=false]

Prerequisite (separate terminals): FSDS.exe on Windows and the FSDS ROS 2 bridge in WSL.
Node parameters live in fs_bringup/config/*.yaml (topic names there match the FSDS bridge).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# <ws>/models: this file is <ws>/src/fs_bringup/launch/... (symlink-install) -> resolve through realpath
WS = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory('fs_bringup')
    cfg = lambda name: os.path.join(share, 'config', name)
    models = os.path.join(WS, 'models')
    yolo = os.path.join(models, 'cones.pt')                    # fine-tuned YOLO weights (optional)
    ann = os.path.join(models, 'ann_controller.pt')

    def node(pkg: str, exe: str, yaml: str = '', **extra) -> Node:
        params = [cfg(yaml)] if yaml else []
        params += [extra.pop('params', {})]
        return Node(package=pkg, executable=exe, name=exe, output='screen', parameters=params, **extra)

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true', description='start RViz2'),
        node('fs_state_estimation', 'state_estimator_node', 'state_estimator.yaml'),
        # model_path '' = LiDAR-only (cone colours unknown) until cones.pt is trained
        node('fs_perception', 'cone_detector_node', 'perception.yaml',
             params={'model_path': yolo if os.path.exists(yolo) else ''}),
        node('fs_slam', 'slam_node', 'slam.yaml'),
        node('fs_slam', 'loop_closure_node', 'loop_closure.yaml'),
        node('fs_planning', 'path_planner_node', 'planner.yaml'),
        node('fs_control', 'ann_controller_node', 'ann.yaml', params={'model_path': ann}),
        node('fs_control', 'mpc_controller_node', 'mpc.yaml'),
        node('fs_bringup', 'viz_node'),
        Node(package='rviz2', executable='rviz2', output='log', condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', os.path.join(share, 'rviz', 'autonomous.rviz')]),
    ])
