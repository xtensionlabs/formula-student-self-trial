#!/usr/bin/env python3
"""List the ROS topics visible right now with their publisher counts (the ros2 CLI is very slow here)."""
import _paths  # noqa: F401  (sets up sys.path)
import time

import rclpy


def main() -> None:
    rclpy.init()
    node = rclpy.create_node('list_topics')
    end = time.time() + 3.0
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    for name, types in sorted(node.get_topic_names_and_types()):
        print(f'{name:45s} pubs={node.count_publishers(name)} {types[0]}')
    print('nodes:', ', '.join(sorted(n for n, _ in node.get_node_names_and_namespaces())))
    rclpy.shutdown()


if __name__ == '__main__':
    main()
