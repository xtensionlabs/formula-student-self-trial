"""Tiny shared helper: every node reports its processing latency on /diagnostics."""
import time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node


class LatencyDiag:
    """Usage: t = diag.start(); ...work...; diag.stop(t)  (publishes one status)."""

    def __init__(self, node: Node) -> None:
        self._node = node
        self._pub = node.create_publisher(DiagnosticArray, '/diagnostics', 10)

    def start(self) -> float:
        return time.perf_counter()

    def stop(self, t0: float) -> float:
        ms = (time.perf_counter() - t0) * 1e3
        msg = DiagnosticArray()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        st = DiagnosticStatus(name=self._node.get_name(), level=DiagnosticStatus.OK)
        st.values = [KeyValue(key='latency_ms', value=f'{ms:.2f}')]
        msg.status = [st]
        self._pub.publish(msg)
        return ms
