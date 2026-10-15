"""viz_node: RViz helpers - coloured cone markers and speed arrows along the centerline."""
from typing import List, Optional
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from fs_perception.msg import ConeArray, SpeedProfile

# Cone.color -> RGB: 0 unknown, 1 yellow, 2 blue, 3 orange
COLORS = {0: (0.6, 0.6, 0.6), 1: (1.0, 0.9, 0.0), 2: (0.1, 0.3, 1.0), 3: (1.0, 0.5, 0.0)}


def cone_markers(m: ConeArray, ns: str, size: float) -> MarkerArray:
    """One SPHERE_LIST marker per colour (cheap to render, one colour each)."""
    out = MarkerArray()
    for col, rgb in COLORS.items():
        mk = Marker()
        mk.header, mk.ns, mk.id = m.header, ns, col
        mk.type, mk.action = Marker.SPHERE_LIST, Marker.ADD
        mk.scale.x = mk.scale.y = mk.scale.z = size
        mk.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=1.0)
        mk.points = [Point(x=c.x, y=c.y, z=c.z + size / 2) for c in m.cones if c.color == col]
        out.markers.append(mk)
    return out


class Viz(Node):
    def __init__(self) -> None:
        super().__init__('viz_node')
        p = self.declare_parameter
        p('arrow_every', 2.0); p('v_max', 12.0)
        self.path: Optional[Path] = None
        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pubs = {n: self.create_publisher(MarkerArray, f'/viz/{n}', 1) for n in ('perception', 'slam_map', 'corrected_map', 'velocity')}
        self.create_subscription(ConeArray, '/perception/cones', lambda m: self.pubs['perception'].publish(cone_markers(m, 'perception', 0.25)), 5)
        self.create_subscription(ConeArray, '/slam/map', lambda m: self.pubs['slam_map'].publish(cone_markers(m, 'slam', 0.35)), 5)
        self.create_subscription(ConeArray, '/slam/map_corrected', lambda m: self.pubs['corrected_map'].publish(cone_markers(m, 'corrected', 0.35)), qos)
        self.create_subscription(Path, '/planning/centerline', lambda m: setattr(self, 'path', m), 5)
        self.create_subscription(SpeedProfile, '/planning/velocity_profile', self.on_profile, 5)

    def on_profile(self, m: SpeedProfile) -> None:
        """Arrows along the path: length and colour (red slow -> green fast) show the target speed."""
        if self.path is None or not self.path.poses:
            return
        g = lambda n: self.get_parameter(n).value
        pts = np.array([[q.pose.position.x, q.pose.position.y] for q in self.path.poses])
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        out, k = MarkerArray(), 0
        out.markers.append(Marker(action=Marker.DELETEALL))
        for target in np.arange(0.0, s[-1], g('arrow_every')):
            i = int(np.searchsorted(s, target))
            if i >= len(pts) - 1:
                break
            v = float(np.interp(s[i], m.s, m.speed))
            d = pts[i + 1] - pts[i]
            d /= max(np.hypot(*d), 1e-9)
            f = float(np.clip(v / g('v_max'), 0, 1))
            mk = Marker()
            mk.header, mk.ns, mk.id = self.path.header, 'speed', k
            mk.type, mk.action = Marker.ARROW, Marker.ADD
            mk.points = [Point(x=float(pts[i, 0]), y=float(pts[i, 1]), z=0.1),
                         Point(x=float(pts[i, 0] + d[0] * v * 0.25), y=float(pts[i, 1] + d[1] * v * 0.25), z=0.1)]
            mk.scale.x, mk.scale.y, mk.scale.z = 0.12, 0.25, 0.25
            mk.color = ColorRGBA(r=1.0 - f, g=f, b=0.1, a=0.9)
            out.markers.append(mk)
            k += 1
        self.pubs['velocity'].publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Viz()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
