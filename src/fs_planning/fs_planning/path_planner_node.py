"""path_planner_node: cone map -> /planning/centerline (Path) + /planning/velocity_profile.

Re-plans at `plan_hz` from the latest map (the loop-closure corrected map wins once it exists).
After every completed lap the speed GP is refined from what the car actually did.
"""
import os
from typing import Optional
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from fs_perception.msg import ConeArray, SpeedProfile
from fs_perception_py.diag import LatencyDiag
from fs_planning.centerline import CenterlineParams, build_centerline, sample
from fs_planning.speed_gp import SpeedGP, limit_accel


class PathPlanner(Node):
    def __init__(self) -> None:
        super().__init__('path_planner_node')
        p = self.declare_parameter
        p('map_topic', '/slam/map'); p('corrected_topic', '/slam/map_corrected')
        p('pose_topic', '/slam/pose'); p('odom_topic', '/state/odometry')
        p('plan_hz', 5.0); p('map_frame', 'map')
        p('a_lat_max', 15.0); p('v_max', 12.0)           # physics prior and speed cap
        p('a_accel', 5.0); p('a_brake', 8.0)             # profile limits [m/s^2]
        p('gp_sigma_margin', 1.0)                        # use mean - k*std of the GP (conservative)
        p('lap_min_length', 30.0); p('lap_radius', 3.0)  # lap detection
        p('slip_limit', 0.25)                            # |vy/vx| above this counts as unstable
        p('speed_up', 0.05); p('speed_down', 0.15)       # per-lap GP evidence step
        p('gp_save_path', '')                            # optional: persist refined GP between runs
        for f in CenterlineParams.__dataclass_fields__:
            p('cl.' + f, getattr(CenterlineParams(), f))
        g = lambda n: self.get_parameter(n).value
        d = CenterlineParams()
        self.cp = CenterlineParams(**{f: type(getattr(d, f))(g('cl.' + f)) for f in CenterlineParams.__dataclass_fields__})
        self.gp = SpeedGP(g('a_lat_max'), g('v_max'))
        self.map: Optional[ConeArray] = None
        self.corrected = False
        self.pose: Optional[np.ndarray] = None
        self.prof: Optional[dict] = None                 # last plan: s, xy, kappa, v
        self.samples: list = []                          # (kappa, speed, slip) logged during the lap
        self.start_xy: Optional[np.ndarray] = None
        self.travelled, self.laps, self.last_xy = 0.0, 0, None
        self.diag = LatencyDiag(self)
        self.create_subscription(ConeArray, g('map_topic'), self.on_map, 10)
        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(ConeArray, g('corrected_topic'), self.on_corrected, qos)
        self.create_subscription(PoseWithCovarianceStamped, g('pose_topic'), self.on_pose, 10)
        self.create_subscription(Odometry, g('odom_topic'), self.on_odom, 10)
        self.path_pub = self.create_publisher(Path, '/planning/centerline', 10)
        self.speed_pub = self.create_publisher(SpeedProfile, '/planning/velocity_profile', 10)
        self.create_timer(1.0 / g('plan_hz'), self.plan)

    # ---- inputs ---------------------------------------------------------------------------
    def on_map(self, m: ConeArray) -> None:
        if not self.corrected:                           # live map only until a corrected one exists
            self.map = m

    def on_corrected(self, m: ConeArray) -> None:
        self.map, self.corrected = m, True

    def on_pose(self, m: PoseWithCovarianceStamped) -> None:
        q = m.pose.pose.orientation
        self.pose = np.array([m.pose.pose.position.x, m.pose.pose.position.y,
                              2 * np.arctan2(q.z, q.w)])
        xy = self.pose[:2]
        if self.start_xy is None:
            self.start_xy = xy.copy()
        if self.last_xy is not None:
            self.travelled += float(np.linalg.norm(xy - self.last_xy))
        self.last_xy = xy.copy()
        g = lambda n: self.get_parameter(n).value
        if self.travelled > g('lap_min_length') and np.linalg.norm(xy - self.start_xy) < g('lap_radius'):
            self.finish_lap()

    def on_odom(self, m: Odometry) -> None:
        """Log (curvature, speed, slip) at the car's position on the current plan."""
        if self.prof is None or self.pose is None:
            return
        i = int(np.argmin(np.linalg.norm(self.prof['xy'] - self.pose[:2], axis=1)))
        vx, vy = m.twist.twist.linear.x, m.twist.twist.linear.y
        self.samples.append((self.prof['kappa'][i], vx, abs(vy) / max(abs(vx), 1.0)))

    # ---- online refinement ----------------------------------------------------------------
    def finish_lap(self) -> None:
        """Refine the GP with per-curvature-bin evidence from the lap just driven."""
        g = lambda n: self.get_parameter(n).value
        self.laps += 1
        self.travelled = 0.0
        S = np.array(self.samples).reshape(-1, 3)
        self.samples = []
        if len(S) < 20:
            return
        bins = np.floor(S[:, 0] / 0.02).astype(int)
        ks, vs = [], []
        for b in np.unique(bins):
            sel = S[bins == b]
            k = sel[:, 0].mean()
            pred = self.gp.predict(np.array([k]))[0][0]
            stable = sel[sel[:, 2] < g('slip_limit')]
            if len(sel) - len(stable) > 0.2 * len(sel):   # lots of slip: this speed was too high
                ks.append(k); vs.append(max(2.0, sel[:, 1].max() * (1 - g('speed_down'))))
            elif len(stable) and stable[:, 1].max() >= 0.9 * pred:   # reached the limit, stayed stable
                ks.append(k); vs.append(min(g('v_max'), stable[:, 1].max() * (1 + g('speed_up'))))
        if ks:
            self.gp.update(np.array(ks), np.array(vs))
        self.get_logger().info(f'lap {self.laps} done: GP refined with {len(ks)} curvature bins')
        path = g('gp_save_path')
        if path:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.gp, f)

    # ---- planning -------------------------------------------------------------------------
    def plan(self) -> None:
        if self.map is None or self.pose is None or not self.map.cones:
            return
        t0 = self.diag.start()
        g = lambda n: self.get_parameter(n).value
        pos = np.array([[c.x, c.y] for c in self.map.cones])
        col = np.array([c.color for c in self.map.cones])
        res = build_centerline(pos, col, self.pose, self.cp)
        if res is None:
            self.get_logger().warn('no centerline: need yellow and blue cones', throttle_duration_sec=5.0)
            return
        spline, length, closed = res
        s, xy, kappa, yaw = sample(spline, length, self.cp.ds, closed)
        mean, std = self.gp.predict(kappa)
        v = limit_accel(s, np.clip(mean - g('gp_sigma_margin') * std, 2.0, g('v_max')),
                        g('a_accel'), g('a_brake'), closed)
        self.prof = {'s': s, 'xy': xy, 'kappa': kappa, 'v': v}
        stamp = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp, path.header.frame_id = stamp, g('map_frame')
        for (x, y), th in zip(xy, yaw):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.z, ps.pose.orientation.w = float(np.sin(th / 2)), float(np.cos(th / 2))
            path.poses.append(ps)
        self.path_pub.publish(path)
        sp = SpeedProfile()
        sp.header = path.header
        sp.s, sp.speed, sp.sigma = s.tolist(), v.tolist(), std.tolist()
        self.speed_pub.publish(sp)
        self.diag.stop(t0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathPlanner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
