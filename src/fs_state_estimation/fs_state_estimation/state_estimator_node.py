"""state_estimator_node: fuses /imu + /gss with a UKF, publishes /state/odometry at 100 Hz."""
from typing import Optional
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from fs_perception_py.diag import LatencyDiag
from fs_state_estimation.ukf import UKF, wrap, h_imu, h_gss_body, h_gss_world, make_h_bicycle


def quat_to_yaw(q) -> float:
    """Yaw from a geometry_msgs quaternion."""
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


class StateEstimator(Node):
    def __init__(self) -> None:
        super().__init__('state_estimator_node')
        # --- parameters (overridden by fs_bringup/config/state_estimator.yaml) ---
        p = self.declare_parameter
        p('imu_topic', '/imu'); p('gss_topic', '/gss'); p('odom_topic', '/state/odometry')
        p('gss_frame', 'auto')           # 'world', 'body' or 'auto' (detected from the first turn)
        p('rate_hz', 100.0)
        p('lr', 0.765)                   # CoG -> rear axle [m]
        p('q_diag', [1e-3, 1e-3, 1e-4, 0.5, 0.05, 0.1])   # process noise (per second)
        p('r_imu', [1e-3, 1e-3])         # [yaw var, yaw-rate var]
        p('r_gss', [0.01, 0.01])         # velocity variances
        p('r_bicycle', 0.2)              # strength of the vy = lr*r constraint
        g = lambda n: self.get_parameter(n).value
        self.gss_world = {'world': True, 'body': False}.get(g('gss_frame'))   # None = detect at runtime
        self.auto_score, self.auto_n = np.zeros(2), 0     # summed |heading error| under (world, body)
        self.R_imu, self.R_gss = np.diag(g('r_imu')), np.diag(g('r_gss'))
        self.R_bic = np.array([[g('r_bicycle')]])
        self.h_bic = make_h_bicycle(g('lr'))
        self.ukf = UKF(np.zeros(6), np.diag([1.0, 1.0, 0.5, 1.0, 1.0, 0.5]), np.diag(g('q_diag')))
        self.last_imu_t: Optional[float] = None
        self.initialised = False         # set once the first IMU yaw is known
        self.diag = LatencyDiag(self)
        self.create_subscription(Imu, g('imu_topic'), self.on_imu, qos_profile_sensor_data)
        self.create_subscription(TwistWithCovarianceStamped, g('gss_topic'), self.on_gss,
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(Odometry, g('odom_topic'), 10)
        self.create_timer(1.0 / g('rate_hz'), self.publish)

    def on_imu(self, m: Imu) -> None:
        """Predict with IMU acceleration, then correct with IMU yaw / yaw rate."""
        t0 = self.diag.start()
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        yaw = quat_to_yaw(m.orientation)
        if not self.initialised:                       # take initial heading from the IMU
            self.ukf.x[2], self.initialised = yaw, True
        elif self.last_imu_t is not None and 0 < t - self.last_imu_t < 0.5:
            self.ukf.predict(m.linear_acceleration.x, t - self.last_imu_t)
            self.ukf.update(np.array([yaw, m.angular_velocity.z]), h_imu, self.R_imu, z_angle_idx=[0])
            self.ukf.update(np.zeros(1), self.h_bic, self.R_bic)   # bicycle constraint
        self.last_imu_t = t
        self.diag.stop(t0)

    def on_gss(self, m: TwistWithCovarianceStamped) -> None:
        """Correct velocity with the ground speed sensor."""
        if not self.initialised:
            return
        z = np.array([m.twist.twist.linear.x, m.twist.twist.linear.y])
        if self.gss_world is None:                      # frame not known yet
            self._detect_gss_frame(z)
            if np.hypot(*z) > 0.3:
                return                                  # until known, only frame-independent standstill updates
        self.ukf.update(z, h_gss_world if self.gss_world is not False else h_gss_body, self.R_gss)

    def _detect_gss_frame(self, z: np.ndarray) -> None:
        """Decide world vs body frame. A car drives along its heading: the velocity direction equals
        yaw if the GSS is in the world frame, and is ~0 if it is in the body frame."""
        yaw = self.ukf.x[2]
        if np.hypot(*z) < 1.0 or abs(wrap(yaw)) < 0.15:   # need motion and a heading that separates the hypotheses
            return
        h = np.arctan2(z[1], z[0])
        self.auto_score += [abs(wrap(h - yaw)), abs(wrap(h))]
        self.auto_n += 1
        if self.auto_n >= 40:
            self.gss_world = bool(self.auto_score[0] < self.auto_score[1])
            self.get_logger().info(
                f'GSS frame detected: {"world" if self.gss_world else "body"} (mean heading error '
                f'world {self.auto_score[0] / 40:.2f} rad, body {self.auto_score[1] / 40:.2f} rad)')

    def publish(self) -> None:
        """Timer callback: publish the current estimate as nav_msgs/Odometry."""
        if not self.initialised:
            return
        x, P = self.ukf.x, self.ukf.P
        o = Odometry()
        o.header.stamp = self.get_clock().now().to_msg()
        o.header.frame_id, o.child_frame_id = 'odom', 'base_link'
        o.pose.pose.position.x, o.pose.pose.position.y = float(x[0]), float(x[1])
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = float(np.sin(x[2] / 2)), float(np.cos(x[2] / 2))
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.angular.z = (float(v) for v in x[3:6])
        # 6x6 row-major covariances: indices 0 (x), 7 (y), 35 (yaw / yaw-rate)
        o.pose.covariance[0], o.pose.covariance[7], o.pose.covariance[35] = P[0, 0], P[1, 1], P[2, 2]
        o.twist.covariance[0], o.twist.covariance[7], o.twist.covariance[35] = P[3, 3], P[4, 4], P[5, 5]
        self.pub.publish(o)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StateEstimator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
