"""ROS integration test of the MPC/ANN nodes with fake inputs (needs a sourced, built workspace)."""
import os
import threading
import time
import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from rclpy.executors import MultiThreadedExecutor                       # noqa: E402
from rclpy.node import Node                                              # noqa: E402
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped    # noqa: E402
from nav_msgs.msg import Odometry, Path                                  # noqa: E402
from fs_msgs.msg import ControlCommand, GoSignal                         # noqa: E402
from fs_perception.msg import SpeedProfile                               # noqa: E402
from fs_control.mpc_controller_node import MpcController                 # noqa: E402
from fs_control.ann_controller_node import AnnController                 # noqa: E402

MODEL = os.path.join(os.path.dirname(__file__), '..', 'models', 'ann_controller.pt')


def feed(helper: Node, y_offset: float):
    """Publishers that describe: straight path along +x, car y_offset to the left, 5 m/s."""
    pose = helper.create_publisher(PoseWithCovarianceStamped, '/slam/pose', 10)
    odom = helper.create_publisher(Odometry, '/state/odometry', 10)
    path = helper.create_publisher(Path, '/planning/centerline', 10)
    prof = helper.create_publisher(SpeedProfile, '/planning/velocity_profile', 10)
    go = helper.create_publisher(GoSignal, '/fs_msgs/msg/GoSignal', 10)

    def publish_state() -> None:
        p = PoseWithCovarianceStamped(); p.pose.pose.position.y = y_offset; p.pose.pose.orientation.w = 1.0
        o = Odometry(); o.pose.pose.position.y = y_offset; o.pose.pose.orientation.w = 1.0
        o.twist.twist.linear.x = 5.0
        pose.publish(p); odom.publish(o)

    def publish_path() -> None:
        m = Path()
        for x in np.arange(0, 100, 0.5):
            ps = PoseStamped(); ps.pose.position.x = float(x); ps.pose.orientation.w = 1.0
            m.poses.append(ps)
        s = np.arange(0, 100, 0.5)
        sp = SpeedProfile(s=s.tolist(), speed=[8.0] * len(s), sigma=[0.0] * len(s))
        path.publish(m); prof.publish(sp)
    return publish_state, publish_path, go


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def test_mpc_node_gates_on_go_and_steers_right_when_left_of_path(ros) -> None:
    mpc, helper = MpcController(), Node('test_helper')
    got = []
    helper.create_subscription(ControlCommand, '/cmd', got.append, 10)
    state, path, go = feed(helper, y_offset=0.5)
    ex = MultiThreadedExecutor()
    for n in (mpc, helper):
        ex.add_node(n)
    t = threading.Thread(target=ex.spin, daemon=True); t.start()
    for _ in range(10):
        state(); path(); time.sleep(0.05)
    assert got and all(c.brake == 1.0 for c in got)                      # no GO yet -> full brake
    got.clear()
    go.publish(GoSignal())
    for _ in range(20):
        state(); path(); time.sleep(0.05)
    last = got[-1]
    assert last.brake == 0.0 and last.steering > 0.0                     # FSDS: +steering = right; car is left of path
    ex.shutdown(); mpc.destroy_node(); helper.destroy_node()


@pytest.mark.skipif(not os.path.exists(MODEL), reason='ANN model not trained')
def test_ann_node_publishes_command(ros) -> None:
    ann, helper = AnnController(), Node('test_helper2')
    ann.set_parameters([rclpy.parameter.Parameter('model_path', value=os.path.abspath(MODEL))])
    ann.model = __import__('torch').jit.load(os.path.abspath(MODEL)).eval()
    got = []
    helper.create_subscription(ControlCommand, '/control/ann_cmd', got.append, 10)
    state, path, _ = feed(helper, y_offset=0.5)
    ex = MultiThreadedExecutor()
    for n in (ann, helper):
        ex.add_node(n)
    t = threading.Thread(target=ex.spin, daemon=True); t.start()
    for _ in range(20):
        state(); path(); time.sleep(0.05)
    assert got and got[-1].steering > 0.0
    ex.shutdown(); ann.destroy_node(); helper.destroy_node()


def test_mpc_falls_back_to_ann_when_solve_too_slow(ros) -> None:
    # Fake ANN publisher in the helper node (the real ANN runs as its own process in a launch).
    mpc, helper = MpcController(), Node('test_helper3')
    mpc.set_parameters([rclpy.parameter.Parameter('solve_limit_ms', value=0.001)])   # impossible budget
    mpc.go = True
    got = []
    helper.create_subscription(ControlCommand, '/cmd', got.append, 10)
    ann_pub = helper.create_publisher(ControlCommand, '/control/ann_cmd', 10)
    state, path, _ = feed(helper, y_offset=0.5)
    ex = MultiThreadedExecutor()
    for n in (mpc, helper):
        ex.add_node(n)
    t = threading.Thread(target=ex.spin, daemon=True); t.start()
    for _ in range(30):
        state(); path(); ann_pub.publish(ControlCommand(steering=0.3, throttle=0.2)); time.sleep(0.05)
    assert mpc.fallbacks > 0
    assert got[-1].steering == pytest.approx(0.3) and got[-1].throttle == pytest.approx(0.2)   # ANN command used
    ex.shutdown(); mpc.destroy_node(); helper.destroy_node()
