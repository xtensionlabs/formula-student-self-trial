"""mpc_controller_node: MPC tracking of the centerline -> /cmd at 20 Hz, ANN fallback on slow solves.

Fallback rule: if the MPC solve fails or takes longer than `solve_limit_ms`, that time step is
driven by the ANN command (/control/ann_cmd). Every fallback is logged as a warning - never silent.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')   # small matrices: threaded BLAS only slows IPOPT down
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from fs_msgs.msg import ControlCommand, GoSignal
from fs_perception_py.diag import LatencyDiag
from fs_control.ctrl_common import ControlInputs, declare_common, from_command, to_command
from fs_control.mpc import Mpc, MpcWeights, VehicleLimits


class MpcController(Node):
    def __init__(self) -> None:
        super().__init__('mpc_controller_node')
        declare_common(self)
        p = self.declare_parameter
        p('cmd_topic', '/cmd'); p('ann_topic', '/control/ann_cmd'); p('go_topic', '/fs_msgs/msg/GoSignal')
        p('rate_hz', 20.0); p('horizon', 20); p('dt', 0.05)
        p('solve_limit_ms', 45.0); p('ann_max_age', 0.2)
        p('require_go', True)                         # stay stopped until the GO signal arrived
        p('wheelbase', 1.53); p('lr', 0.765)
        d = MpcWeights()
        for f in d.__dataclass_fields__:              # cost weights: parameters 'w.<name>'
            p('w.' + f, getattr(d, f))
        g = lambda n: self.get_parameter(n).value
        self.w = MpcWeights(**{f: float(g('w.' + f)) for f in d.__dataclass_fields__})
        lim = VehicleLimits(L=g('wheelbase'), lr=g('lr'), d_max=g('d_max'), a_min=g('a_min'), a_max=g('a_max'))
        self.mpc = Mpc(g('horizon'), g('dt'), lim, g('solve_limit_ms') * 1e-3)
        self.sub_group = ReentrantCallbackGroup()      # subscriptions must not wait for the control timer
        self.inputs = ControlInputs(self, self.sub_group)
        self.u_prev = np.zeros(2)                     # last applied [delta, a]
        self.go = not g('require_go')
        self.ann_cmd, self.ann_t = None, None
        self.fallbacks = 0
        self.diag = LatencyDiag(self)
        self.create_subscription(GoSignal, g('go_topic'), self._on_go, 10, callback_group=self.sub_group)
        self.create_subscription(ControlCommand, g('ann_topic'), self._on_ann, 10, callback_group=self.sub_group)
        self.pub = self.create_publisher(ControlCommand, g('cmd_topic'), 10)
        self.create_timer(1.0 / g('rate_hz'), self.step)

    def _on_go(self, _: GoSignal) -> None:
        if not self.go:
            self.get_logger().info('GO signal received')
        self.go = True

    def _on_ann(self, m: ControlCommand) -> None:
        self.ann_cmd, self.ann_t = m, self.get_clock().now()

    def _brake(self, why: str) -> None:
        """Safe default: full brake, no steering."""
        self.get_logger().warn(f'braking: {why}', throttle_duration_sec=2.0)
        c = ControlCommand(brake=1.0)
        c.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(c)

    def step(self) -> None:
        g = lambda n: self.get_parameter(n).value
        x, ref = self.inputs.state(), self.inputs.ref
        if not self.go:
            return self._brake('waiting for GO signal')
        if x is None or ref is None:
            return self._brake('no pose / path yet')
        t0 = self.diag.start()
        x0 = self.mpc.predict(x, self.u_prev)                         # compensate one step of delay
        r = ref.horizon(x0[:2], x0[3], self.mpc.N, self.mpc.dt)
        u, info = self.mpc.solve(x0, r, self.u_prev, self.w)
        if u is not None and info['ms'] <= g('solve_limit_ms'):
            cmd = to_command(self, float(u[0]), float(u[1]))
            self.u_prev = np.array(u)
        else:
            cmd = self._fallback(u is None, info['ms'])
            if cmd is None:
                return
        self.pub.publish(cmd)
        self.diag.stop(t0)

    def _fallback(self, failed: bool, ms: float):
        """Use the ANN command for this step; warn loudly. Returns the command or None."""
        self.fallbacks += 1
        reason = 'solve failed' if failed else f'solve took {ms:.1f} ms > {self.get_parameter("solve_limit_ms").value} ms'
        fresh = (self.ann_cmd is not None and
                 (self.get_clock().now() - self.ann_t).nanoseconds * 1e-9 < self.get_parameter('ann_max_age').value)
        if fresh:
            self.get_logger().warn(f'MPC {reason}: ANN fallback used (#{self.fallbacks})')
            self.u_prev = np.array(from_command(self, self.ann_cmd))
            return self.ann_cmd
        self.get_logger().error(f'MPC {reason} and no fresh ANN command: braking (#{self.fallbacks})')
        self._brake('MPC and ANN unavailable')
        return None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MpcController()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor(num_threads=3))
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
