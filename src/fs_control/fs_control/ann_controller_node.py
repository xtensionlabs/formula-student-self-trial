"""ann_controller_node: TorchScript MLP [cte, heading_err, speed_err, curvature] -> [delta, a].

Publishes ControlCommand on /control/ann_cmd (the MPC node uses it as fallback; remap it to /cmd
to drive with the ANN alone).
"""
import numpy as np
import rclpy
import torch
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from fs_msgs.msg import ControlCommand
from fs_perception_py.diag import LatencyDiag
from fs_control.ctrl_common import ControlInputs, declare_common, to_command


class AnnController(Node):
    def __init__(self) -> None:
        super().__init__('ann_controller_node')
        declare_common(self)
        p = self.declare_parameter
        p('model_path', ''); p('ann_topic', '/control/ann_cmd'); p('rate_hz', 20.0)
        g = lambda n: self.get_parameter(n).value
        self.model = None
        if g('model_path'):
            self.model = torch.jit.load(g('model_path')).eval()        # TorchScript, no python class needed
            self.get_logger().info(f'ANN loaded: {g("model_path")}')
        else:
            self.get_logger().error('model_path not set: ANN controller will publish nothing')
        self.sub_group = ReentrantCallbackGroup()      # subscriptions must not wait for the control timer
        self.inputs = ControlInputs(self, self.sub_group)
        self.diag = LatencyDiag(self)
        self.pub = self.create_publisher(ControlCommand, g('ann_topic'), 10)
        self.create_timer(1.0 / g('rate_hz'), self.step)

    def step(self) -> None:
        x, ref = self.inputs.state(), self.inputs.ref
        if self.model is None or x is None or ref is None:
            return
        t0 = self.diag.start()
        feat = torch.tensor(ref.features(x[:3], x[3]), dtype=torch.float32)[None]
        with torch.no_grad():
            delta, a = self.model(feat)[0].tolist()
        self.pub.publish(to_command(self, delta, a))
        self.diag.stop(t0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AnnController()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor(num_threads=3))
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
