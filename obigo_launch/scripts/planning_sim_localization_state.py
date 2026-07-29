#!/usr/bin/env python3

import rclpy
from autoware_adapi_v1_msgs.msg import LocalizationInitializationState
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy


class PlanningSimLocalizationState(Node):
    def __init__(self):
        super().__init__("planning_sim_localization_state")
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._state = LocalizationInitializationState.UNINITIALIZED
        self._pub = self.create_publisher(
            LocalizationInitializationState, "/localization/initialization_state", qos
        )
        self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose3d", self._on_initial_pose, 1
        )
        self.create_timer(1.0, self._publish)
        self._publish()

    def _on_initial_pose(self, _msg):
        self._state = LocalizationInitializationState.INITIALIZED
        self._publish()

    def _publish(self):
        msg = LocalizationInitializationState()
        msg.stamp = self.get_clock().now().to_msg()
        msg.state = self._state
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = PlanningSimLocalizationState()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
