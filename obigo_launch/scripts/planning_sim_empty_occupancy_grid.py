#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from rclpy.node import Node


class PlanningSimEmptyOccupancyGrid(Node):
    def __init__(self):
        super().__init__("planning_sim_empty_occupancy_grid")
        self._resolution = self.declare_parameter("resolution", 0.5).value
        self._width = self.declare_parameter("width", 300).value
        self._height = self.declare_parameter("height", 300).value
        self._frame_id = self.declare_parameter("frame_id", "map").value
        self._center_x = 0.0
        self._center_y = 0.0
        self._data = [0] * (self._width * self._height)

        self._pub = self.create_publisher(OccupancyGrid, "/perception/occupancy_grid_map/map", 1)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose3d", self._on_pose, 1)
        self.create_subscription(Odometry, "/localization/kinematic_state", self._on_odometry, 1)
        self.create_timer(0.5, self._publish)

    def _on_pose(self, msg):
        self._center_x = msg.pose.pose.position.x
        self._center_y = msg.pose.pose.position.y

    def _on_odometry(self, msg):
        self._center_x = msg.pose.pose.position.x
        self._center_y = msg.pose.pose.position.y

    def _publish(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.info.resolution = self._resolution
        msg.info.width = self._width
        msg.info.height = self._height
        msg.info.origin.position.x = self._center_x - self._width * self._resolution * 0.5
        msg.info.origin.position.y = self._center_y - self._height * self._resolution * 0.5
        msg.info.origin.orientation.w = 1.0
        msg.data = self._data
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = PlanningSimEmptyOccupancyGrid()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
