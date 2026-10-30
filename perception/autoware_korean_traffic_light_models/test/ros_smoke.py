"""Run with the pinned ROS 2 dependencies. No hardware or driving outputs."""
from types import SimpleNamespace
import time
import numpy as np
import rclpy
from sensor_msgs.msg import Image
from tier4_perception_msgs.msg import TrafficLightRoi, TrafficLightRoiArray, TrafficLightElement
from autoware_korean_traffic_light_models import onnx_backend, shadow_node


class FakeBackend:
    batch = 1
    def __init__(self, unused): pass
    def infer(self, data): return np.array([[0, 0, 0, 1, 0, 0]], dtype=np.float32)
    def close(self): pass


class Capture:
    def __init__(self): self.messages = []
    def publish(self, message): self.messages.append(message)


def main():
    onnx_backend.ONNXBackend = FakeBackend
    args = SimpleNamespace(backend='onnx', model=None, output_topic='/korean_tl/candidate/signals',
                           image_topic='/test/image', roi_topic='/test/rois', car_only=True,
                           confidence=0.8, margin=0.2, timeout=0.5)
    rclpy.init()
    node = shadow_node.ShadowNode(args)
    results = Capture()
    node.publisher = results
    node.diagnostics = Capture()
    try:
        image = Image()
        image.header.stamp = node.get_clock().now().to_msg()
        image.header.frame_id = 'camera'
        image.height, image.width, image.step, image.encoding = 20, 70, 210, 'rgb8'
        image.data = np.random.default_rng(1).integers(0, 256, (20, 70, 3), dtype=np.uint8).tobytes()
        roi = TrafficLightRoi()
        roi.traffic_light_id, roi.traffic_light_type = 7, 0
        roi.roi.width, roi.roi.height = 70, 20
        regions = TrafficLightRoiArray()
        regions.header = image.header
        regions.rois = [roi]
        node.receive(regions, False)
        node.receive(image, True)
        elements = results.messages[-1].signals[0].elements
        assert [(e.color, e.shape) for e in elements] == [(TrafficLightElement.GREEN, TrafficLightElement.CIRCLE), (TrafficLightElement.GREEN, TrafficLightElement.LEFT_ARROW)]
        assert len(results.messages) == 1
        node.receive(image, True)
        assert len(results.messages) == 1, 'Duplicate stamp republished'
        node.last_publish_time = time.monotonic() - 1
        node.watchdog()
        assert results.messages[-1].signals[0].elements[0].color == TrafficLightElement.UNKNOWN
        assert len(results.messages) == 2
        node.watchdog()
        assert len(results.messages) == 2
        assert node.resolve_topic_name(args.output_topic) == args.output_topic
        print('ROS smoke: exact sync, green-left mapping, duplicate rejection, stale UNKNOWN, one-shot timeout: PASS')
    finally:
        node.backend.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__': main()
