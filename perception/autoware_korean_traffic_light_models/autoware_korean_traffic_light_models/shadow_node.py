"""Opt-in ROS 2 adapter. Never publishes to the production perception namespace."""
import argparse
import copy
import json
import signal
import time
from collections import OrderedDict
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tier4_perception_msgs.msg import TrafficLight, TrafficLightArray, TrafficLightElement, TrafficLightRoiArray
from autoware_korean_traffic_light_models.adapter_core import ROI, classify, image_from_message, unknown


def stamp(message):
    return message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec


def rois(message):
    return [ROI(int(x.traffic_light_id),int(x.traffic_light_type),int(x.roi.x_offset),
        int(x.roi.y_offset),int(x.roi.width),int(x.roi.height)) for x in message.rois]


class ShadowNode(Node):
    def __init__(self,args):
        super().__init__("aihub188_classifier_c")
        self.args = args
        # Check the resolved name as ROS remappings can override the CLI topic.
        if not self.resolve_topic_name(args.output_topic).startswith('/korean_tl/candidate/'):
            raise ValueError('Candidate output must stay under /korean_tl/candidate/')
        if args.backend == 'onnx':
            from autoware_korean_traffic_light_models.onnx_backend import ONNXBackend
            self.backend = ONNXBackend(args.model)
        else:
            from autoware_korean_traffic_light_models.trt_backend import TRTBackend
            self.backend = TRTBackend(args.engine,require_validated=True)
        self.images,self.regions = OrderedDict(),OrderedDict()
        self.last_regions = None
        self.last_publish_time = 0
        self.last_stamp = -1
        self.stale_sent = True
        self.publisher = self.create_publisher(TrafficLightArray,args.output_topic,10)
        self.diagnostics = self.create_publisher(String,"/korean_tl/candidate/diagnostics",10)
        self.create_subscription(Image,args.image_topic,lambda m:self.receive(m,True),qos_profile_sensor_data)
        self.create_subscription(TrafficLightRoiArray,args.roi_topic,lambda m:self.receive(m,False),qos_profile_sensor_data)
        self.create_timer(0.05,self.watchdog)

    def receive(self,message,is_image):
        if not is_image and self.args.car_only:
            message=copy.deepcopy(message)
            message.rois=[x for x in message.rois if x.traffic_light_type==0]
        key=stamp(message)
        # Reject future/zero stamps before cache ordering can poison subsequent inputs.
        if key<=0 or key>self.get_clock().now().nanoseconds+100_000_000:
            if not is_image:
                self.publish_result(message,[unknown(x,"stale_or_future_source_stamp") for x in rois(message)[:64]],0)
            return
        if key<=self.last_stamp:
            return
        if not is_image and len(message.rois)>64:
            message=copy.deepcopy(message)
            message.rois=message.rois[:64]
            self.last_stamp=key
            self.publish_result(message,[unknown(x,"too_many_rois") for x in rois(message)],0)
            self.last_regions=None
            self.stale_sent=True
            return
        cache=self.images if is_image else self.regions
        cache[key]=(time.monotonic(),message)
        while len(cache)>10:
            _,(_,evicted)=cache.popitem(last=False)
            if not is_image and stamp(evicted)>self.last_stamp:
                self.publish_result(evicted,[unknown(x,"queue_overflow") for x in rois(evicted)],0)
                self.last_stamp=stamp(evicted)
        if key not in self.images or key not in self.regions:
            return
        _,image=self.images.pop(key)
        _,regions=self.regions.pop(key)
        start=time.perf_counter()
        region_list=rois(regions)
        age=(self.get_clock().now().nanoseconds-key)/1e9
        reason=None
        if not image.header.frame_id or image.header.frame_id != regions.header.frame_id:
            reason="frame_id_mismatch"
        elif key<=0 or age>self.args.timeout or age < -0.1:
            reason="stale_or_future_source_stamp"
        if reason:
            results=[unknown(x,reason) for x in region_list]
        else:
            try:
                array=image_from_message(image)
                classify_fn=classify
                results=classify_fn(array,region_list,self.backend,self.args.confidence,self.args.margin)
            except Exception as exc:
                results=[unknown(x,str(exc)) for x in region_list]
        # Check processing did not itself outlive the validity window.
        if (self.get_clock().now().nanoseconds-key)/1e9>self.args.timeout:
            results=[unknown(x,"expired_during_processing") for x in region_list]
        if age >= -0.1:
            self.last_stamp=key
        self.last_regions=copy.deepcopy(regions)
        self.last_publish_time=time.monotonic()
        self.stale_sent=False
        self.publish_result(regions,results,(time.perf_counter()-start)*1000)

    def publish_result(self,source,results,elapsed):
        output=TrafficLightArray()
        output.header=copy.deepcopy(source.header)
        for r in results:
            signal=TrafficLight()
            signal.traffic_light_id=r["id"]
            signal.traffic_light_type=r["type"]
            for value in r["elements"]:
                element=TrafficLightElement()
                for key,field in value.items():
                    setattr(element,key,field)
                signal.elements.append(element)
            output.signals.append(signal)
        self.publisher.publish(output)
        diagnostic=String()
        diagnostic.data=json.dumps({"source_stamp_ns":stamp(source),"processing_ms":elapsed,
            "results":results,"status":"SHADOW_ONLY_NOT_VALIDATED_FOR_DRIVING"})
        self.diagnostics.publish(diagnostic)

    def watchdog(self):
        now=time.monotonic()
        for key,(received,message) in list(self.regions.items()):
            if key<=self.last_stamp:
                self.regions.pop(key)
            elif now-received>self.args.timeout:
                self.regions.pop(key)
                self.publish_result(message,[unknown(x,"missing_matching_image") for x in rois(message)],0)
                self.last_stamp=key
                self.last_regions=copy.deepcopy(message)
                self.stale_sent=True
        for key,(received,_) in list(self.images.items()):
            if now-received>self.args.timeout:
                self.images.pop(key)
        if self.last_regions is not None and not self.stale_sent and now-self.last_publish_time>self.args.timeout:
            self.publish_result(self.last_regions,[unknown(x,"input_timeout") for x in rois(self.last_regions)],0)
            self.stale_sent=True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--enable-shadow",action="store_true",help="Required explicit opt-in")
    parser.add_argument("--backend",choices=('onnx','tensorrt'),default='onnx')
    parser.add_argument("--model",type=Path)
    parser.add_argument("--engine",type=Path)
    parser.add_argument("--image-topic",default="/korean_tl/candidate/image")
    parser.add_argument("--roi-topic",default="/korean_tl/candidate/rois")
    parser.add_argument("--output-topic",default="/korean_tl/candidate/signals")
    parser.add_argument("--car-only",action="store_true")
    parser.add_argument("--confidence",type=float,default=0.8)
    parser.add_argument("--margin",type=float,default=0.2)
    parser.add_argument("--timeout",type=float,default=0.5)
    args,ros_args=parser.parse_known_args()
    if not args.enable_shadow:
        raise SystemExit("Disabled by default. Use --enable-shadow for isolated experiments.")
    if (args.backend == 'onnx' and args.model is None) or (args.backend == 'tensorrt' and args.engine is None):
        raise SystemExit('Select --model for ONNX or --engine for TensorRT')
    if not 0<=args.confidence<=1 or not 0<=args.margin<=1 or not 0.05<=args.timeout<=5:
        raise SystemExit("Invalid policy settings")
    rclpy.init(args=ros_args)
    node=None
    try:
        node=ShadowNode(args)
        rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException):
        pass
    finally:
        # ros2 launch may forward a second SIGINT while cleanup is already in progress.
        signal.signal(signal.SIGINT,signal.SIG_IGN)
        if node is not None:
            node.backend.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__=="__main__":
    main()
