"""Explicit opt-in evaluation only. Does not publish driving perception topics."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    get = lambda name: LaunchConfiguration(name).perform(context)
    if get('enable_candidate').lower() != 'true':
        return []
    from autoware_korean_traffic_light_models.model_io import model_record
    folder = Path(get('models_dir'))
    detector = folder / 'korean_tl_yolox_s_pilot_batch_6.onnx'
    classifier = folder / 'candidate_c_batch_6.onnx'
    model_record(detector, 'detector')
    model_record(classifier, 'classifier')
    share = Path(get_package_share_directory('autoware_korean_traffic_light_models'))
    sim_time = get('use_sim_time').lower() == 'true'
    return [
        Node(package='autoware_traffic_light_fine_detector', executable='traffic_light_fine_detector_node',
             name='korean_candidate_detector', namespace='/korean_tl/candidate', output='screen',
             parameters=[{
                 'use_sim_time': sim_time, 'fine_detector_model_path': str(detector),
                 'fine_detector_label_path': str(share / 'config/detector.labels.txt'),
                 'fine_detector_precision': 'fp32', 'fine_detector_score_thresh': 0.3,
                 'fine_detector_nms_thresh': 0.65, 'approximate_sync': False,
                 'cuda_preprocess': False, 'gpu_id': 0,
             }],
             remappings=[('~/input/image', get('image_topic')), ('~/input/rois', get('rough_roi_topic')),
                         ('~/expect/rois', get('expected_roi_topic')), ('~/output/rois', '/korean_tl/candidate/rois')]),
        Node(package='autoware_korean_traffic_light_models', executable='classifier_shadow',
             name='korean_candidate_classifier', output='screen', parameters=[{'use_sim_time': sim_time}],
             arguments=['--enable-shadow', '--backend', 'onnx', '--model', str(classifier),
                        '--image-topic', get('image_topic'), '--roi-topic', '/korean_tl/candidate/rois',
                        '--output-topic', '/korean_tl/candidate/signals', '--car-only']),
    ]


def generate_launch_description():
    defaults = {
        'enable_candidate': 'false', 'models_dir': '', 'use_sim_time': 'false',
        'image_topic': '/sensing/camera/camera6/image_raw',
        'rough_roi_topic': '/perception/traffic_light_recognition/camera6/detection/rough/rois',
        'expected_roi_topic': '/perception/traffic_light_recognition/camera6/detection/expect/rois',
    }
    return LaunchDescription([*[DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()], OpaqueFunction(function=setup)])
