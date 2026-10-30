# 한국 신호등 인지 소스 후보

YOLOX-S `scale_02`와 MobileNetV2 `candidate C`를 위한 **명시적 선택형 평가 패키지**입니다.
CARLA, C-Track, 합성 신호, GT 위치, 임의 지도 좌표에 의존하지 않습니다.
기존 센서 드라이버·판단·제어·차량 보정 및 기본 신호등 인지 launch는 변경하지 않습니다.
현재는 소스 후보이며 **HIL·실차 승인 또는 검증 완료 배포판이 아닙니다.**

## 제공 내용

- 학습 모델 4개(batch 1/6)의 URL·크기·SHA-256과 다운로드/검증 도구.
- 기존 검증에 사용한 C 전처리, 6개 출력 해석, confidence 0.8/margin 0.2, UNKNOWN 처리.
- CPU ONNX 평가 backend와 기존 TensorRT 10/CUDA 12 classifier backend.
- native YOLOX fine detector와 C classifier를 별도 출력에 연결하는 후보 launch.
- 단위 테스트 및 ROS 메시지 smoke 테스트. 전체 파이프라인/ARM64 검증은 별도입니다.

모델은 이미 공개된 `katech-autoware-sils`의 Release에서 공유합니다. 저장소 이름이 SILS여도
이 모델 파일에 CARLA 의존성은 없습니다. Git에는 모델 본체·데이터셋·GPU 엔진을 넣지 않습니다.
기존 `TFLite` 5분류 어댑터는 이번 C 6출력 모델과 다르므로 가져오지 않았습니다.

## 입력/출력 계약

YOLOX: BGR float32 NCHW `[N,3,416,416]`, 0..255, cubic letterbox/padding 114.
출력 `[N,3549,8]`은 stride 8/16/32 raw grid입니다. 세 class 슬롯은
background(항상 0), 차량 신호등, 보행 신호등입니다. 차량/보행자 객체 탐지 모델이 아닙니다.
native detector의 기존 GPU bilinear 전처리는 기본값을 유지하고, 후보 launch에서만
새 `cuda_preprocess=false` 옵션으로 CPU cubic 전처리를 선택합니다.

Classifier: RGB → PIL bicubic 70×20 → CHW /255, ImageNet 정규화 없음.
출력: red, yellow, green, greenleft, redleft, reject_crop.
단순 녹색은 CIRCLE만, greenleft/redleft는 LEFT_ARROW를 추가합니다.
이 출력 계약이 실제 차량 판단기의 좌회전 정책을 검증했다는 의미는 아닙니다.
보행 신호 분류는 이 패키지의 대상이 아닙니다.

## 소스 빌드 및 모델 확보

ROS 2 Humble 및 해당 k-autoware 의존성을 준비한 workspace에서 실행합니다.

```bash
colcon build --packages-up-to autoware_korean_traffic_light_models
source install/setup.bash
python3 -m pip install -r src/universe/autoware.universe/perception/autoware_korean_traffic_light_models/requirements-onnx.txt
ros2 run autoware_korean_traffic_light_models models download --directory /absolute/path/models/korean_tl
ros2 run autoware_korean_traffic_light_models models verify --directory /absolute/path/models/korean_tl
```

의존성 설치는 평가용 환경에서만 수행하십시오. 기존 운영 Python 환경을 덮어쓰지 마십시오.
모델 경로는 사용자가 선택하며, 모델 폴더는 native detector 엔진 캐시를 쓸 수 있어야 합니다.
기존 파일의 해시가 다르면 덮어쓰지 않고 중단합니다.

## 후보 실행 (기존 주행 출력에 연결하지 않음)

먼저 image/rough ROI/expected ROI가 같은 timestamp와 frame_id를 갖는지 확인합니다.
ROI와 신호 ID는 실차의 카메라 보정·위치추정·Lanelet2 투영에서 공급해야 합니다.
이 패키지는 map-based ROI 생성기나 센서 드라이버를 대신 실행하지 않습니다.

```bash
ros2 launch autoware_korean_traffic_light_models candidate.launch.py \
  enable_candidate:=true models_dir:=/absolute/path/models/korean_tl \
  image_topic:=/sensing/camera/camera6/image_raw \
  rough_roi_topic:=/perception/traffic_light_recognition/camera6/detection/rough/rois \
  expected_roi_topic:=/perception/traffic_light_recognition/camera6/detection/expect/rois
```

기본 `enable_candidate=false`에서는 노드를 실행하지 않습니다. 결과는
`/korean_tl/candidate/signals`에만 발행하며 기존 planner/traffic-signal fusion에 연결되지 않습니다.
classifier는 ROS remap 후에도 이 후보 namespace 밖으로 출력하도록 설정하면 중단합니다.
카메라 한 대 평가용이며 다중 카메라는 노드/출력 namespace 분리 작업이 필요합니다.
default는 classifier CPU ONNX, detector NVIDIA TensorRT이며 지연/처리율은 미보증입니다.
상위 저장소 기본 No-CUDA 이미지에서는 이 native detector를 실행할 수 없습니다.
NVIDIA/CUDA를 갖춘 별도 평가 빌드가 필요하며, 기존 공식 이미지 발행 제한은 유지합니다.
TensorRT classifier 옵션은 대상 GPU에서 재생성하고 `trt_validation.json`과 엔진 해시를
검증한 파일만 받습니다. ONNX를 기존 native MobileNet 설정에 단순 교체하면 안 됩니다.

## 아직 필요한 검증

amd64 전체 perception 재빌드, native detector+classifier 연속 메시지 재생,
실센서/오검출/timeout/신호 ID 대응, TensorRT 수치 비교, 처리율, HIL과 차량 회귀 검증이 남습니다.
ARM64/Jetson은 별도 이미지 빌드 및 JetPack/CUDA/TensorRT 호환 확인이 필요합니다.
amd64 바이너리·TensorRT 엔진을 ARM64로 복사해서 사용하지 않습니다.
후보 출력을 실제 인지 출력으로 전환하거나 main 병합하는 것은 이번 소스 공개에 포함되지 않습니다.
