# %% Setup
### requirees package: imageio[ffmpeg] (??)
# pytubefix not works for almost Youtube shorts. Download fails ; "<video_id> is unavailable"
# ➡️ yt-dlp ; https://pypi.org/project/yt-dlp/
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell
from ultralytics import YOLO

InteractiveShell.ast_node_interactivity = "all"

# Directory setup
remote_video_save_directory: Path = Path.home() / "remote_videos"
remote_video_save_directory.mkdir(parents=True, exist_ok=True)
captured_video_save_directory: Path = Path.home() / "remote_videos" / "captured"
captured_video_save_directory.mkdir(parents=True, exist_ok=True)
pyskl_path: Path = Path.home() / "repo/synergy-hub/external/pyskl"
pyskl_data_path: Path = pyskl_path / "data"
# %%
# python -c 'import tensorrt as trt;  print(trt.__version__)'
"""

py`thon -m pip install nvidia-cudnn-cu12==9.0.0.312 --no-cache-dir venv/scripts/python -m pip install --pre --extra-index-url https://pypi.nvidia.com/ tensorrt==9.3.0.post12.dev1 --no-cache-dir venv/scripts/python -m pip uninstall -y nvidia-cudnn-cu12 venv/scripts/python -m pip install -r requirements.txt

스왑 파일 생성 및 설정:
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

스왑이 정상적으로 설정되었는지 확인:
swapon --show
free -h  # 메모리 및 스왑 사용량 확인

영구적으로 스왑 활성화 (재부팅 후에도 유지되도록):
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab



sudo nvpmodel -m 0  # Max-N 모드로 전환
sudo jetson_clocks
sudo nvpmodel -q  # 관리자 권한으로 현재 전력 모드 확인

sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
sudo sh -c 'echo 1 > /sys/devices/pwm-fan/target_pwm'
sudo sh -c 'echo 64 > /sys/devices/pwm-fan/target_pwm'


    model.export(
        format="engine",  # Export as TensorRT engine
        device="cuda:0",
        half=True,  # INT8 quantization for better memory efficiency
        # dynamic=False,
        batch=1,
        workspace=256,
        simplify=True,
        # imgsz=imgsz
        verbose=True,
    )



"""

# Load the YOLOv8 model
model = YOLO("yolo11n.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolov11n.engine'
### Required: Python binding for NVIDIA's TensorRT
## 🧮 %shell> sudo apt install -y tensorrt python3-libnvinfer

# %%
# # Load the exported TensorRT model
# tensorrt_model = YOLO("yolov11n.engine")

# # Run inference
# results = tensorrt_model("https://ultralytics.com/images/bus.jpg")


# # 💡 Remember calibration for INT8 is specific to each device
# # It is critical to ensure that the same device that will use the TensorRT model weights for deployment is used for exporting with INT8 precision, as the calibration results can vary across devices.

# model = YOLO("yolov11n-pose.pt")
# model.export(
#     format="engine",
#     dynamic=True,
#     batch=8,
#     workspace=2,
#     int8=True,
#     data="coco.yaml",
# )

# # Load the exported TensorRT INT8 model
# model = YOLO("yolov11n-pose.engine", task="pose")

# # Run inference
# result = model.predict("https://ultralytics.com/images/bus.jpg")
