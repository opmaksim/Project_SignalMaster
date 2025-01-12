# %% Setup
### requirees package: imageio[ffmpeg] (??)
# pytubefix not works for almost Youtube shorts. Download fails ; "<video_id> is unavailable"
# ➡️ yt-dlp ; https://pypi.org/project/yt-dlp/
import threading
import time
import tkinter as tk
from collections import defaultdict, deque
from pathlib import Path

import cv2
import torch
import torch.nn
import yt_dlp
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import Image as IPImage
from IPython.display import clear_output, display
from ultralytics import YOLO
from ultralytics.engine.results import Results

InteractiveShell.ast_node_interactivity = "all"

# Directory setup
remote_video_save_directory: Path = Path.home() / "remote_videos"
remote_video_save_directory.mkdir(parents=True, exist_ok=True)
captured_video_save_directory: Path = Path.home() / "remote_videos" / "captured"
captured_video_save_directory.mkdir(parents=True, exist_ok=True)
pyskl_path: Path = Path.home() / "repo/synergy-hub/external/pyskl"
pyskl_data_path: Path = pyskl_path / "data"
# %%
from mmcv.runner import load_checkpoint
from pyskl.models import Recognizer3D, build_model

action_classes = [
    "drink water",  # 1
    "eat meal/snack",  # 2
    "brushing teeth",  # 3
    "brushing hair",  # 4
    "drop",  # 5
    "pickup",  # 6
    "throw",  # 7
    "sitting down",  # 8
    "standing up (from sitting position)",  # 9
    "clapping",  # 10
    "reading",  # 11
    "writing",  # 12
    "tear up paper",  # 13
    "wear jacket",  # 14
    "take off jacket",  # 15
    "wear a shoe",  # 16
    "take off a shoe",  # 17
    "wear on glasses",  # 18
    "take off glasses",  # 19
    "put on a hat/cap",  # 20
    "take off a hat/cap",  # 21
    "cheer up",  # 22
    "hand waving",  # 23
    "kicking something",  # 24
    "reach into pocket",  # 25
    "hopping (one foot jumping)",  # 26
    "jump up",  # 27
    "make a phone call/answer phone",  # 28
    "playing with phone/tablet",  # 29
    "typing on a keyboard",  # 30
    "pointing to something with finger",  # 31
    "taking a selfie",  # 32
    "check time (from watch)",  # 33
    "rub two hands together",  # 34
    "nod head/bow",  # 35
    "shake head",  # 36
    "wipe face",  # 37
    "salute",  # 38
    "put the palms together",  # 39
    "cross hands in front",  # 40
    "sneeze/cough",  # 41
    "staggering",  # 42
    "falling",  # 43
    "touch head (headache)",  # 44
    "touch chest (stomachache/heart pain)",  # 45
    "touch back (backache)",  # 46
    "touch neck (neckache)",  # 47
    "nausea or vomiting condition",  # 48
    "use a fan (with hand or paper)/feeling warm",  # 49
    "punching/slapping other person",  # 50
    "kicking other person",  # 51
    "pushing other person",  # 52
    "pat on back of other person",  # 53
    "point finger at the other person",  # 54
    "hugging other person",  # 55
    "giving something to other person",  # 56
    "touch other person's pocket",  # 57
    "handshaking",  # 58
    "walking towards each other",  # 59
    "walking apart from each other",  # 60
]


def print_dict_structure(d, indent=0):
    """
    Written at 📅 2024-10-17 19:33:00

    🛍️ e.g. print_dict_structure(checkpoint)
    """
    for key, value in d.items():
        print("    " * indent + f"{key}: ", end="")
        if isinstance(value, dict):
            print("{")
            print_dict_structure(value, indent + 1)
            print("    " * indent + "}")
        else:
            print(f"{type(value)}")


def execute_code_and_extract_variables(code_str: str) -> dict:
    """
    Written at 📅 2024-10-17 19:33:00

    Executes a given string of Python code and returns the defined variables as a dictionary.

    Args:
        code_str (str): The Python code to execute as a string.

    Returns:
        dict: A dictionary containing the variables defined in the executed code.
    """
    local_vars = {}
    exec(code_str, {}, local_vars)  # Execute within an isolated scope.
    return local_vars


## Tkinter


from typing import List, Tuple


def popup_action_recognition(results: List[Tuple[int, str]], duration: int = 5) -> None:
    """
    Show a popup window at the center of the screen to display recognized actions for multiple IDs.

    :param results: A list of tuples, each containing (ID, action) to display.
    :param duration: The time (in seconds) for the popup to remain visible before closing.
    """
    # Create a new Tkinter window for the popup
    root = tk.Tk()
    root.title("Action Recognition")

    # Set the window size
    window_width = 400
    window_height = max(
        100, 30 * len(results) + 50
    )  # Adjust height based on number of results

    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate the position to center the window
    position_x = (screen_width - window_width) // 2
    position_y = (screen_height - window_height) // 2

    # Set the geometry of the window (width x height + x_offset + y_offset)
    root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

    # Create and pack labels for each ID and action
    for track_id, action in results:
        label = tk.Label(
            root,
            text=f"ID {track_id}: {action}",
            font=("Helvetica", 14),
            padx=20,
            pady=5,
        )
        label.pack()

    # Close the popup after the specified duration
    root.after(duration * 1000, root.destroy)

    # Start the Tkinter event loop to display the popup
    root.mainloop()


def show_action_popup(results: List[Tuple[int, str]], duration: int = 5) -> None:
    """
    Launch the action recognition popup in a separate thread.

    :param results: A list of tuples, each containing (ID, action) to display.
    :param duration: The time (in seconds) for the popup to remain visible.
    """
    # Create a new thread to run the popup window
    popup_thread = threading.Thread(
        target=popup_action_recognition, args=(results, duration)
    )
    popup_thread.start()


### Process
# YOLO Pose 모델 로드
model = YOLO("yolo11n-pose.torchscript")
# model = YOLO("yolo11m-pose")


# 체크포인트 경로
checkpoint_path = (
    Path.home()
    / "repo/synergy-hub/external/pyskl/work_dirs/posec3d/x3d_shallow_ntu60_xsub/joint/best_top1_acc_epoch_24.pth"
)

# 체크포인트 로드
checkpoint = torch.load(str(checkpoint_path), map_location="cuda")
assert isinstance(checkpoint, dict)

# 체크포인트에서 모델 아키텍처와 가중치 가져오기
posec3d_model_config = execute_code_and_extract_variables(checkpoint["meta"]["config"])

## written from inference_pytorch() ...
checkpoint_path = (
    pyskl_path
    / "work_dirs/posec3d/x3d_shallow_ntu60_xsub/joint/best_top1_acc_epoch_24.pth"
)

# 1️⃣ Initialize the model architecture
# The model is initialized first to define the layers and architecture.
# No weights are loaded at this point.
posec3d_model: Recognizer3D = build_model(posec3d_model_config["model"])

# 2️⃣ Load checkpoint weights
# Loading the trained weights from the checkpoint file.
# This step ensures the model is in the correct state with the learned parameters.
# It MUST be done before torch.compile() ⚠️
load_checkpoint(posec3d_model, str(checkpoint_path), map_location="cuda")

# 3️⃣ Compile the model with torch.compile()
# torch.compile() optimizes the model for faster inference on supported hardware.
# It wraps the model, so any further setup (like CUDA or eval) must come AFTER this step. 🏎️
posec3d_model = torch.compile(model=posec3d_model)

# 4️⃣ Move the model to CUDA and set to evaluation mode
# After compiling, it's crucial to move the model to GPU with `.cuda()` 🚀
# and set it to evaluation mode using `.eval()` to disable dropout, etc. 🧪
posec3d_model = posec3d_model.cuda().eval()

# At this point, the model is ready for real-time inference!
# 🎬 Start inference using the optimized and properly configured model.


# 각 track_id에 대해 48 프레임을 저장할 deque와 마지막 업데이트 시간 저장
track_data = defaultdict(lambda: deque(maxlen=48))
last_update = {}  # 각 track_id의 마지막 업데이트 시간 기록


def preprocess_keypoints(result: Results):
    """
    각 사람의 keypoints를 (1, 2, 17) 형식으로 추출합니다.
    """
    if result.keypoints is not None and result.keypoints.data.size(0) > 0:
        keypoints_tensor = result.keypoints.data[:, :, :2]  # (N, 17, 2) 형식
        keypoints_tensor = keypoints_tensor[0].permute(1, 0).unsqueeze(0)  # (1, 2, 17)
        return keypoints_tensor
    return torch.zeros(1, 2, 17)  # 데이터가 없을 경우 기본 텐서 반환


def remove_stale_ids():
    """30프레임 이상 업데이트되지 않은 ID를 삭제합니다."""
    current_time = time()
    stale_ids = [
        track_id
        for track_id, last_time in last_update.items()
        if current_time - last_time > 30 / cap.get(cv2.CAP_PROP_FPS)  # 30프레임 기준
    ]
    for track_id in stale_ids:
        del track_data[track_id]
        del last_update[track_id]


# %%
print(type(posec3d_model))


# %%
import torch
import torch_tensorrt
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


class TorchModelUtility:
    def __init__(self, model: torch.nn.Module) -> None:
        """Initialize the utility with a PyTorch model."""
        self.model = model

    def export_and_quantize_model(self, example_inputs: tuple) -> torch.fx.GraphModule:
        """
        Capture the model graph, apply static quantization, and return the quantized model.

        Args:
            example_inputs (tuple): Example inputs to trace the model graph.

        Returns:
            torch.fx.GraphModule: Quantized version of the original model.
        """
        # Step 1: Capture the model graph using torch.export
        print("Capturing the model graph...")
        exported_model = capture_pre_autograd_graph(self.model, *example_inputs)

        # Step 2: Configure the quantizer for static quantization
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        prepared_model = prepare_pt2e(exported_model, quantizer)

        # Step 3: Calibration (optional if using dummy input for calibration)
        print("Calibrating the model...")
        prepared_model(*example_inputs)

        # Step 4: Convert the model to a quantized version
        print("Converting the model to a quantized version...")
        quantized_model = convert_pt2e(prepared_model)

        return quantized_model

    def export_to_tensorrt(
        self, quantized_model: torch.nn.Module, input_shape: tuple
    ) -> torch.nn.Module:
        """
        Convert the quantized model to TensorRT format for optimized inference on NVIDIA GPUs.

        Args:
            quantized_model (torch.nn.Module): The quantized model to be converted.
            input_shape (tuple): Example input shape (e.g., (1, 3, 224, 224)).

        Returns:
            torch.nn.Module: TensorRT-optimized model.
        """
        print("Converting to TensorRT...")
        trt_model = torch_tensorrt.compile(
            quantized_model,
            inputs=[torch_tensorrt.Input(input_shape)],
            enabled_precisions={torch.float16},  # Enable FP16 for TensorRT
        )
        return trt_model


# %%

# Open the camera
cap = cv2.VideoCapture(0)

# Initialize tracking data and last update timestamps
track_data = defaultdict(list)
last_update = {}

try:
    while True:
        # Read a frame from the webcam
        success, frame = cap.read()
        if not success:
            break

        # Perform YOLO Pose inference
        results: list[Results] = model.track(frame, persist=True, verbose=False)
        ## from https://docs.ultralytics.com/modes/track/#persisting-tracks-loop
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Prepare batch keypoints and track IDs for multiple persons
        batch_keypoints = []
        track_ids = []

        if not results:
            continue
        # Yolo pose infers by getting one frame. not frames.
        result = results[0]
        # Check if boxes and track IDs are available
        for box in result.boxes:
            for a_id in box.id:
                # Extract track_id from the detected person
                track_id = int(a_id.item())

                # Preprocess keypoints (still on CPU)
                keypoints = preprocess_keypoints(result)
                # Store keypoints and update timestamp
                track_data[track_id].append(keypoints)
                last_update[track_id] = time.time()

                # If 48 frames are accumulated, add them to the batch
                if len(track_data[track_id]) == 48:
                    input_tensor = torch.stack(
                        list(track_data[track_id]),
                        dim=2,  # (1, 2, 17, 48)
                    ).to("cuda", dtype=torch.float32)
                    batch_keypoints.append(input_tensor)
                    track_ids.append(track_id)

        # Perform inference for multiple persons in a batch
        if batch_keypoints:
            # Stack keypoints into a batch (N, 1, 2, 17, 48)
            ##⚠️ It may make bottleneck when processing in cpu and try to move into GPU
            batch_tensor = torch.cat(batch_keypoints, dim=0).cuda()  # (N, 2, 17, 48)

            # Adjust dimensions: (N, 2, 17, 48) -> (N, 17, 48, 2)
            batch_tensor = batch_tensor.permute(0, 2, 3, 1)  # (N, 17, 48, 2)

            # Expand dimensions: (N, 17, 48, 2) -> (N, 17, 48, 1, 1)
            batch_tensor = batch_tensor.unsqueeze(-1).unsqueeze(-1)

            # Perform PoseC3D inference
            with torch.no_grad():
                output = posec3d_model(return_loss=False, imgs=batch_tensor)

            # Collect results for the popup display
            popup_results = []
            for i, track_id in enumerate(track_ids):
                action = output[i].argmax()
                action_label = action_classes[action]

                # Store the ID and action as a tuple
                popup_results.append((track_id, action_label))

            # Display the results in a popup window for 5 seconds
            show_action_popup(popup_results, duration=2)

        # Convert the frame to JPEG format for IPython display
        _, buffer = cv2.imencode(".jpg", annotated_frame)
        img_bytes = buffer.tobytes()

        # Clear previous output and display the current frame
        clear_output(wait=True)
        display(IPImage(data=img_bytes))

except Exception as e:
    # Print any errors that occur during execution
    print(e)

finally:
    # Release the video capture and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# %%


# %% Test Ipython display
cap = cv2.VideoCapture(0)  # 웹캠 초기화

try:
    while True:
        success, frame = cap.read()  # 프레임 읽기
        if not success:
            print("Failed to read frame.")
            break

        # OpenCV 배열을 JPEG로 인코딩
        _, buffer = cv2.imencode(".jpg", frame)
        img_bytes = buffer.tobytes()

        # IPython display로 이미지 출력
        clear_output(wait=True)
        display(IPImage(data=img_bytes))

except KeyboardInterrupt:
    print("Streaming stopped.")
finally:
    cap.release()  # 자원 해제


# %% Test Yolo11 pose for Youtube shorts

# coco-pose.yml ; https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml
# YouTube video URL
youtube_url = "https://www.youtube.com/shorts/8JT6uqvyx8M"


# Function to download the video using yt-dlp (built-in progress output)
def download_youtube_video_yt_dlp(
    url: str, save_directory: Path, video_filename: str
) -> bool:
    try:
        ydl_opts = {
            "format": "best",  # Download the best available quality
            "outtmpl": str(
                object=save_directory / video_filename
            ),  # Set output file path
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])  # Download the video
        print(f"Download completed: {video_filename}")
        return True
    except Exception as e:
        print(f"Error during download: {e}")
        return False


# Generate filename and check if the video exists
video_filename: str = "shorts_" + youtube_url.split("/")[-1]
video_path: Path = remote_video_save_directory / video_filename
print(video_path)
ret = True
# Download the video if it doesn't exist
if not video_path.exists():
    ret = download_youtube_video_yt_dlp(
        youtube_url, remote_video_save_directory, video_filename
    )
else:
    print(f"Video already exists: {video_path}")

# Ensure the download was successful
assert ret is True

# Load YOLO model
model = YOLO("yolo11m-pose.pt")  # Load an official Pose model

# Perform tracking with the model
results = model.track(f"{video_path}", show=True)  # Tracking with default tracker


# %% Test Yolo multiple object tracking
from collections import defaultdict

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Load the YOLO11 model
model = YOLO("yolo11m-pose.pt")

# Open the video file
cap = cv2.VideoCapture(0)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
try:
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results: list[Results] = model.track(frame, persist=True, verbose=False)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            # cv2.imshow("YOLO11 Tracking", annotated_frame)
            # for result in results:
            #     print("-============")
            #     print(type(result))
            #     print(result)
            #     print(type(result.keypoints))
            #     print(result.keypoints)
            #     print(type(result.keypoints.data))
            #     print(result.keypoints.data)
            #     print(result.keypoints.data.shape)
            #     print("-============")

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
finally:
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
# %%

"""
1. load submodule and install packages
git submodule update --init --recursive
# Deinitialize the submodule (if it's misconfigured)
# %shell> git submodule deinit -f -- external/pyskl
## ⚓ Prepare Skeleton dataset ; https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md

poetry install --with ml-pyskl
#### https://github.com/kennymckormick/pyskl/?tab=readme-ov-file

### https://github.com/kennymckormick/pyskl/tree/main/configs/posec3d




## ⚠️ fix mmcv File.. ....<venv>/lib/python3.12/site-packages/mmcv/utils/config.py
text, _ = FormatCode(text, style_config=yapf_style, verify=True)
to
text, _ = FormatCode(text, style_config=yapf_style)

## Training & Testing
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]

# For example: train PoseC3D on FineGYM (HRNet 2D skeleton, Joint Modality) with 8 GPUs, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/posec3d/x3d_shallow_ntu60_xsub/joint.py 1 --validate --test-last --test-best

##
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test PoseC3D on FineGYM (HRNet 2D skeleton, Joint Modality) with metrics `top_k_accuracy` and `mean_class_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/posec3d/x3d_shallow_ntu60_xsub/joint.py checkpoints/SOME_CHECKPOINT.pth 8 --eval top_k_accuracy mean_class_accuracy --out result.pkl
bash tools/dist_test.sh configs/posec3d/x3d_shallow_ntu60_xsub/joint.py work_dirs/posec3d/x3d_shallow_ntu60_xsub/joint/latest.pth 1 --eval top_k_accuracy mean_class_accuracy --out result.pkl


## Examples
⚓ Inference speed ; https://github.com/kennymckormick/pyskl/blob/main/examples/inference_speed.ipynb
⚓ Custom dataset ; https://github.com/kennymckormick/pyskl/blob/main/examples/extract_diving48_skeleton/diving48_example.ipynb
https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/custom_dataset_training.md


# train .. ❓ --deterministic
python tools/train.py configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint.py \
    --seed=0

    data/skeleton/gym_2d.pkl


mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest
wget https://download.openmmlab.com/mmaction/kinetics400_tiny.zip
mkdir -p data/
unzip kinetics400_tiny.zip -d data/


---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
File /home/wbfw109v2/repo/synergy-hub/mmaction_or_pyskl-files/ipython_test.py:156
    154 # Perform inference using PoseC3D model
    155 with torch.no_grad():
--> 156     output = posec3d_model(return_loss=False, imgs=input_tensor)
    158 # Get the predicted action class
    159 action = output.argmax(dim=1).item()

File ~/.cache/pypoetry/virtualenvs/python-study-JojGeswE-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1553, in Module._wrapped_call_impl(self, *args, **kwargs)
   1551     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1552 else:
-> 1553     return self._call_impl(*args, **kwargs)

File ~/.cache/pypoetry/virtualenvs/python-study-JojGeswE-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1562, in Module._call_impl(self, *args, **kwargs)
   1557 # If we don't have any hooks, we want to skip the rest of the logic in
   1558 # this function, and just call forward.
   1559 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1560         or _global_backward_pre_hooks or _global_backward_hooks
   1561         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1562     return forward_call(*args, **kwargs)
   1564 try:
   1565     result = None

File ~/repos/synergy-hub/external/pyskl/pyskl/models/recognizers/base.py:158, in BaseRecognizer.forward(self, imgs, label, return_loss, **kwargs)
...
--> 603 return F.conv3d(
    604     input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    605 )

RuntimeError: Expected 4D (unbatched) or 5D (batched) input to conv3d, but got input of size: [2, 48, 17, 1, 1, 1]



---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
File /home/wbfw109v2/repo/synergy-hub/mmaction_or_pyskl-files/ipython_test.py:113
    111 # Perform inference using PoseC3D model
    112 with torch.no_grad():
--> 113     output = posec3d_model(return_loss=False, imgs=input_tensor)
    115 # Get the predicted action class by finding the index of the maximum output value
    116 action = output.argmax(dim=1).item()

File ~/.cache/pypoetry/virtualenvs/python-study-JojGeswE-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1553, in Module._wrapped_call_impl(self, *args, **kwargs)
   1551     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1552 else:
-> 1553     return self._call_impl(*args, **kwargs)

File ~/.cache/pypoetry/virtualenvs/python-study-JojGeswE-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1562, in Module._call_impl(self, *args, **kwargs)
   1557 # If we don't have any hooks, we want to skip the rest of the logic in
   1558 # this function, and just call forward.
   1559 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1560         or _global_backward_pre_hooks or _global_backward_hooks
   1561         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1562     return forward_call(*args, **kwargs)
   1564 try:
   1565     result = None

File ~/repos/synergy-hub/external/pyskl/pyskl/models/recognizers/base.py:158, in BaseRecognizer.forward(self, imgs, label, return_loss, **kwargs)
...
--> 603 return F.conv3d(
    604     input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    605 )

🚨 RuntimeError: Given groups=1, weight of size [24, 17, 1, 3, 3], expected input[1, 34, 48, 1, 1] to have 17 channels, but got 34 channels instead


---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
File /home/wbfw109v2/repo/synergy-hub/mmaction_or_pyskl-files/ipython_test.py:118
    116 # Perform inference using PoseC3D model
    117 with torch.no_grad():
--> 118     output = posec3d_model(return_loss=False, imgs=input_tensor)
    120 # Get the predicted action class by finding the index of the maximum output value
    121 action = output.argmax(dim=1).item()

File ~/.cache/pypoetry/virtualenvs/python-study-JojGeswE-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1553, in Module._wrapped_call_impl(self, *args, **kwargs)
   1551     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1552 else:
-> 1553     return self._call_impl(*args, **kwargs)

File ~/.cache/pypoetry/virtualenvs/python-study-JojGeswE-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1562, in Module._call_impl(self, *args, **kwargs)
   1557 # If we don't have any hooks, we want to skip the rest of the logic in
   1558 # this function, and just call forward.
   1559 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1560         or _global_backward_pre_hooks or _global_backward_hooks
   1561         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1562     return forward_call(*args, **kwargs)
   1564 try:
   1565     result = None

File ~/repos/synergy-hub/external/pyskl/pyskl/models/recognizers/base.py:158, in BaseRecognizer.forward(self, imgs, label, return_loss, **kwargs)
...
    535 def _check_input_dim(self, input):
    536     if input.dim() != 5:
--> 537         raise ValueError(f"expected 5D input (got {input.dim()}D input)")

ValueError: expected 5D input (got 4D input)




---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
File /home/wbfw109v2/repo/synergy-hub/mmaction_or_pyskl-files/ipython_test.py:121
    119 # Perform inference using PoseC3D model
    120 with torch.no_grad():
--> 121     output = posec3d_model(return_loss=False, imgs=input_tensor)
    123 # Get the predicted action class by finding the index of the maximum output value
    124 action = output.argmax(dim=1).item()

File ~/.cache/pypoetry/virtualenvs/python-study-JojGeswE-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1553, in Module._wrapped_call_impl(self, *args, **kwargs)
   1551     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1552 else:
-> 1553     return self._call_impl(*args, **kwargs)

File ~/.cache/pypoetry/virtualenvs/python-study-JojGeswE-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1562, in Module._call_impl(self, *args, **kwargs)
   1557 # If we don't have any hooks, we want to skip the rest of the logic in
   1558 # this function, and just call forward.
   1559 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1560         or _global_backward_pre_hooks or _global_backward_hooks
   1561         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1562     return forward_call(*args, **kwargs)
   1564 try:
   1565     result = None

File ~/repos/synergy-hub/external/pyskl/pyskl/models/recognizers/base.py:158, in BaseRecognizer.forward(self, imgs, label, return_loss, **kwargs)
...
--> 603 return F.conv3d(
    604     input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    605 )

RuntimeError: Given groups=1, weight of size [24, 17, 1, 3, 3], expected input[17, 48, 1, 2, 1] to have 17 channels, but got 48 channels instead

---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
File /home/wbfw109v2/repo/synergy-hub/mmaction_or_pyskl-files/ipython_test.py:116
    113 input_tensor = input_tensor.permute(0, 3, 2, 1)  # (1, 17, 48, 2)
    115 # Ensure input tensor is 5D (N, C, T, H, W)
--> 116 input_tensor = input_tensor.view(1, 17, 48, 1, 2)  # (N, C, T, H, W)
    118 # Perform inference using PoseC3D model
    119 with torch.no_grad():

RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead

이 문제는 input_tensor의 메모리 배치가 연속적이지 않아서 발생한 것입니다. 이를 해결하기 위해 view 대신 reshape를 사용할 수 있습니다
.. ? ❓ 최적화 해야하나.. view 쓰는 코드로..



str -> dictd 로 내가 변환해서 하면.. load_checkpoint ㅏ하면 이렇게 됨..
load checkpoint from local path: /home/wbfw109v2/repo/synergy-hub/external/pyskl/work_dirs/posec3d/x3d_shallow_ntu60_xsub/joint/best_top1_acc_epoch_24.pth
The model and loaded state dict do not match exactly

unexpected key in source state_dict: backbone.conv1_s.conv.weight, backbone.conv1_t.conv.weight, backbone.conv1_t.bn.weight, backbone.conv1_t.bn.bias, backbone.conv1_t.bn.running_mean, backbone.conv1_t.bn.running_var, backbone.conv1_t.bn.num_batches_tracked, backbone.layer1.0.downsample.conv.weight, backbone.layer1.0.downsample.bn.weight, backbone.layer1.0.downsample.bn.bias, backbone.layer1.0.downsample.bn.running_mean, backbone.layer1.0.downsample.bn.running_var, backbone.layer1.0.downsample.bn.num_batches_tracked, backbone.layer1.0.conv1.conv.weight, backbone.layer1.0.conv1.bn.weight, backbone.layer1.0.conv1.bn.bias, backbone.layer1.0.conv1.bn.running_mean, backbone.layer1.0.conv1.bn.running_var, backbone.layer1.0.conv1.bn.num_batches_tracked, backbone.layer1.0.conv2.conv.weight, backbone.layer1.0.conv2.bn.weight, backbone.layer1.0.conv2.bn.bias, backbone.layer1.0.conv2.bn.running_mean, backbone.layer1.0.conv2.bn.running_var, backbone.layer1.0.conv2.bn.num_batches_tracked, backbone.layer1.0.conv3.conv.weight, backbone.layer1.0.conv3.bn.weight, backbone.layer1.0.conv3.bn.bias, backbone.layer1.0.conv3.bn.running_mean, backbone.layer1.0.conv3.bn.running_var, backbone.layer1.0.conv3.bn.num_batches_tracked, backbone.layer1.1.conv1.conv.weight, backbone.layer1.1.conv1.bn.weight, backbone.layer1.1.conv1.bn.bias, backbone.layer1.1.conv1.bn.running_mean, backbone.layer1.1.conv1.bn.running_var, backbone.layer1.1.conv1.bn.num_batches_tracked, backbone.layer1.1.conv2.conv.weight, backbone.layer1.1.conv2.bn.weight, backbone.layer1.1.conv2.bn.bias, backbone.layer1.1.conv2.bn.running_mean, backbone.layer1.1.conv2.bn.running_var, backbone.layer1.1.conv2.bn.num_batches_tracked, backbone.layer1.1.conv3.conv.weight, backbone.layer1.1.conv3.bn.weight, backbone.layer1.1.conv3.bn.bias, backbone.layer1.1.conv3.bn.running_mean, backbone.layer1.1.conv3.bn.running_var, backbone.layer1.1.conv3.bn.num_batches_tracked, backbone.layer2.0.downsample.conv.weight, backbone.layer2.0.downsample.bn.weight, backbone.layer2.0.downsample.bn.bias, backbone.layer2.0.downsample.bn.running_mean, backbone.layer2.0.downsample.bn.running_var, backbone.layer2.0.downsample.bn.num_batches_tracked, backbone.layer2.0.conv1.conv.weight, backbone.layer2.0.conv1.bn.weight, backbone.layer2.0.conv1.bn.bias, backbone.layer2.0.conv1.bn.running_mean, backbone.layer2.0.conv1.bn.running_var, backbone.layer2.0.conv1.bn.num_batches_tracked, backbone.layer2.0.conv2.conv.weight, backbone.layer2.0.conv2.bn.weight, backbone.layer2.0.conv2.bn.bias, backbone.layer2.0.conv2.bn.running_mean, backbone.layer2.0.conv2.bn.running_var, backbone.layer2.0.conv2.bn.num_batches_tracked, backbone.layer2.0.conv3.conv.weight, backbone.layer2.0.conv3.bn.weight, backbone.layer2.0.conv3.bn.bias, backbone.layer2.0.conv3.bn.running_mean, backbone.layer2.0.conv3.bn.running_var, backbone.layer2.0.conv3.bn.num_batches_tracked, backbone.layer2.1.conv1.conv.weight, backbone.layer2.1.conv1.bn.weight, backbone.layer2.1.conv1.bn.bias, backbone.layer2.1.conv1.bn.running_mean, backbone.layer2.1.conv1.bn.running_var, backbone.layer2.1.conv1.bn.num_batches_tracked, backbone.layer2.1.conv2.conv.weight, backbone.layer2.1.conv2.bn.weight, backbone.layer2.1.conv2.bn.bias, backbone.layer2.1.conv2.bn.running_mean, backbone.layer2.1.conv2.bn.running_var, backbone.layer2.1.conv2.bn.num_batches_tracked, backbone.layer2.1.conv3.conv.weight, backbone.layer2.1.conv3.bn.weight, backbone.layer2.1.conv3.bn.bias, backbone.layer2.1.conv3.bn.running_mean, backbone.layer2.1.conv3.bn.running_var, backbone.layer2.1.conv3.bn.num_batches_tracked, backbone.layer2.2.conv1.conv.weight, backbone.layer2.2.conv1.bn.weight, backbone.layer2.2.conv1.bn.bias, backbone.layer2.2.conv1.bn.running_mean, backbone.layer2.2.conv1.bn.running_var, backbone.layer2.2.conv1.bn.num_batches_tracked, backbone.layer2.2.conv2.conv.weight, backbone.layer2.2.conv2.bn.weight, backbone.layer2.2.conv2.bn.bias, backbone.layer2.2.conv2.bn.running_mean, backbone.layer2.2.conv2.bn.running_var, backbone.layer2.2.conv2.bn.num_batches_tracked, backbone.layer2.2.conv3.conv.weight, backbone.layer2.2.conv3.bn.weight, backbone.layer2.2.conv3.bn.bias, backbone.layer2.2.conv3.bn.running_mean, backbone.layer2.2.conv3.bn.running_var, backbone.layer2.2.conv3.bn.num_batches_tracked, backbone.layer2.3.conv1.conv.weight, backbone.layer2.3.conv1.bn.weight, backbone.layer2.3.conv1.bn.bias, backbone.layer2.3.conv1.bn.running_mean, backbone.layer2.3.conv1.bn.running_var, backbone.layer2.3.conv1.bn.num_batches_tracked, backbone.layer2.3.conv2.conv.weight, backbone.layer2.3.conv2.bn.weight, backbone.layer2.3.conv2.bn.bias, backbone.layer2.3.conv2.bn.running_mean, backbone.layer2.3.conv2.bn.running_var, backbone.layer2.3.conv2.bn.num_batches_tracked, backbone.layer2.3.conv3.conv.weight, backbone.layer2.3.conv3.bn.weight, backbone.layer2.3.conv3.bn.bias, backbone.layer2.3.conv3.bn.running_mean, backbone.layer2.3.conv3.bn.running_var, backbone.layer2.3.conv3.bn.num_batches_tracked, backbone.layer2.4.conv1.conv.weight, backbone.layer2.4.conv1.bn.weight, backbone.layer2.4.conv1.bn.bias, backbone.layer2.4.conv1.bn.running_mean, backbone.layer2.4.conv1.bn.running_var, backbone.layer2.4.conv1.bn.num_batches_tracked, backbone.layer2.4.conv2.conv.weight, backbone.layer2.4.conv2.bn.weight, backbone.layer2.4.conv2.bn.bias, backbone.layer2.4.conv2.bn.running_mean, backbone.layer2.4.conv2.bn.running_var, backbone.layer2.4.conv2.bn.num_batches_tracked, backbone.layer2.4.conv3.conv.weight, backbone.layer2.4.conv3.bn.weight, backbone.layer2.4.conv3.bn.bias, backbone.layer2.4.conv3.bn.running_mean, backbone.layer2.4.conv3.bn.running_var, backbone.layer2.4.conv3.bn.num_batches_tracked, backbone.layer3.0.downsample.conv.weight, backbone.layer3.0.downsample.bn.weight, backbone.layer3.0.downsample.bn.bias, backbone.layer3.0.downsample.bn.running_mean, backbone.layer3.0.downsample.bn.running_var, backbone.layer3.0.downsample.bn.num_batches_tracked, backbone.layer3.0.conv1.conv.weight, backbone.layer3.0.conv1.bn.weight, backbone.layer3.0.conv1.bn.bias, backbone.layer3.0.conv1.bn.running_mean, backbone.layer3.0.conv1.bn.running_var, backbone.layer3.0.conv1.bn.num_batches_tracked, backbone.layer3.0.conv2.conv.weight, backbone.layer3.0.conv2.bn.weight, backbone.layer3.0.conv2.bn.bias, backbone.layer3.0.conv2.bn.running_mean, backbone.layer3.0.conv2.bn.running_var, backbone.layer3.0.conv2.bn.num_batches_tracked, backbone.layer3.0.conv3.conv.weight, backbone.layer3.0.conv3.bn.weight, backbone.layer3.0.conv3.bn.bias, backbone.layer3.0.conv3.bn.running_mean, backbone.layer3.0.conv3.bn.running_var, backbone.layer3.0.conv3.bn.num_batches_tracked, backbone.layer3.1.conv1.conv.weight, backbone.layer3.1.conv1.bn.weight, backbone.layer3.1.conv1.bn.bias, backbone.layer3.1.conv1.bn.running_mean, backbone.layer3.1.conv1.bn.running_var, backbone.layer3.1.conv1.bn.num_batches_tracked, backbone.layer3.1.conv2.conv.weight, backbone.layer3.1.conv2.bn.weight, backbone.layer3.1.conv2.bn.bias, backbone.layer3.1.conv2.bn.running_mean, backbone.layer3.1.conv2.bn.running_var, backbone.layer3.1.conv2.bn.num_batches_tracked, backbone.layer3.1.conv3.conv.weight, backbone.layer3.1.conv3.bn.weight, backbone.layer3.1.conv3.bn.bias, backbone.layer3.1.conv3.bn.running_mean, backbone.layer3.1.conv3.bn.running_var, backbone.layer3.1.conv3.bn.num_batches_tracked, backbone.layer3.2.conv1.conv.weight, backbone.layer3.2.conv1.bn.weight, backbone.layer3.2.conv1.bn.bias, backbone.layer3.2.conv1.bn.running_mean, backbone.layer3.2.conv1.bn.running_var, backbone.layer3.2.conv1.bn.num_batches_tracked, backbone.layer3.2.conv2.conv.weight, backbone.layer3.2.conv2.bn.weight, backbone.layer3.2.conv2.bn.bias, backbone.layer3.2.conv2.bn.running_mean, backbone.layer3.2.conv2.bn.running_var, backbone.layer3.2.conv2.bn.num_batches_tracked, backbone.layer3.2.conv3.conv.weight, backbone.layer3.2.conv3.bn.weight, backbone.layer3.2.conv3.bn.bias, backbone.layer3.2.conv3.bn.running_mean, backbone.layer3.2.conv3.bn.running_var, backbone.layer3.2.conv3.bn.num_batches_tracked, backbone.conv5.conv.weight, backbone.conv5.bn.weight, backbone.conv5.bn.bias, backbone.conv5.bn.running_mean, backbone.conv5.bn.running_var, backbone.conv5.bn.num_batches_tracked, cls_head.fc_cls.weight, cls_head.fc_cls.bias

missing keys in source state_dict: _orig_mod.backbone.conv1_s.conv.weight, _orig_mod.backbone.conv1_t.conv.weight, _orig_mod.backbone.conv1_t.bn.weight, _orig_mod.backbone.conv1_t.bn.bias, _orig_mod.backbone.conv1_t.bn.running_mean, _orig_mod.backbone.conv1_t.bn.running_var, _orig_mod.backbone.layer1.0.downsample.conv.weight, _orig_mod.backbone.layer1.0.downsample.bn.weight, _orig_mod.backbone.layer1.0.downsample.bn.bias, _orig_mod.backbone.layer1.0.downsample.bn.running_mean, _orig_mod.backbone.layer1.0.downsample.bn.running_var, _orig_mod.backbone.layer1.0.conv1.conv.weight, _orig_mod.backbone.layer1.0.conv1.bn.weight, _orig_mod.backbone.layer1.0.conv1.bn.bias, _orig_mod.backbone.layer1.0.conv1.bn.running_mean, _orig_mod.backbone.layer1.0.conv1.bn.running_var, _orig_mod.backbone.layer1.0.conv2.conv.weight, _orig_mod.backbone.layer1.0.conv2.bn.weight, _orig_mod.backbone.layer1.0.conv2.bn.bias, _orig_mod.backbone.layer1.0.conv2.bn.running_mean, _orig_mod.backbone.layer1.0.conv2.bn.running_var, _orig_mod.backbone.layer1.0.conv3.conv.weight, _orig_mod.backbone.layer1.0.conv3.bn.weight, _orig_mod.backbone.layer1.0.conv3.bn.bias, _orig_mod.backbone.layer1.0.conv3.bn.running_mean, _orig_mod.backbone.layer1.0.conv3.bn.running_var, _orig_mod.backbone.layer1.1.conv1.conv.weight, _orig_mod.backbone.layer1.1.conv1.bn.weight, _orig_mod.backbone.layer1.1.conv1.bn.bias, _orig_mod.backbone.layer1.1.conv1.bn.running_mean, _orig_mod.backbone.layer1.1.conv1.bn.running_var, _orig_mod.backbone.layer1.1.conv2.conv.weight, _orig_mod.backbone.layer1.1.conv2.bn.weight, _orig_mod.backbone.layer1.1.conv2.bn.bias, _orig_mod.backbone.layer1.1.conv2.bn.running_mean, _orig_mod.backbone.layer1.1.conv2.bn.running_var, _orig_mod.backbone.layer1.1.conv3.conv.weight, _orig_mod.backbone.layer1.1.conv3.bn.weight, _orig_mod.backbone.layer1.1.conv3.bn.bias, _orig_mod.backbone.layer1.1.conv3.bn.running_mean, _orig_mod.backbone.layer1.1.conv3.bn.running_var, _orig_mod.backbone.layer2.0.downsample.conv.weight, _orig_mod.backbone.layer2.0.downsample.bn.weight, _orig_mod.backbone.layer2.0.downsample.bn.bias, _orig_mod.backbone.layer2.0.downsample.bn.running_mean, _orig_mod.backbone.layer2.0.downsample.bn.running_var, _orig_mod.backbone.layer2.0.conv1.conv.weight, _orig_mod.backbone.layer2.0.conv1.bn.weight, _orig_mod.backbone.layer2.0.conv1.bn.bias, _orig_mod.backbone.layer2.0.conv1.bn.running_mean, _orig_mod.backbone.layer2.0.conv1.bn.running_var, _orig_mod.backbone.layer2.0.conv2.conv.weight, _orig_mod.backbone.layer2.0.conv2.bn.weight, _orig_mod.backbone.layer2.0.conv2.bn.bias, _orig_mod.backbone.layer2.0.conv2.bn.running_mean, _orig_mod.backbone.layer2.0.conv2.bn.running_var, _orig_mod.backbone.layer2.0.conv3.conv.weight, _orig_mod.backbone.layer2.0.conv3.bn.weight, _orig_mod.backbone.layer2.0.conv3.bn.bias, _orig_mod.backbone.layer2.0.conv3.bn.running_mean, _orig_mod.backbone.layer2.0.conv3.bn.running_var, _orig_mod.backbone.layer2.1.conv1.conv.weight, _orig_mod.backbone.layer2.1.conv1.bn.weight, _orig_mod.backbone.layer2.1.conv1.bn.bias, _orig_mod.backbone.layer2.1.conv1.bn.running_mean, _orig_mod.backbone.layer2.1.conv1.bn.running_var, _orig_mod.backbone.layer2.1.conv2.conv.weight, _orig_mod.backbone.layer2.1.conv2.bn.weight, _orig_mod.backbone.layer2.1.conv2.bn.bias, _orig_mod.backbone.layer2.1.conv2.bn.running_mean, _orig_mod.backbone.layer2.1.conv2.bn.running_var, _orig_mod.backbone.layer2.1.conv3.conv.weight, _orig_mod.backbone.layer2.1.conv3.bn.weight, _orig_mod.backbone.layer2.1.conv3.bn.bias, _orig_mod.backbone.layer2.1.conv3.bn.running_mean, _orig_mod.backbone.layer2.1.conv3.bn.running_var, _orig_mod.backbone.layer2.2.conv1.conv.weight, _orig_mod.backbone.layer2.2.conv1.bn.weight, _orig_mod.backbone.layer2.2.conv1.bn.bias, _orig_mod.backbone.layer2.2.conv1.bn.running_mean, _orig_mod.backbone.layer2.2.conv1.bn.running_var, _orig_mod.backbone.layer2.2.conv2.conv.weight, _orig_mod.backbone.layer2.2.conv2.bn.weight, _orig_mod.backbone.layer2.2.conv2.bn.bias, _orig_mod.backbone.layer2.2.conv2.bn.running_mean, _orig_mod.backbone.layer2.2.conv2.bn.running_var, _orig_mod.backbone.layer2.2.conv3.conv.weight, _orig_mod.backbone.layer2.2.conv3.bn.weight, _orig_mod.backbone.layer2.2.conv3.bn.bias, _orig_mod.backbone.layer2.2.conv3.bn.running_mean, _orig_mod.backbone.layer2.2.conv3.bn.running_var, _orig_mod.backbone.layer2.3.conv1.conv.weight, _orig_mod.backbone.layer2.3.conv1.bn.weight, _orig_mod.backbone.layer2.3.conv1.bn.bias, _orig_mod.backbone.layer2.3.conv1.bn.running_mean, _orig_mod.backbone.layer2.3.conv1.bn.running_var, _orig_mod.backbone.layer2.3.conv2.conv.weight, _orig_mod.backbone.layer2.3.conv2.bn.weight, _orig_mod.backbone.layer2.3.conv2.bn.bias, _orig_mod.backbone.layer2.3.conv2.bn.running_mean, _orig_mod.backbone.layer2.3.conv2.bn.running_var, _orig_mod.backbone.layer2.3.conv3.conv.weight, _orig_mod.backbone.layer2.3.conv3.bn.weight, _orig_mod.backbone.layer2.3.conv3.bn.bias, _orig_mod.backbone.layer2.3.conv3.bn.running_mean, _orig_mod.backbone.layer2.3.conv3.bn.running_var, _orig_mod.backbone.layer2.4.conv1.conv.weight, _orig_mod.backbone.layer2.4.conv1.bn.weight, _orig_mod.backbone.layer2.4.conv1.bn.bias, _orig_mod.backbone.layer2.4.conv1.bn.running_mean, _orig_mod.backbone.layer2.4.conv1.bn.running_var, _orig_mod.backbone.layer2.4.conv2.conv.weight, _orig_mod.backbone.layer2.4.conv2.bn.weight, _orig_mod.backbone.layer2.4.conv2.bn.bias, _orig_mod.backbone.layer2.4.conv2.bn.running_mean, _orig_mod.backbone.layer2.4.conv2.bn.running_var, _orig_mod.backbone.layer2.4.conv3.conv.weight, _orig_mod.backbone.layer2.4.conv3.bn.weight, _orig_mod.backbone.layer2.4.conv3.bn.bias, _orig_mod.backbone.layer2.4.conv3.bn.running_mean, _orig_mod.backbone.layer2.4.conv3.bn.running_var, _orig_mod.backbone.layer3.0.downsample.conv.weight, _orig_mod.backbone.layer3.0.downsample.bn.weight, _orig_mod.backbone.layer3.0.downsample.bn.bias, _orig_mod.backbone.layer3.0.downsample.bn.running_mean, _orig_mod.backbone.layer3.0.downsample.bn.running_var, _orig_mod.backbone.layer3.0.conv1.conv.weight, _orig_mod.backbone.layer3.0.conv1.bn.weight, _orig_mod.backbone.layer3.0.conv1.bn.bias, _orig_mod.backbone.layer3.0.conv1.bn.running_mean, _orig_mod.backbone.layer3.0.conv1.bn.running_var, _orig_mod.backbone.layer3.0.conv2.conv.weight, _orig_mod.backbone.layer3.0.conv2.bn.weight, _orig_mod.backbone.layer3.0.conv2.bn.bias, _orig_mod.backbone.layer3.0.conv2.bn.running_mean, _orig_mod.backbone.layer3.0.conv2.bn.running_var, _orig_mod.backbone.layer3.0.conv3.conv.weight, _orig_mod.backbone.layer3.0.conv3.bn.weight, _orig_mod.backbone.layer3.0.conv3.bn.bias, _orig_mod.backbone.layer3.0.conv3.bn.running_mean, _orig_mod.backbone.layer3.0.conv3.bn.running_var, _orig_mod.backbone.layer3.1.conv1.conv.weight, _orig_mod.backbone.layer3.1.conv1.bn.weight, _orig_mod.backbone.layer3.1.conv1.bn.bias, _orig_mod.backbone.layer3.1.conv1.bn.running_mean, _orig_mod.backbone.layer3.1.conv1.bn.running_var, _orig_mod.backbone.layer3.1.conv2.conv.weight, _orig_mod.backbone.layer3.1.conv2.bn.weight, _orig_mod.backbone.layer3.1.conv2.bn.bias, _orig_mod.backbone.layer3.1.conv2.bn.running_mean, _orig_mod.backbone.layer3.1.conv2.bn.running_var, _orig_mod.backbone.layer3.1.conv3.conv.weight, _orig_mod.backbone.layer3.1.conv3.bn.weight, _orig_mod.backbone.layer3.1.conv3.bn.bias, _orig_mod.backbone.layer3.1.conv3.bn.running_mean, _orig_mod.backbone.layer3.1.conv3.bn.running_var, _orig_mod.backbone.layer3.2.conv1.conv.weight, _orig_mod.backbone.layer3.2.conv1.bn.weight, _orig_mod.backbone.layer3.2.conv1.bn.bias, _orig_mod.backbone.layer3.2.conv1.bn.running_mean, _orig_mod.backbone.layer3.2.conv1.bn.running_var, _orig_mod.backbone.layer3.2.conv2.conv.weight, _orig_mod.backbone.layer3.2.conv2.bn.weight, _orig_mod.backbone.layer3.2.conv2.bn.bias, _orig_mod.backbone.layer3.2.conv2.bn.running_mean, _orig_mod.backbone.layer3.2.conv2.bn.running_var, _orig_mod.backbone.layer3.2.conv3.conv.weight, _orig_mod.backbone.layer3.2.conv3.bn.weight, _orig_mod.backbone.layer3.2.conv3.bn.bias, _orig_mod.backbone.layer3.2.conv3.bn.running_mean, _orig_mod.backbone.layer3.2.conv3.bn.running_var, _orig_mod.backbone.conv5.conv.weight, _orig_mod.backbone.conv5.bn.weight, _orig_mod.backbone.conv5.bn.bias, _orig_mod.backbone.conv5.bn.running_mean, _orig_mod.backbone.conv5.bn.running_var, _orig_mod.cls_head.fc_cls.weight, _orig_mod.cls_head.fc_cls.bias

load_checkpoint(posec3d_model, str(checkpoint_path), map_location="cuda")

posec3d_model 변수가 아키텍처고 가중치를 cechkpoint_path 로부터 로드




Written at 📅 2024-10-17 21:48:25

# Why YOLO Returns `list[Results]`

## 1. Batch Processing Support
- YOLO can process **multiple frames in a batch**.
- Each frame’s result is stored in one `Results` object.
- Example:
  ```python
  frames = [frame1, frame2, frame3]
  results = model.track(frames, persist=True)  # List with 3 results
  print(len(results))  # Output: 3
  ```

## 2. Consistent API for Single and Batch Frames
- Even with a **single frame**, the result is returned as a list for **consistency**.
- This allows YOLO to handle both **single frame and batch inputs** with the same API.
  ```python
  results = model.track(frame, persist=True)
  print(len(results))  # Output: 1
  result = results[0]  # Access the first (and only) result
  ```

## 3. Multi-Camera Stream Support
- When processing **multiple camera streams**, each frame from different cameras gets its own `Results` object in the list.
  ```python
  frames = [camera1_frame, camera2_frame]
  results = model.track(frames, persist=True)
  print(len(results))  # Output: 2
  ```

## 4. Why `boxes` is an Array
- `result.boxes` contains **all detected objects** within a single frame.
- Each box corresponds to one detected object, including **ID, bounding box, and keypoints**.

Example:
  ```plaintext
  Number of detected boxes: 2
  Object 0: ID: [0], Box: [100, 200, 50, 80]
  Object 1: ID: [1], Box: [300, 400, 70, 90]
  ```

❓ 정적 양자화
    - PoseC3D 가 (N, C, T, H, W) 를 받는데, N 은 항상 달라질 수 있으므로 정적 양자화는 불가능하다고 한다.
    
NTU RGB+D: 켈리브레이션 데이터셋이 필요한가?

"""

# %%
``