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
from mmcv.runner import load_checkpoint
from pyskl.models import Recognizer3D, build_model
from ultralytics import YOLO
from ultralytics.engine.results import Results

InteractiveShell.ast_node_interactivity = "all"

# Directory setup
remote_video_save_directory: Path = Path.home() / "remote_videos"
remote_video_save_directory.mkdir(parents=True, exist_ok=True)
captured_video_save_directory: Path = Path.home() / "remote_videos" / "captured"
captured_video_save_directory.mkdir(parents=True, exist_ok=True)
pyskl_path: Path = Path.home() / "repo/intel-edge-academy-6/external/pyskl"
pyskl_data_path: Path = pyskl_path / "data"

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
    / "repo/intel-edge-academy-6/external/pyskl/work_dirs/posec3d/x3d_shallow_ntu60_xsub/joint/best_top1_acc_epoch_24.pth"
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

        # Iterate over the detected persons in the results
        for result in results:
            print(f"length of results {len(results)}")

            # Check if boxes and track IDs are available
            if result.boxes is not None and result.boxes.id is not None:
                print(f"length of boxes {len(result.boxes)}")

                # Extract track_id from the detected person
                track_id = int(result.boxes.id[0].item())

                # Preprocess keypoints and store them
                keypoints = preprocess_keypoints(result)
                track_data[track_id].append(keypoints)
                last_update[track_id] = time.time()  # Update last seen timestamp

                # If 48 frames are accumulated, add them to the batch
                if len(track_data[track_id]) == 48:
                    input_tensor = torch.stack(
                        list(track_data[track_id]),
                        dim=2,  # (1, 2, 17, 48)
                    )
                    batch_keypoints.append(input_tensor)
                    track_ids.append(track_id)

        # Perform inference for multiple persons in a batch
        if batch_keypoints:
            # Stack keypoints into a batch (N, 1, 2, 17, 48)
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
