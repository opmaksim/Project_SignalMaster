# %%
import cv2

# Import necessary libraries
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output, display

InteractiveShell.ast_node_interactivity = "all"


print("hello")

# %% Camera streaming test


def display_frame_in_notebook(frame, delay=1):
    """Displays a single frame in Jupyter notebook."""
    clear_output(wait=True)  # Clear previous output to simulate a video stream
    plt.imshow(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )  # Convert BGR to RGB for correct colors
    plt.axis("off")  # Hide axes
    display(plt.gcf())  # Display the current figure
    plt.close()  # Close to prevent memory leaks


# Open the video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Display the frame in the notebook
        display_frame_in_notebook(frame)

except KeyboardInterrupt:
    # Stop the video on manual interrupt
    print("Video stream stopped")

finally:
    cap.release()  # Release the camera
    cv2.destroyAllWindows()


# %% Camera open test

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

print(frame)
# if __name__ == "__main__":
#     cap = cv2.VideoCapture(0)
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             cv2.imshow("Test Window", frame)

#             # 종료 조건 (ESC 키를 누르면 종료)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()


# %%
print("??")

"""
sudo apt update
sudo apt install -y x11-apps


[ WARN:0@0.125] global cap_v4l.cpp:999 open VIDEOIO(V4L2:/dev/video0): can't open camera by index
[ WARN:0@0.126] global obsensor_stream_channel_v4l2.cpp:82 xioctl ioctl: fd=-1, req=-2140645888
[ WARN:0@0.126] global obsensor_stream_channel_v4l2.cpp:138 queryUvcDeviceInfoList ioctl error return: 9
[ERROR:0@0.126] global obsensor_uvc_stream_channel.cpp:158 getStreamChannelGroup Camera index out of range
>> ls -l /dev/video0
🧮 sudo usermod -aG video vscode

"""
