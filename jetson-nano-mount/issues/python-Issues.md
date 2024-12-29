# Issues

## Errors

### Jupyter Kernel Crash During cv2.imshow()

- **Description**: Python Jupyter Interactive Kernel crashed.

#### 📅 Written at: 2024-10-17 20:45:56

#### **When it Occurs**

- Happens when using `cv2.imshow()` after initializing **PoseC3D models**:

    ```python
    import torch
    from pathlib import Path
    from pyskl.models import Recognizer3D, build_model
    from mmcv.runner import load_checkpoint

    pyskl_path: Path = Path.home() / "repo/synergy-hub/external/pyskl"
    pyskl_data_path: Path = pyskl_path / "data"

    checkpoint_path = (
        Path.home()
        / "repo/synergy-hub/external/pyskl/work_dirs/posec3d/x3d_shallow_ntu60_xsub/joint/best_top1_acc_epoch_24.pth"
    )

    checkpoint = torch.load(str(checkpoint_path), map_location="cuda")
    assert isinstance(checkpoint, dict)

    posec3d_model_config = execute_code_and_extract_variables(checkpoint["meta"]["config"])

    checkpoint_path = (
        pyskl_path
        / "work_dirs/posec3d/x3d_shallow_ntu60_xsub/joint/best_top1_acc_epoch_24.pth"
    )

    posec3d_model: Recognizer3D = build_model(posec3d_model_config["model"])

    load_checkpoint(posec3d_model, str(checkpoint_path), map_location="cuda")

    posec3d_model = torch.compile(model=posec3d_model)

    posec3d_model = posec3d_model.cuda().eval()
    ```

#### **Output Behavior**

- **In Jupyter Interactive Kernel**:

    ```plaintext
    The Kernel crashed while executing code in the current cell or a previous cell.
    Please review the code in the cell(s) to identify a possible cause of the failure.
    Click here for more info.
    View Jupyter log for further details.
    ```

- **If run as an executable**:

    ```plaintext
    <no message>
    ```

#### **Root Cause**

- Using `cv2.imshow()` within Jupyter can cause kernel crashes due to compatibility or memory management issues between **OpenCV** and the **Jupyter environment**.

#### **Workaround**

- Use **`from IPython.display import clear_output, display`** instead of `cv2.imshow()` to avoid crashes in Jupyter.

🛍️ **Example**: [OpenVINO Monodepth Notebook Example](https://docs.openvino.ai/2024/notebooks/vision-monodepth-with-output.html)

---

## Bugs

(You can add relevant bugs here.)
