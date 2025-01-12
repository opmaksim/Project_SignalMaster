# Issues

- [Issues](#issues)
  - [Errors](#errors)
    - [Jupyter Kernel Crash During cv2.imshow()](#jupyter-kernel-crash-during-cv2imshow)
      - [**When it Occurs**](#when-it-occurs)
      - [**Output Behavior**](#output-behavior)
      - [**Root Cause**](#root-cause)
      - [**Workaround**](#workaround)
    - [TensorRT Export Failure: AssertionError with Range Operator (Error Occurred During Model Quantization)](#tensorrt-export-failure-assertionerror-with-range-operator-error-occurred-during-model-quantization)
      - [**When it Occurs**](#when-it-occurs-1)
      - [**Output Behavior**](#output-behavior-1)
      - [**Root Cause**](#root-cause-1)
      - [**Workaround**](#workaround-1)
      - [**References**](#references)
  - [Bugs](#bugs)

## Errors

### Jupyter Kernel Crash During cv2.imshow()

- 📅 Written at: 2024-10-17 20:45:56

  - Written by wbfw109v2

**Description**: Python Jupyter Interactive Kernel crashed.

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

&nbsp;

---

### TensorRT Export Failure: AssertionError with Range Operator (Error Occurred During Model Quantization)

- 📅 Written at: 2025-01-13 23:44:05

  - Written by wbfw109v2

**Description**: TensorRT export fails when converting a YOLO model to TensorRT format due to an assertion error related to the Range operator in the ONNX model.

#### **When it Occurs**

- Happens during the TensorRT export process when executing the `model.export()` function in the following context:

  ```python
  model.export(
      format="engine",
      device="cuda",
      int8=True,
      workspace=512,
      batch=1,
      simplify=True,
  )
  ```

- Specifically, this issue occurs while processing the ONNX model during TensorRT conversion.

#### **Output Behavior**

- **Console Output**:

  ```plaintext
  TensorRT: starting export with TensorRT 8.2.0.6...
  [TRT] [E] ModelImporter.cpp:773: While parsing node number 263 [Range -> "onnx::Add_674"]:
  [TRT] [E] ModelImporter.cpp:775: Assertion failed: inputs.at(0).isInt32() && "For range operator with dynamic inputs, this version of TensorRT only supports INT32!"
  TensorRT: export failure ❌ 87.2s: failed to load ONNX file: /path/to/yolo11n-pose.onnx
  ```

- **Exception in Python**:

  ```plaintext
  RuntimeError: failed to load ONNX file: /path/to/yolo11n-pose.onnx
  ```

#### **Root Cause**

- The issue arises because the TensorRT version (8.2.0.6) does not fully support the ONNX `Range` operator with dynamic inputs unless all inputs are `INT32`.
- The ONNX model contains a `Range` operator that is not compatible with the TensorRT parser's expectations.

#### **Workaround**

1. **Convert to FP16 Precision**:

   - Since Jetson Nano is limited to Ubuntu 18 with fixed Python and CUDA versions, upgrading TensorRT or Python versions is not feasible.
   - Modify the export configuration to use FP16 precision instead of INT8 quantization:

   ```python
   model.export(
       format="engine",
       device="cuda",
       fp16=True,  # Use FP16 precision instead of INT8
       workspace=512,
       batch=1,
       simplify=True,
   )
   ```

   This avoids the INT32 constraint by utilizing FP16, which is more compatible with the Jetson Nano's hardware.

2. **Preprocess ONNX Model**:

   - Use an ONNX graph editor to identify and replace or simplify nodes involving the `Range` operator.
   - Tools such as `onnx-simplifier` or `onnxruntime` can help with graph modifications.

   ```bash
   python3 -m onnxsim input_model.onnx output_model.onnx
   ```

3. **Alternative Backends**:

   - Export the model to a different runtime format (e.g., ONNX Runtime or OpenVINO) if TensorRT conversion is not essential.

4. **Debug ONNX Model**:
   - Inspect the ONNX model to locate the problematic `Range` node using ONNX visualization tools like Netron.

#### **References**

- GitHub Issue: [Ultralytics/ultralytics#2084](https://github.com/ultralytics/ultralytics/issues/2084)
- StackOverflow Discussion: [Assertion failed: inputs.at(0).isInt32() for Range Operator](https://stackoverflow.com/questions/76926155/assertion-failed-inputs-at0-isint32-for-range-operator-with-dynamic-inp)

---

## Bugs

(You can add relevant bugs here.)
