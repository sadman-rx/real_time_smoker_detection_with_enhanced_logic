# Real-Time Smoker Detection with Enhanced Logic

This repository provides a real-time smoking detection system featuring enhanced multi-signal logic. It delivers two deployment tracks:

- Desktop/High‑end devices (Python + PyTorch/Ultralytics + TensorFlow MoveNet via TF Hub)
- Raspberry Pi/Edge devices (Picamera2 + ONNX Runtime + TFLite Runtime)

The system fuses object detection (YOLO-based smoker/non-smoker/cigarette classes) with human pose estimation (MoveNet multipose) and custom post-processing logic to robustly infer smoking behavior.


## Key Features

- Dual deployment targets:
  - Desktop apps: real-time webcam pipeline and a drag-and-drop GUI for static images.
  - Edge device apps: Raspberry Pi 4 Model B with Pi Camera v3.
- Multi-model fusion:
  - YOLO model for person/cigarette detection.
  - MoveNet multipose for pose keypoints (nose, wrists, elbows, shoulders, etc.).
  - Landmark-to-person association and helper weights combining YOLO confidence and landmark proximity to cigarette.
- Enhanced logic for robust labeling:
  - Final labels: `confident_smoker`, `possible_smoker`, `nonSmoker`, plus separate `cigarette` boxes.
  - Non-maximum suppression (NMS) for stable bounding boxes.
  - Overlap- and distance-based assignment of cigarettes to the most plausible person.
  - Gesture cues (on edge variants) using relative distances and elbow angles near the face.
- Optimized runtimes:
  - Desktop uses Ultralytics YOLO with PyTorch weights and TF Hub MoveNet.
  - Edge uses ONNX Runtime for YOLO and TFLite Runtime for MoveNet with Picamera2.
- Usability:
  - Temp file management, threaded processing, FPS/metrics overlay, optional saving of smoker frames (edge UI demo).


## Repository Structure

```
desktop_applications/
  improved_smoking_detection2_webcam.py        # Real-time webcam detection (desktop)
  smoker_detection_with_gui_for_static_image.py# Tkinter GUI for static images (desktop)
  smoking_detection.py                         # Core desktop pipeline for static images

edge_device/
  smoker_detection_management_system_demo.py   # Tkinter-based Pi management/demo UI, saves smoker frames
  smoker_detection_v2_raspberry_pi_pipeline.py # Real-time Pi pipeline with Picamera2

models/
  altf4_v2_smoking_detection_model.onnx        # YOLO model for edge (ONNX)
  movenet_multipose_lightning.tflite           # MoveNet multipose for edge (TFLite)
  trained_model_SD_large_v2_noiseFree.pt       # YOLO model for desktop (PyTorch)

Training/
  SD_YOLOV8_LARGE_V2_noiseFreeFinal.ipynb      # Training notebook (YOLOv8 Large)
```


## Models

- YOLO classes (as used in code):
  - `0 = cigarette`, `1 = non_smoker`, `2 = smoker`.
- Desktop:
  - YOLO (PyTorch) weights: `models/trained_model_SD_large_v2_noiseFree.pt`
  - Pose: MoveNet Multipose Lightning from TF Hub (downloaded at runtime)
- Edge (Raspberry Pi):
  - YOLO (ONNX): `models/altf4_v2_smoking_detection_model.onnx`
  - Pose: `models/movenet_multipose_lightning.tflite`


## How It Works (High Level)

1. YOLO detects persons and cigarettes.
2. MoveNet multipose produces pose keypoints per person.
3. Landmarks (notably nose, wrists, elbows, shoulders) are associated to detected person boxes.
4. Cigarettes are assigned to the most plausible person using overlap and distance to key landmarks; a helper weight combines YOLO confidence with landmark proximity.
5. Final label per person:
   - `confident_smoker` if strong agreement (e.g., unique cigarette overlap or YOLO+pipeline agree).
   - `possible_smoker` if signals disagree but indicate potential smoking.
   - `nonSmoker` if neither signal indicates smoking.
6. An annotated image/frame is rendered with colored boxes and labels. Edge UI optionally saves frames with smokers.


## Demo Video

Watch a presentation describing how the system works:

- [![Watch the video](https://img.youtube.com/vi/Cggbm7x8ayg/hqdefault.jpg)](https://youtu.be/Cggbm7x8ayg)
- Video link: https://youtu.be/Cggbm7x8ayg


## Environment Setup

### 1) Desktop/High‑end Device

- Tested with: Python 3.12.7 on Windows (desktop)
- Suggested packages:
  - ultralytics, torch, torchvision, torchaudio (per your CUDA/CPU setup)
  - tensorflow>=2.11, tensorflow-hub
  - opencv-python, pillow, numpy, matplotlib
  - tkinter (usually included with Python on Windows/macOS; install via system package manager on Linux)
  - Optional: tkinterdnd2 (for drag-and-drop in the GUI)

Example setup (CPU-only, Windows Powershell):

```
python -m venv .venv
.venv\Scripts\activate
pip install ultralytics==8.* opencv-python pillow numpy matplotlib tensorflow tensorflow-hub tkinterdnd2
# If you need PyTorch (CPU only example):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Note: Choose the correct PyTorch build for your GPU/CPU from the official website if using CUDA.


### 2) Raspberry Pi 4 Model B (Edge)

Hardware: Raspberry Pi 4 Model B + Pi Camera v3.

System prerequisites (Bookworm/Bullseye):
- libcamera and Picamera2
- Python 3 and venv

Install dependencies (typical commands):

```
sudo apt update
sudo apt install -y python3-pip python3-venv python3-tk python3-picamera2 libatlas-base-dev
python3 -m venv ~/sd-edge-venv
source ~/sd-edge-venv/bin/activate
pip install numpy pillow opencv-python screeninfo
pip install onnxruntime  # or onnxruntime==1.17.* (ARM wheels available)
pip install tflite-runtime  # Use Raspberry Pi compatible wheels
```

Notes:
- tflite-runtime is preferred over full TensorFlow on Pi for performance.
- If `onnxruntime` pip install fails, install the appropriate aarch64/armv7 wheel from ONNX Runtime releases.
- Picamera2 may already be installed via `sudo apt install python3-picamera2`.

Tested with: Python 3.9.2 on Raspberry Pi OS (edge device)


## Running the Applications

### Desktop

1) Static image GUI (drag-and-drop or file browser):

```
python desktop_applications/smoker_detection_with_gui_for_static_image.py
```

- Loads models at startup and lets you browse or drag in an image.
- Produces annotated results in the right panel.

2) Real-time webcam pipeline:

```
python desktop_applications/improved_smoking_detection2_webcam.py
```

- Runs YOLO + MoveNet in parallel threads and renders annotated frames.
- Creates a `temp/` directory to store transient frames for MoveNet input.

3) Core static image pipeline (static image):

By default, `desktop_applications/smoking_detection.py` uses a hardcoded image path `test.png` in `main()`. You can:

- Run it as-is (ensure `test.png` is present next to where you run the command):

```
python desktop_applications/smoking_detection.py
```

- Or import and call `detect_smoking` programmatically with your own path:

```
from desktop_applications.smoking_detection import load_models, detect_smoking

object_model, landmark_model = load_models()
annotated_image, results = detect_smoking("path/to/your_image.jpg", object_model, landmark_model, save_output=True)
```

- When `save_output=True`, the script writes `output_<image_name>` next to the source.
- This module is also imported by the GUI.


### Raspberry Pi / Edge

1) Streamlined real-time pipeline with Picamera2:

```
python edge_device/smoker_detection_v2_raspberry_pi_pipeline.py
```

- Initializes Pi Camera v3, loads ONNX/TFLite models, and displays annotated frames. 
- Falls back gracefully and performs temp file cleanup on start.

2) Management system demo UI (Tkinter):

```
python edge_device/smoker_detection_management_system_demo.py
```

- GUI to start/stop detection and view logs.
- Saves smoker frames with readable timestamps under `edge_device/saved_smoker_frames/`.
- Limits queue size to reduce lag; includes warmup passes and multi-threaded pipeline.


## Training

- The `Training/SD_YOLOV8_LARGE_V2_noiseFreeFinal.ipynb` notebook documents training the YOLOv8L model used for detection.
- Exported for two targets:
  - Desktop: PyTorch `.pt` weights
  - Edge: ONNX model for ONNX Runtime

## Dataset

The dataset is not being published due to the presence of uncensored sensitive data. Summary statistics:

- Total instances: nonSmoker 1709, smoker 1655, cigarette 1655
- Training:        nonSmoker 1208, smoker 1165, cigarette 1165
- Validation:      nonSmoker 339,  smoker 330,  cigarette 330
- Test:            nonSmoker 162,  smoker 160,  cigarette 160


## Tips and Troubleshooting

- Performance:
  - No specific FPS claims; performance depends on device capabilities and configuration.
  - Desktop: ensure correct PyTorch build (CPU vs CUDA). Close other heavy apps.
  - Pi: prefer `tflite-runtime` and `onnxruntime` CPU EP; reduce camera resolution if needed.
- Model paths:
  - All scripts reference models via relative paths under `models/`. Keep folder structure intact.
- GUI requirements:
  - Desktop GUI optionally uses `tkinterdnd2` for drag-and-drop; falls back if unavailable.
  - Edge demo uses Tkinter; make sure `python3-tk` is installed.
- Temp files:
  - Desktop webcam and edge pipelines create a `temp/` folder and clean it up periodically/startup.
- Classes:
  - Ensure your own training maintains class ordering: `0=cigarette, 1=non_smoker, 2=smoker`.
- Permissions:
  - On Raspberry Pi, ensure your user has access to the camera and that `libcamera-hello` works.


## Roadmap / Possible Extensions

- Lightweight post-processing variants for lower-latency edge.
- Broader gesture modeling and temporal smoothing across frames.
- MQTT/HTTP alerts, cloud logging, and remote management.
- Dockerized desktop app and systemd service for Pi.


## Safety and Ethical Use

This system is intended for research and demonstration purposes. If deploying in public or semi-public spaces, comply with all local laws, institutional policies, and privacy regulations. Inform users and bystanders about video analytics as required.


## Acknowledgements

- Ultralytics YOLOv8
- Google MoveNet (TF Hub / TFLite)
- Raspberry Pi, Picamera2, ONNX Runtime, TFLite Runtime


## License

Licensed under the MIT License.

You may copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the terms of the MIT License. Consider adding a `LICENSE` file with the full text for clarity.


## Contact

Primary contacts:

- Sadman Hossain — sadmanhossainwork@gmail.com — https://www.linkedin.com/in/sadmanhossain-in/ — https://www.fiverr.com/s/yv157kV
- Arnab Banik — official.arnab.b@gmail.com
- Abdullah Al Mukit — dymmukit5824@gmail.com — https://www.linkedin.com/in/abdullah-al-mukit-01b865353/

For issues: please open a GitHub Issue in this repository.
