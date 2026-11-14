# VisionPort – Individual Deep Learning Tasks

**Project:** Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning

This repository contains separate, independently runnable programs for the tasks in the instructor's task sheet. Task 01 (environment installation) is intentionally left without an implementation, as requested. Work starts with Task 02.

## Setup

Use Python and PowerShell. From this `VisionPort` folder, install the shared libraries:

```powershell
py -m pip install -r requirements.txt
```

The scripts use `py` as the Windows Python launcher. Open PowerShell in the relevant task folder and run the script shown in that folder's README.

For YOLO tasks, the first run may download the pretrained `yolov8n.pt` model. An internet connection may be needed for this download. Do not manually download model files unless required.

## Task list

| Folder | Instructor task | Expected result |
|---|---|---|
| Task-01-Environment | Install Python, OpenCV and required libraries | Working Python environment (skipped) |
| Task-02-Camera-Capture | Capture an image of a traffic signal using a camera | Traffic image capture |
| Task-03-Image-Display | Read and display traffic images using OpenCV | Image display window |
| Task-04-Video-Playback | Load and play a traffic video using OpenCV | Video playback |
| Task-05-Frame-Extraction | Extract individual frames from traffic videos | Frames saved as images |
| Task-06-Vehicle-Detection | Detect cars in a traffic video using YOLO | Car detection with bounding boxes |
| Task-07-Vehicle-Classification | Detect bikes, buses and trucks | Multiple vehicle detection |
| Task-08-Confidence-Scores | Display vehicle names and confidence scores | Labelled detection output |
| Task-09-Vehicle-Counting | Count the number of vehicles in a video frame | Vehicle count |
| Task-10-Traffic-Density | Calculate traffic density based on vehicle count | Low, medium and high traffic |
| Task-11-Congestion-Detection | Detect traffic congestion at signal junctions | Congestion alerts |
| Task-12-Vehicle-Tracking | Track individual vehicles across video frames | Vehicle tracking IDs |
| Task-13-Speed-Estimation | Calculate approximate vehicle speed | Speed estimation |
| Task-14-Traffic-Signal-Detection | Identify traffic signal status (red, yellow, green) | Signal status detection |
| Task-15-Waiting-Time | Estimate traffic waiting time | Waiting-time report |
| Task-16-Entry-Exit-Counting | Count vehicles entering and leaving a traffic junction | Entry and exit vehicle count |
| Task-17-Wrong-Turn-Detection | Detect vehicles making wrong turns at intersections | Wrong-turn detection |
| Task-18-Wrong-Way-Detection | Identify vehicles moving in the wrong direction | Wrong-way vehicle alerts |
| Task-19-Illegal-Parking | Detect vehicles parked in restricted traffic areas | Illegal parking detection |
| Task-20-Triple-Riding-Detection | Detect two-wheelers carrying more than two persons | Triple-riding detection |

## Important notes

- Use real traffic images/videos for meaningful results. The webcam task needs a working camera.
- Vehicle detection uses the pretrained YOLOv8 Nano model and COCO class IDs: car (2), motorcycle (3), bus (5), truck (7), and traffic light (9).
- Traffic density and congestion thresholds are simple demonstration settings. They should be adjusted for the camera and road scene.
- Speed estimation requires a measured meters-per-pixel calibration for the camera view. Without calibration, pixel movement cannot be converted reliably into km/h.
- Waiting-time estimation is approximate. It uses tracked vehicle movement and a stationary-pixel threshold; camera movement and tracking errors affect the result.
- Traffic-signal colour detection is an approximate colour check inside a YOLO-detected traffic-light box. It may return `UNKNOWN` when the light is small, blurry, or poorly lit.
- Tasks 16–20 are additional demonstrations. They use configurable virtual lines, intersection zones, direction assumptions, and rider-region estimates; configure these for the supplied camera/video before relying on their output.
- The scripts save results under their task folder (usually an `outputs` or `extracted_frames` folder). Generated videos and images are not included in this starter ZIP.

## GitHub workflow

Review and run each task before committing it. Add only the files for the task you have checked, then commit and push it separately. Do not commit large source videos or model-weight files.
