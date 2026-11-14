# Task 17 – Wrong-Turn Detection

**Objective:** Follow vehicle paths through an intersection and alert when the detected turn is not in the configured allowed-turn list.

## Run

```powershell
py detect_wrong_turns.py
```

Enter an intersection video path. Press `Q` to stop. Output: `outputs/wrong_turn_detection.mp4`.

## Important configuration

- `ROI` defines the intersection rectangle as frame ratios. The default is the centre half of the image; adjust it to cover your junction.
- `ALLOWED_TURNS` is an example for one incoming lane. Set it to the turns permitted for the lane you are analysing, such as `{"STRAIGHT", "RIGHT"}`.
- LEFT/RIGHT are screen-relative, not guaranteed road-direction labels. Camera orientation and the chosen approach lane matter.

This is a demonstration based on short tracked trajectories. For reliable enforcement, a fixed calibrated camera, lane-specific movement paths, and validation against labelled junction footage are required. Tracking loss or occlusion can produce UNKNOWN or incorrect results.
