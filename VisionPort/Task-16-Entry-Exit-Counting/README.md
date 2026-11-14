# Task 16 – Entry and Exit Vehicle Count

**Objective:** Count tracked cars, motorcycles, buses and trucks crossing a virtual line at a junction.

## Run

From this folder:

```powershell
py count_entry_exit.py
```

Enter the path to a traffic video when asked. Press `Q` to stop. The annotated video is saved to `outputs/entry_exit_count.mp4`.

## Camera setup

The default counting line is vertical at the centre of the frame. Edit `LINE_POSITION_RATIO` near the top of the script to move it (0.0 = left edge, 1.0 = right edge). The example assumes left-to-right means **entering** and right-to-left means **leaving**. Reverse the labels/conditions if your camera faces the other direction.

Each tracked vehicle is counted once. Tracking IDs can change when vehicles are hidden or leave the camera view, so results are approximate and should be checked with a fixed camera view.
