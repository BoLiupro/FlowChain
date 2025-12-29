import os
import glob
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from tqdm import tqdm

# Constants
W, H = 20, 10
CROSSING_POLY = np.array([
    [0.3 * W, 0.18 * H],
    [0.5 * W, 0.17 * H],
    [0.52 * W, 0.34 * H],
    [0.35 * W, 0.35 * H]
])
CROSSING_POLY_SHAPELY = Polygon(CROSSING_POLY)

DATA_DIR = "red_data/red_full"
LIGHT_FILE = os.path.join(DATA_DIR, "light_3.txt")

def load_light_data():
    df = pd.read_csv(LIGHT_FILE)
    # Filter for Red light
    # Assuming 'Color' column exists and 'Red' is the value
    # The file content shows: Frame,Class,Color,Conf,x1,y1,x2,y2
    # 0,traffic light,Red,0.7008,...
    
    # We need to know which frames are Red.
    # There might be multiple traffic lights detected per frame.
    # If ANY traffic light is Red, we consider the frame as Red? 
    # Or maybe there is only one relevant traffic light.
    # For now, if any row for a frame says Red, we mark it Red.
    
    red_frames = set(df[df['Color'] == 'Red']['Frame'].unique())
    return red_frames

def load_trajectories():
    records = []
    files = glob.glob(os.path.join(DATA_DIR, "output_blurred_*.txt"))
    
    print(f"Loading {len(files)} trajectory files...")
    for fname in tqdm(files):
        basename = os.path.basename(fname)
        try:
            frame_id = int("".join(filter(str.isdigit, basename)))
        except ValueError:
            continue
            
        with open(fname, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue

                # Format: cls xc yc w h tid
                cls, xc, yc, w, h, tid = parts
                tid = int(float(tid))
                xc, yc, w, h = map(float, (xc, yc, w, h))
                
                # Convert coordinates
                # Using logic from red_light_jump_data_process_7.py
                if cls == "0":  # Pedestrian
                    x = xc * W
                    y = H - (yc + h / 2) * H
                else:
                    continue
                
                records.append([frame_id, tid, x, y])
                
    df = pd.DataFrame(records, columns=["frame_id", "track_id", "x", "y"])
    df.sort_values(["track_id", "frame_id"], inplace=True)
    return df

def analyze_jumps(traj_df, red_frames):
    results = []
    
    for tid, group in tqdm(traj_df.groupby("track_id"), desc="Analyzing tracks"):
        group = group.sort_values("frame_id")
        frames = group["frame_id"].values
        xs = group["x"].values
        ys = group["y"].values
        
        # Check intersection with polygon
        # We can vectorize this or loop
        inside_mask = []
        for x, y in zip(xs, ys):
            p = Point(x, y)
            inside_mask.append(CROSSING_POLY_SHAPELY.contains(p))
        
        inside_mask = np.array(inside_mask)
        
        # Find entry point: inside[i] is True, inside[i-1] is False
        # We need at least 2 points
        if len(inside_mask) < 2:
            continue
            
        # Identify indices where transition happens
        # diff: True if status changed
        # We want False -> True
        
        # Prepend False to handle case where it starts inside? 
        # User says "entered", implies it was outside before.
        
        for i in range(1, len(inside_mask)):
            if inside_mask[i] and not inside_mask[i-1]:
                # Entered at index i (frame frames[i])
                entry_frame = frames[i]
                prev_frame = frames[i-1]
                
                # Check if light was Red at entry_frame
                if entry_frame in red_frames:
                    # Found a jump
                    # center_frame is the previous time step
                    # User says "10hz corresponds to 0.1s before".
                    # If our frames are consecutive, prev_frame is the one.
                    # If frames are not consecutive (missing detections), we should be careful.
                    # But "previous time step" usually implies the data point immediately preceding.
                    
                    center_frame = prev_frame
                    results.append({
                        "track_id": tid,
                        "center_frame": center_frame,
                        "entry_frame": entry_frame
                    })
                    
                    # We only care about the first entry? Or all?
                    # Usually "jump" is a single event. Let's record it and maybe break if we only want one per track.
                    # But a track could enter, leave, and enter again.
                    # Let's keep all for now, or just the first one.
                    # "filter out ... track_id" implies the track is the unit.
                    break 
                    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Loading light data...")
    red_frames = load_light_data()
    print(f"Found {len(red_frames)} red light frames.")
    
    print("Loading trajectory data...")
    traj_df = load_trajectories()
    print(f"Loaded {len(traj_df)} trajectory points.")
    
    print("Analyzing jumps...")
    jumps_df = analyze_jumps(traj_df, red_frames)
    
    print(f"Found {len(jumps_df)} red light jumps.")
    print(jumps_df)
    
    jumps_df.to_csv("red_light_jumps.csv", index=False)
