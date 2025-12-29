import subprocess
import re
import numpy as np
import sys
import os
import pandas as pd
import time
from draw_pt_line import plot_pt

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout, stderr, process.returncode

def parse_region_prob(output):
    # Look for "区域概率序列 (%):"
    # The output might be spread across multiple lines if it's a large array
    # Example:
    # 区域概率序列 (%): [ 0.  0.  0. 10. 50. ...]
    
    match = re.search(r"区域概率序列 \(%\): \[(.*?)\]", output, re.DOTALL)
    if match:
        content = match.group(1)
        # Clean up newlines and extra spaces
        content = content.replace('\n', ' ')
        # Parse numbers
        try:
            probs = [float(x) for x in content.split() if x.strip()]
            return probs
        except ValueError:
            return None
    return None

def calculate_center_frame(entry_frame):
    # Find largest number <= entry_frame ending in 1
    # (entry_frame - 1) // 10 * 10 + 1
    return (entry_frame - 1) // 10 * 10 + 1

def main():
    # Read red_light_jumps.csv
    try:
        jumps_df = pd.read_csv("red_light_jumps.csv")
    except FileNotFoundError:
        print("red_light_jumps.csv not found. Please run find_red_light_jumps.py first.")
        return

    results = []
    
    print(f"Processing {len(jumps_df)} red light jumps...")
    
    for idx, row in jumps_df.iterrows():
        track_id = int(row['track_id'])
        entry_frame = int(row['entry_frame'])
        
        center_frame = calculate_center_frame(entry_frame)

        # center_frame -= 20
        
        print(f"Processing Track {track_id}, Entry {entry_frame} -> Center {center_frame}...", end=' ')
        
        # 1. Run data process with track_id
        cmd_process = f"{sys.executable} red_light_jump_data_process_6.py --center_frame {center_frame} --track_id {track_id}"
        out_p, err_p, code_p = run_command(cmd_process)
        
        if code_p != 0:
            print(f"Error in data process: {err_p}")
            continue
            
        if "No data generated." in out_p:
            print("No data generated.")
            continue
            
        if "[警告] 车辆" in out_p and "程序终止" in out_p:
            print("Skipped (Vehicle in crossing)")
            continue

        # 2. Run data preprocessing
        cmd_data = f"{sys.executable} src/data/TP/process_data.py"
        out_d, err_d, code_d = run_command(cmd_data)
        
        if code_d != 0:
            print(f"Error in process_data: {err_d}")
            continue

        # 3. Run validation/prediction
        cmd_val = f"{sys.executable} src/main_5.py"
        out_v, err_v, code_v = run_command(cmd_val)
        
        if code_v != 0:
            print(f"Error in main_5.py: {err_v}")
            continue
        
        # 4. Parse results
        probs = parse_region_prob(out_v)
        
        if probs is not None:
            print("Success")
            results.append({
                "track_id": track_id,
                "entry_frame": entry_frame,
                "center_frame": center_frame,
                "region_prob_seq": probs
            })
        else:
            print("Could not parse region probabilities")

    # Save results
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv("red_light_jump_probs.csv", index=False)
        print(f"\nSaved results for {len(results)} tracks to red_light_jump_probs.csv")
        print(res_df)
    else:
        print("\nNo valid results collected.")

if __name__ == "__main__":
    main()
    plot_pt()
