import subprocess
import re
import numpy as np
import sys
import os
import time

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout, stderr, process.returncode

def parse_metrics(output):
    # Look for the dictionary print
    # We look for 'ade': <float> and 'fde': <float>
    # Example: {'score': 0.5, 'ade': 0.2, 'fde': 0.4, ...}
    
    # Use regex to find the dictionary or specific keys
    ade_match = re.search(r"'ade':\s*([0-9.]+)", output)
    fde_match = re.search(r"'fde':\s*([0-9.]+)", output)
    
    if ade_match and fde_match:
        return float(ade_match.group(1)), float(fde_match.group(1))
    return None, None

def main():
    ade_list = []
    fde_list = []
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=81)
    parser.add_argument('--end', type=int, default=4301)
    args = parser.parse_args()
    
    start_frame = args.start
    end_frame = args.end
    
    print(f"Starting batch processing from frame {start_frame} to {end_frame}...")
    
    for center_frame in range(start_frame, end_frame + 1, 10):
        # print(f"Processing frame {center_frame}...", end='\r')
        
        # 1. Run data process
        cmd_process = f"{sys.executable} red_light_jump_data_process_6.py --center_frame {center_frame}"
        out_p, err_p, code_p = run_command(cmd_process)
        
        if "[警告] 车辆" in out_p and "程序终止" in out_p:
            # Vehicle entered crossing, skip this frame
            # print(f"Frame {center_frame}: Skipped (Vehicle in crossing)")
            continue
            
        if "No data generated." in out_p:
            # print(f"Frame {center_frame}: No data generated.")
            continue
            
        if code_p != 0:
            print(f"Frame {center_frame}: Error in data process. Stderr: {err_p}")
            continue
            
        # Check if data was actually generated (sometimes it returns empty DataFrame)
        # The script prints nothing if successful, but we can check if it exited early?
        # If df is empty, it returns pd.DataFrame() and exits?
        # Let's check the script again. It returns pd.DataFrame() but doesn't print anything special.
        # But if len(all_rows) == 0, it returns empty df.
        # Then df_final is empty.
        # Then to_csv writes an empty file?
        # If empty file, main_4.py might fail.
        cmd_val = f"{sys.executable} src/data/TP/process_data.py"
        out_v, err_v, code_v = run_command(cmd_val)
        
        # 2. Run validation
        cmd_val = f"{sys.executable} src/main_4.py"
        out_v, err_v, code_v = run_command(cmd_val)
        
        if code_v != 0:
            # print(f"Frame {center_frame}: Error in validation. Stderr: {err_v}")
            # This often happens if data is empty or invalid
            continue
        
        # 3. Parse results
        ade, fde = parse_metrics(out_v)
        
        if ade is not None and fde is not None:
            print(f"Frame {center_frame}: ADE={ade:.4f}, FDE={fde:.4f}")
            ade_list.append(ade)
            fde_list.append(fde)
        else:
            # print(f"Frame {center_frame}: Could not parse metrics")
            pass

    if ade_list:
        avg_ade = np.mean(ade_list)
        avg_fde = np.mean(fde_list)
        print(f"\nFinal Results over {len(ade_list)} valid frames:")
        print(f"Average ADE: {avg_ade:.4f}")
        print(f"Average FDE: {avg_fde:.4f}")
    else:
        print("\nNo valid results collected.")

if __name__ == "__main__":
    main()
