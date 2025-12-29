import subprocess
import re
import numpy as np
import sys
import os
import time
import shutil
import multiprocessing
import argparse

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout, stderr, process.returncode

def parse_metrics(output):
    ade_match = re.search(r"'ade':\s*([0-9.]+)", output)
    fde_match = re.search(r"'fde':\s*([0-9.]+)", output)
    
    if ade_match and fde_match:
        return float(ade_match.group(1)), float(fde_match.group(1))
    return None, None

def process_frame(center_frame):
    pid = os.getpid()
    # Create unique temporary directories
    base_temp_dir = os.path.abspath(f"temp_data_{pid}_{center_frame}")
    raw_dir = os.path.join(base_temp_dir, "TP", "raw_data", "zara2")
    processed_dir = os.path.join(base_temp_dir, "TP", "processed_data")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        # 1. Run data process
        # Output to raw_dir
        # Note: red_light_jump_data_process_5.py appends /train, /test, /val to the output dir
        # So we pass raw_dir which is .../zara2
        cmd_process = f"{sys.executable} red_light_jump_data_process_6.py --center_frame {center_frame} --output_dir {raw_dir}"
        out_p, err_p, code_p = run_command(cmd_process)
        
        if "[警告] 车辆" in out_p and "程序终止" in out_p:
            return center_frame, None, None, "Skipped (Vehicle in crossing)"
            
        if "No data generated." in out_p:
            return center_frame, None, None, "No data generated"
            
        if code_p != 0:
            return center_frame, None, None, f"Error in data process: {err_p}"
            
        # 2. Run process_data.py
        # Input: base_temp_dir/TP/raw_data
        # Output: processed_dir
        raw_input_path = os.path.join(base_temp_dir, "TP", "raw_data")
        cmd_data = f"{sys.executable} src/data/TP/process_data.py --raw_path {raw_input_path} --processed_path {processed_dir}"
        out_d, err_d, code_d = run_command(cmd_data)
        
        if code_d != 0:
             return center_frame, None, None, f"Error in process_data: {err_d}"

        # 3. Run validation
        # DATA.PATH should point to base_temp_dir (which contains TP/processed_data)
        # Note: unified_loader looks for {DATA.PATH}/{DATA.TASK}/processed_data
        # So if DATA.PATH is base_temp_dir, it looks for base_temp_dir/TP/processed_data. Correct.
        cmd_val = f"{sys.executable} src/main_4.py DATA.PATH {base_temp_dir}"
        out_v, err_v, code_v = run_command(cmd_val)
        
        if code_v != 0:
            return center_frame, None, None, f"Error in validation: {err_v}"
        
        # 4. Parse results
        ade, fde = parse_metrics(out_v)
        
        if ade is not None and fde is not None:
            return center_frame, ade, fde, "Success"
        else:
            return center_frame, None, None, "Could not parse metrics"

    finally:
        # Cleanup
        if os.path.exists(base_temp_dir):
            shutil.rmtree(base_temp_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=81)
    parser.add_argument('--end', type=int, default=4301)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    
    start_frame = args.start
    end_frame = args.end
    workers = args.workers
    
    frames = range(start_frame, end_frame + 1, 10)
    
    print(f"Starting batch processing from frame {start_frame} to {end_frame} with {workers} workers...")
    
    ade_list = []
    fde_list = []
    
    with multiprocessing.Pool(processes=workers) as pool:
        for center_frame, ade, fde, msg in pool.imap_unordered(process_frame, frames):
            if ade is not None:
                print(f"Frame {center_frame}: ADE={ade:.4f}, FDE={fde:.4f}")
                ade_list.append(ade)
                fde_list.append(fde)
            else:
                print(f"Frame {center_frame}: {msg}")
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
