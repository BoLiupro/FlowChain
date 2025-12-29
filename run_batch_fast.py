import sys
import os
import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from copy import deepcopy
import time

# Add src to path so we can import modules from it
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import FlowChain modules
from utils import load_config
from data.unified_loader import unified_loader
from models.build_model import Build_Model
from metrics.build_metrics import Build_Metrics
from visualization.build_visualizer import Build_Visualizer

# Import data processing module
import red_light_jump_data_process_5 as data_process

def parse_args():
    parser = argparse.ArgumentParser(description="Fast batch processing")
    parser.add_argument("--config_file", type=str, default='config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth.yml',
                        metavar="FILE", help='path to config file')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--start", type=int, default=20)
    parser.add_argument("--end", type=int, default=4300)
    parser.add_argument("--visualize", action='store_true', default=False)
    parser.add_argument("--mode", type=str, default="test")
    # Add other args expected by load_config
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args)
    
    # 1. Load Model Once
    print("Loading model...")
    model = Build_Model(cfg)
    try:
        model.load()
    except FileNotFoundError:
        print("Error: No model saved found!")
        return
    model.eval()
    
    metrics_calc = Build_Metrics(cfg)
    
    # 2. Load Data Source Once
    print("Loading raw data...")
    folder = "/root/workspace/FlowChain-ICCV2023/red_data/red_full"
    df_raw = data_process.load_ultralytics(folder)
    df_2_5hz = data_process.resample_to_2_5hz(df_raw)
    
    # Output path for the intermediate file
    out_dir = "/root/workspace/FlowChain-ICCV2023/src/data/TP/raw_data/zara2/test"
    out_file = f"{out_dir}/converted_tracks.txt"
    os.makedirs(out_dir, exist_ok=True)
    
    ade_list = []
    fde_list = []
    
    print(f"Starting fast batch processing from {args.start} to {args.end}...")
    
    # 3. Loop
    for center_frame in tqdm(range(args.start, args.end + 1)):
        # --- Data Generation ---
        # We can call the functions directly
        df_final = data_process.build_observe_predict(df_2_5hz, center_frame)
        
        if df_final.empty:
            # Vehicle warning or no data
            continue
            
        df_final = data_process.smooth_tracks(df_final)
        
        if df_final.empty:
            continue
            
        df_final = df_final.sort_values(["frame_id", "track_id"])
        
        # Write to file (Data Loader expects this)
        df_final.to_csv(out_file, sep="\t", index=False, header=False)
        
        # --- Inference ---
        # Re-initialize data loader for the new file
        # unified_loader reads the file we just wrote
        # We use split="test" or "train" depending on what main_4.py used. 
        # main_4.py used split="test" in test() function.
        # But wait, the file path we wrote to is .../eth/train/converted_tracks.txt
        # unified_loader logic:
        # if split == 'train': path = .../train_data.pkl (usually)
        # But here we are using a raw text file?
        # Let's check how unified_loader works.
        # In main_4.py: data_loader = unified_loader(cfg, rand=False, split="test")
        # If we look at TP_metrics.py, it loads "processed_data/{dataset}_train.pkl" for env.
        # But the data loader loads the actual trajectories.
        
        # Assuming unified_loader reads from the location we wrote to if we configure it right.
        # The original script wrote to: src/data/TP/raw_data/eth/train/converted_tracks.txt
        # And main_4.py ran with default config.
        # Let's assume unified_loader(..., split="test") reads that file or processes it.
        # Actually, usually raw data needs preprocessing into pkl.
        # Does main_4.py do preprocessing on the fly?
        # If main_4.py worked in the slow batch script, it means it worked.
        # The slow script ran `python src/main_4.py`.
        # `src/main_4.py` calls `unified_loader(cfg, rand=False, split="test")`.
        
        # We need to be careful. If unified_loader caches data, we might be in trouble.
        # But usually for "test" split, it might read directly or we might need to clear cache.
        
        try:
            # We instantiate a new loader every time to ensure it reads the new file
            # This is still much faster than reloading the model
            data_loader = unified_loader(cfg, rand=False, split="test")
            
            # Run inference
            with torch.no_grad():
                for data_dict in data_loader:
                    data_dict = {k: data_dict[k].cuda()
                                 if isinstance(data_dict[k], torch.Tensor)
                                 else data_dict[k]
                                 for k in data_dict}
                    
                    # Predict
                    result_dict = model.predict(deepcopy(data_dict), return_prob=True)
                    
                    # Metrics
                    # We need to wrap result in list as metrics expects list
                    dict_list = [deepcopy(result_dict)]
                    dict_list = metrics_calc.denormalize(dict_list)
                    
                    # Calculate metrics
                    res = metrics_calc(dict_list)
                    
                    ade = res['ade']
                    fde = res['fde']
                    
                    # ade/fde might be numpy arrays or scalars
                    if np.ndim(ade) == 0:
                        ade_list.append(float(ade))
                        fde_list.append(float(fde))
                    else:
                        # If batch size > 1 (unlikely here), mean it
                        ade_list.append(float(np.mean(ade)))
                        fde_list.append(float(np.mean(fde)))
                        
        except Exception as e:
            print(f"Error processing frame {center_frame}: {e}")
            import traceback
            traceback.print_exc()
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
