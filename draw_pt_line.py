import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_pt():
    csv_path = 'red_light_jump_probs.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    output_dir = '/root/workspace/FlowChain-ICCV2023/plot/pt_curves'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    for idx, row in df.iterrows():
        track_id = int(row['track_id'])
        entry_frame = int(row['entry_frame'])
        probs_str = row['region_prob_seq']
        
        try:
            # Convert string representation of list to actual list
            # Handle potential spaces and newlines
            clean_str = probs_str.strip('[]').replace('\n', ' ')
            if not clean_str.strip():
                continue
            probs = [float(x) for x in clean_str.split(',')]
            
            plt.plot(probs, marker='o', label=f'Track {track_id} (Entry {entry_frame})')
        except ValueError as e:
            print(f"Error parsing probabilities for track {track_id}: {e}")
            continue

    plt.title('Red Light Jump Probability Curves')
    plt.xlabel('Time Step')
    plt.ylabel('Probability (%)')
    plt.ylim(0, 105)
    plt.grid(True)
    
    # Adjust legend to be outside if there are many tracks
    if len(df) > 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
    else:
        plt.legend()
    
    save_path = os.path.join(output_dir, 'all_tracks_pt_curve.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {save_path}")

if __name__ == "__main__":
    plot_pt()
