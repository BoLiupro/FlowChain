#!/bin/bash
set -e

echo "========== Step 1: Running red_light_jump_data_process_6.py =========="
python /root/workspace/FlowChain-ICCV2023/red_light_jump_data_process_5.py

echo "========== Step 2: Running process_data.py =========="
python /root/workspace/FlowChain-ICCV2023/src/data/TP/process_data.py

echo "========== Step 3: Running main_5.py =========="
python /root/workspace/FlowChain-ICCV2023/src/main_4.py

# echo "========== Step 4: Running plot.py =========="
# python /root/workspace/FlowChain-ICCV2023/plot/plot.py

echo "========== ALL DONE =========="
