#!/bin/bash

threshold=${1:-20}

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r index used total; do
    current_hour=$(date +%H)
    # if [ "$current_hour" -ge 8 ] && [ "$current_hour" -lt 23 ]; then
    # if [ "$index" -eq 2 ] || [ "$index" -eq 3 ]; then
    if [ "$index" -eq 3 ]; then
        continue
    fi
    # fi
    usage=$(echo "scale=2; $used / $total * 100" | bc)
    if (( $(echo "$usage < $threshold" | bc -l) )); then
        if ! nvidia-smi --query-compute-apps=gpu,pid --format=csv,noheader | grep -q "^$index,"; then
            echo "$index" 
            break
        fi
    fi
done