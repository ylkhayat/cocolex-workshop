#!/bin/bash

threshold=10
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r index used total; do
    if [ "$index" -eq 0 ] || [ "$index" -eq 1 ]; then
        continue
    fi
    # Calculate GPU memory usage percentage
    usage=$(echo "scale=2; $used / $total * 100" | bc)
    # Check if GPU usage is below the threshold
    if (( $(echo "$usage < $threshold" | bc -l) )); then
        # Check if there are any active processes on the GPU
        if ! nvidia-smi --query-compute-apps=gpu,pid --format=csv,noheader | grep -q "^$index,"; then
            # GPU is available
            echo "$index"  # Output GPU index for use in other scripts
            break
        fi
    fi
done