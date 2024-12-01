#!/bin/bash

cd "$(dirname "$0")" || exit

. ./dynamic_input_handler.sh --required dataset,setup,split,device --method rag "$@"

if [[ $? -ne 0 ]]; then
    exit 1
fi

model_names=(mistralai/Mistral-7B-Instruct-v0.3)
for model_name in "${model_names[@]}"; do
    python experiment_rag.py --model "$model_name" \
                            $(env | grep -E "^[a-z]" | awk -F= '{print "--"$1" "$2}')
done



