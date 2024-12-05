#!/bin/bash

cd "$(dirname "$0")" || exit

. ./utils/dynamic_input_handler.sh --required dataset,setup,split --method cad "$@"

if [[ $? -ne 0 ]]; then
    exit 1
fi

python experiment_cad.py $(env | grep -E "^[a-z]" | awk -F= '{print "--"$1" "$2}')



