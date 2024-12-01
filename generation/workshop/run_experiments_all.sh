#!/bin/bash

cd "$(dirname "$0")" || exit

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <split> <max_num_token> <cuda_device>"
    echo "Error: Missing required arguments."
    exit 1
fi

split=$1
max_num_token=$2
cuda=$3

data_percentage=0.005

model_names=(mistralai/Mistral-7B-Instruct-v0.3)
setups=(bm25_oracle_passages_oracle_documents bm25_relevant_passages_oracle_documents)
for model_name in "${model_names[@]}"; do
    for setup in "${setups[@]}"; do
        python experiment_rag.py --model_name "$model_name" --oracle_setup "$setup" --dataset_percentage "$data_percentage" --split "$split" --device "$cuda" --max_num_token "$max_num_token"
        cad_methods=(cad adacad)
        for cad_method in "${cad_methods[@]}"; do
            python experiment_cad.py --model_name "$model_name" --oracle_setup "$setup" --dataset_percentage "$data_percentage" --method "$cad_method" --device "$cuda" --decoding_strategy greedy --split "$split" --max_num_token "$max_num_token"
        done
        knn_methods=(constant entropy)
        for knn_method in "${knn_methods[@]}"; do
            python experiment_knnlm.py --model_name "$model_name" --oracle_setup "$setup" --dataset_percentage "$data_percentage" --device "$cuda" --decoding_strategy greedy --split "$split" --lamba_strategy "$knn_method" --max_num_token "$max_num_token"
        done
    done
done



