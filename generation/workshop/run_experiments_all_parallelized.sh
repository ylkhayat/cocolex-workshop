#!/bin/bash

cd "$(dirname "$0")" || exit

if [ -z "$1" ]; then
    echo "Usage: $0 <split>"
    echo "Error: Missing required arguments."
    exit 1
fi

split=$1
run_mode=${2:-all}

models=(mistralai/Mistral-7B-Instruct-v0.3)
setups=(bm25_oracle_passages_oracle_documents bm25_relevant_passages_oracle_documents dense_oracle_passages_oracle_documents/jhu-clsp_LegalBERT-DPR-CLERC-ft dense_relevant_passages_oracle_documents/jhu-clsp_LegalBERT-DPR-CLERC-ft)
instructions=(0 1)
cad_methods=(constant adacad)
knnlm_methods=(constant entropy)
knnlm_variants=(normal context plus context_plus)
datasets=(clerc)
data_percentage=0.1


# Function to wait for an available GPU
wait_for_gpu() {
    while true; do
        gpu=$(./utils/check_gpus.sh) # Call the external script to get an available GPU
        if [[ -n $gpu ]]; then
            echo "$gpu"
            return
        fi
        sleep 5
    done
}

# Define colors and styles
GREEN="\033[32m"
RED="\033[31m"
PURPLE="\033[35m"
YELLOW="\033[33m"
CYAN="\033[36m"
BLUE="\033[34m"
BOLD="\033[1m"
RESET="\033[0m"

log_new_experiment() {
    experiment_name=$1
    current_extra_info=$2
    short_setup=$(echo "$setup" | awk -F'_' '{for(i=1;i<=NF;i++)$i=toupper(substr($i,1,1))}1' OFS='')
    upper_dataset=$(echo "$dataset" | tr '[:lower:]' '[:upper:]')
    first_setup="[$upper_dataset][$short_setup]"
    prefix="${BLUE}●${RESET} ${BOLD}${RED}[${experiment_name}]${RESET} ${YELLOW}${model}${RESET} ● ${GREEN}${first_setup}${RESET} ● ${PURPLE}${current_extra_info}${RESET} ● ${CYAN}GPU ${gpu}${RESET}"
    echo -e "$prefix"
}

session_name="experiment_session"
tmux new-session -d -s "$session_name" || tmux attach -t "$session_name"

echo -e "${BOLD} [!]\t running mode '$run_mode'"
for dataset in "${datasets[@]}"; do
    for instructed in "${instructions[@]}"; do
        for model in "${models[@]}"; do
            for setup in "${setups[@]}"; do
                extra_info=""
                if [ "$instructed" -eq 1 ]; then
                    extra_info="[instructed]"
                else
                    extra_info="[non-instructed]"
                fi

                if [[ "$run_mode" == "rag" || "$run_mode" == "all" ]]; then
                    gpu=$(wait_for_gpu)
                    log_new_experiment "RAG" "$extra_info"
                    tmux new-window -t "$session_name" -n "rag_${setup}" \
                        "./run_experiments_rag.sh --model \"$model\" \
                                    --dataset \"$dataset\" \
                                    --dataset_percentage \"$data_percentage\" \
                                    --device \"$gpu\" \
                                    --setup \"$setup\" \
                                    --split \"$split\" \
                                    --use_instructions \"$instructed\"; \
                        if [ \$? -eq 1 ]; then read; else tmux kill-window; fi"
                    sleep 30
                fi

                if [[ "$run_mode" == "cad" || "$run_mode" == "all" ]]; then
                    for strategy in "${cad_methods[@]}"; do
                        gpu=$(wait_for_gpu)
                        log_new_experiment "CAD" "$extra_info[$strategy]"
                        tmux new-window -t "$session_name" -n "cad_${setup}_${strategy}" \
                            "./run_experiments_cad.sh --model \"$model\" \
                                                    --dataset \"$dataset\" \
                                                    --dataset_percentage \"$data_percentage\" \
                                                    --decoding_strategy greedy \
                                                    --device \"$gpu\" \
                                                    --setup \"$setup\" \
                                                    --split \"$split\" \
                                                    --strategy \"$strategy\" \
                                                    --use_instructions \"$instructed\"; \
                            if [ \$? -eq 1 ]; then read; else tmux kill-window; fi"
                        sleep 30
                    done
                fi



                if [[ "$run_mode" == "knnlm" || "$run_mode" == "all" ]]; then
                    for knn_method in "${knnlm_methods[@]}"; do
                        for knn_variant in "${knnlm_variants[@]}"; do
                            gpu=$(wait_for_gpu)
                            log_new_experiment "KNNLM" "$extra_info[$knn_method][$knn_variant]"
                            tmux new-window -t "$session_name" -n "knn_${setup}_${knn_method}_${knn_variant}" \
                                "./run_experiments_knnlm.sh --model \"$model\" \
                                                            --dataset \"$dataset\" \
                                                            --dataset_percentage \"$data_percentage\" \
                                                            --decoding_strategy greedy \
                                                            --device \"$gpu\" \
                                                            --strategy \"$knn_method\" \
                                                            --setup \"$setup\" \
                                                            --split \"$split\" \
                                                            --use_instructions \"$instructed\" \
                                                            --variant \"$knn_variant\"; \
                                if [ \$? -eq 1 ]; then read; else tmux kill-window; fi"
                            sleep 30
                        done
                    done
                fi
            done
        done
    done
done

echo "All experiments launched in tmux session: $session_name"
echo "You can attach to the session with: tmux attach -t experiment_session"