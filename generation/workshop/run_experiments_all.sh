#!/bin/bash

source /srv/elkhyo/anaconda3/etc/profile.d/conda.sh
conda activate pld

cd "$(dirname "$0")" || exit

if [ -z "$1" ]; then
    echo "Usage: $0 <split>"
    echo "Error: Missing required arguments."
    exit 1
fi

split=$1
run_mode=${2:-all}
variant_mode=${3:-all}
time_between_experiments=28

models=(
    # meta-llama/Llama-3.1-8B-Instruct
    mistralai/Mistral-7B-Instruct-v0.3
    Equall/Saul-7B-Instruct-v1 
    )
setups=(
    bm25_relevant_passages_oracle_documents 
    bm25_oracle_passages_oracle_documents 
    bm25_noisy_oracle_passages_oracle_documents
    # dense_oracle_passages_oracle_documents/jhu-clsp_LegalBERT-DPR-CLERC-ft 
    # dense_relevant_passages_oracle_documents/jhu-clsp_LegalBERT-DPR-CLERC-ft
    )
instructions=(1)
cad_methods=(constant adacad)
knnlm_methods=(constant entropy)
knnlm_variants=(
    context 
    context_adacad 
    # plus
    context_plus 
    context_adacad_plus
    )
datasets=(
    cuad
    obli_qa
    clerc
    echr_qa
    # oal_qa
    )
# dataset_percentage=0.01
# dataset_percentage=0.1
dataset_percentage=1.0

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

check_experiment() {
    script_path="$1"
    passed_args="$2"

    tmux new-window -t "$session_name" -n experiment_checker \
        "$script_path $passed_args --check_only 1; echo \$? > check_exit_code.txt; \
        echo 'done' > check_status.txt"
    while true; do
        if [[ -f check_status.txt && $(cat check_status.txt) == "done" ]]; then
            break
        fi
        sleep 1
    done
    while [[ ! -f check_exit_code.txt ]]; do
        sleep 1
    done

    exit_code=$(cat check_exit_code.txt)
    rm check_exit_code.txt check_status.txt
    return $exit_code
}



GREEN="\033[32m"
RED="\033[31m"
PURPLE="\033[35m"
YELLOW="\033[33m"
CYAN="\033[36m"
BLUE="\033[34m"
BOLD="\033[1m"
RESET="\033[0m"

log_new_experiment() {
    run_command=$1
    experiment_name=$(echo "$run_command" | awk -F'_' '{print toupper($3)}' | awk -F'.' '{print toupper($1)}')
    current_extra_info=$2
    setup=$(echo "$run_command" | awk -F'--setup' '{print $2}' | awk '{print $1}' | tr -d '[:space:]' | tr -d '"')
    dataset=$(echo "$run_command" | awk -F'--dataset' '{print $2}' | awk '{print $1}' | tr -d '[:space:]' | tr -d '"')
    model=$(echo "$run_command" | awk -F'--model' '{print $2}' | awk '{print $1}' | tr -d '[:space:]' | tr -d '"')
    short_setup=$(echo "$setup" | awk -F'_' '{for(i=1;i<=NF;i++)$i=toupper(substr($i,1,1))}1' OFS='')
    upper_dataset=$(echo "$dataset" | tr '[:lower:]' '[:upper:]')
    first_setup="[$upper_dataset][$short_setup]"
    prefix="${BLUE}●${RESET} ${BOLD}${RED}[${experiment_name}]${RESET} ${YELLOW}${model}${RESET} ● ${GREEN}${first_setup}${RESET} ● ${PURPLE}${current_extra_info}${RESET} ● ${CYAN}GPU ${gpu}${RESET}"
    echo -e "$prefix"
}

log_skip_experiment() {
    run_command=$1
    experiment_name=$(echo "$run_command" | awk -F'_' '{print toupper($3)}' | awk -F'.' '{print toupper($1)}')
    current_extra_info=$2
    setup=$(echo "$run_command" | awk -F'--setup' '{print $2}' | awk '{print $1}' | tr -d '[:space:]' | tr -d '"')
    dataset=$(echo "$run_command" | awk -F'--dataset' '{print $2}' | awk '{print $1}' | tr -d '[:space:]' | tr -d '"')
    model=$(echo "$run_command" | awk -F'--model' '{print $2}' | awk '{print $1}' | tr -d '[:space:]' | tr -d '"')
    short_setup=$(echo "$setup" | awk -F'_' '{for(i=1;i<=NF;i++)$i=toupper(substr($i,1,1))}1' OFS='')
    upper_dataset=$(echo "$dataset" | tr '[:lower:]' '[:upper:]')
    first_setup="[$upper_dataset][$short_setup]"
    prefix="${BLUE}●${RESET} ${BOLD}${RED}[${experiment_name}]${RESET} ${YELLOW}${model}${RESET} ● ${GREEN}${first_setup}${RESET} ● ${PURPLE}${current_extra_info}${RESET}"
    echo -e "${prefix} — ${RED}SKIPPED${RESET}"
}

session_name="experiment_session"
tmux new-session -d -s "$session_name" || tmux attach -t "$session_name"
should_run=0
echo -e "${BOLD} [!]\t running mode '$run_mode'"
echo -e "${BOLD} [!]\t variant mode '$variant_mode'"

override=1
if [ ! -f "all_runs.json" ] || [ "$override" -eq 1 ]; then
    all_runs=()
    all_window_names=()
    extra_info="[instructed]"
    for dataset in "${datasets[@]}"; do
        for instructed in "${instructions[@]}"; do
            for model in "${models[@]}"; do
                for setup in "${setups[@]}"; do
                    common_suffix="$(echo "$setup" | awk -F'_' '{for(i=1;i<=NF;i++)$i=toupper(substr($i,1,1))}1' OFS='')_$(echo "$dataset" | tr '[:lower:]' '[:upper:]')_$(echo "$model" | awk -F'/' '{print toupper(substr($2,1,1))}')"
                    extra_info=""
                    if [ "$instructed" -eq 1 ]; then
                        extra_info="[instructed]"
                    else
                        extra_info="[non-instructed]"
                    fi

                    if [[ "$run_mode" == "rag" || "$run_mode" == "all" ]]; then
                        python_args="--model \"$model\" \
                                    --dataset \"$dataset\" \
                                    --dataset_percentage $dataset_percentage \
                                    --setup \"$setup\" \
                                    --split \"$split\" \
                                    --use_instructions \"$instructed\""
                        window_name="rag_${common_suffix}"
                        all_runs+=("./run_experiments_rag.sh $python_args")
                        all_window_names+=("$window_name;")
                        all_infos+=("$extra_info")
                    fi

                    if [[ "$run_mode" == "cad" || "$run_mode" == "all" ]]; then
                        for strategy in "${cad_methods[@]}"; do
                            python_args="--model \"$model\" \
                                        --dataset \"$dataset\" \
                                        --dataset_percentage $dataset_percentage \
                                        --decoding_strategy greedy \
                                        --setup \"$setup\" \
                                        --split \"$split\" \
                                        --strategy \"$strategy\" \
                                        --use_instructions \"$instructed\""
                            window_name="cad_${strategy}_${common_suffix}"
                            all_runs+=("./run_experiments_cad.sh $python_args;")
                            all_window_names+=("$window_name;")
                            all_infos+=("$extra_info[$strategy]")
                        done
                    fi

                    if [[ "$run_mode" == "knnlm" || "$run_mode" == "all" ]]; then
                        for knn_method in "${knnlm_methods[@]}"; do
                            for knn_variant in "${knnlm_variants[@]}"; do
                                if [[ "$knn_method" == "constant" && ("$knn_variant" == *"adacad"* || "$knn_variant" == *"plus"*) ]]; then
                                    continue
                                fi
                                python_args="--model \"$model\" \
                                            --dataset \"$dataset\" \
                                            --dataset_percentage $dataset_percentage \
                                            --decoding_strategy greedy \
                                            --strategy \"$knn_method\" \
                                            --setup \"$setup\" \
                                            --split \"$split\" \
                                            --use_instructions \"$instructed\" \
                                            --variant \"$knn_variant\""
                                window_name="knn_${knn_method}_${knn_variant}_${common_suffix}"
                                all_runs+=("./run_experiments_knnlm.sh $python_args;")
                                all_window_names+=("$window_name;")
                                all_infos+=("$extra_info[$knn_method][$knn_variant]")
                            done
                        done
                    fi
                done
            done
        done
    done

    echo "Total number of runs: ${#all_runs[@]}"

    combined_file="all_runs.json"
    echo "Creating combined JSON file: $combined_file"
    echo "[" > "$combined_file"
    for i in "${!all_runs[@]}"; do
        run="${all_runs[$i]}"
        window_name="${all_window_names[$i]}"
        info="${all_infos[$i]}"
        run=$(echo "$run" | sed 's/"/\\"/g')
        window_name=$(echo "$window_name" | sed 's/"/\\"/g')
        info=$(echo "$info" | sed 's/"/\\"/g')
        json_entry=$(cat <<EOF
{
"run": "${run%;}",
"window": "${window_name%;}",
"info": "$info"
}
EOF
)
        if [ $i -lt $((${#all_runs[@]} - 1)) ]; then
            json_entry="$json_entry,"
        fi

        echo "$json_entry" >> "$combined_file"
    done
    echo "]" >> "$combined_file"
    echo "All combined data have been dumped into $combined_file"
fi

python filter_experiments.py
sleep 5

filtered_runs_file="to_run.txt"
IFS=$'\n' read -r -d '' -a filtered_runs_array < <(cat "$filtered_runs_file")
echo "[!] runs to go: ${#filtered_runs_array[@]}"

for filtered_run in "${filtered_runs_array[@]}"; do
    IFS='|' read -r info_name window_name run_command <<< "$filtered_run"
    if ! tmux list-windows -t "$session_name" | grep -q "$window_name"; then
        gpu=$(wait_for_gpu)
        log_new_experiment "$run_command" "$info_name"
        tmux new-window -t "$session_name" -n "$window_name" \
                "$run_command --device $gpu; tmux kill-window -t \"\${session_name}:$window_name\";"
        sleep $time_between_experiments
    else
        log_skip_experiment "$run_command" "$info_name"
    fi
done

echo "All experiments launched in tmux session: $session_name"
echo "You can attach to the session with: tmux attach -t experiment_session"