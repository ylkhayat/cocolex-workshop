#!/bin/bash

# Function to display usage
usage() {
    # echo "Usage: $0 --required <param1,param2,...> [--param <value>]..."
    # echo "Example: $0 --required dataset,setup,split,method --dataset my_dataset --setup my_setup --split train --method rag --device 0"
    exit 1
}



declare -A inputs
declare -a required_params

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --required)
            IFS=',' read -ra required_params <<< "$2"  # Split required parameters into an array
            shift 2
            ;;
        --*)
            param="${key#--}"
            value="$2"
            if [[ -z "$value" || "$value" == --* ]]; then
                echo "Error: Missing value for parameter '$param'"
                usage
            fi
            inputs["$param"]="$value"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            # usage
            ;;
    esac
done


# Ensure all required parameters are provided
missing_params=()
for param in "${required_params[@]}"; do
    if [[ -z "${inputs[$param]}" ]]; then
        missing_params+=("$param")
    fi
done

if [[ ${#missing_params[@]} -gt 0 ]]; then
    echo "Error: Missing required parameters: ${missing_params[*]}"
    usage
fi


# Declare dynamic validation rules
declare -A validations

# Define validations for specific parameters
validations["dataset"]="clerc|echr"
validations["setup"]="bm25_oracle_passages_oracle_documents|bm25_relevant_passages_oracle_documents|dense_oracle_passages_oracle_documents|dense_relevant_passages_oracle_documents|bm25_noisy_oracle_passages_oracle_documents"
validations["split"]="train|test"
validations["method"]="rag|cad|knnlm"
validations["variant"]="normal"
if [[ "${inputs["method"]}" == "knnlm" ]]; then
    validations["variant"]="normal|plus|context|context_plus"
fi
validations["strategy"]="constant|adacad"
if [[ "${inputs["method"]}" == "knnlm" ]]; then
    validations["strategy"]="constant|entropy"
fi
validations["max_num_token"]="^[0-9]+$"
validations["device"]="^[0-9]+$"
validations["use_instructions"]="0|1"

# Validate inputs based on dynamic validations
for param in "${!inputs[@]}"; do
    value="${inputs[$param]}"
    if [[ -n "${validations[$param]}" ]]; then
        validation_regex="${validations[$param]}"
        if ! [[ "$value" =~ $validation_regex ]]; then
            echo "Error: Invalid value '$value' for parameter '$param'."
            echo "Valid options or format: ${validations[$param]}"
            exit 1
        fi
    fi
done

# Export inputs as environment variables
for param in "${!inputs[@]}"; do
    export "$param"="${inputs[$param]}"
done

# Display the parameters
# echo -e "\nParameters:"
# for param in $(echo "${!inputs[@]}" | tr ' ' '\n' | sort); do
#     echo -e "[!] $param:\t${inputs[$param]}"
# done