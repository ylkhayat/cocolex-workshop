import json
import sys


from generation.workshop.dataloader import ModelInputPreprocessor
from generation.workshop.experiment_utils import load_experiment, parse_args_string
from tqdm import tqdm


script_start = "./run_experiments_{type}.sh"


def process_run(run_item):
    run = run_item["run"]
    filtered_run = None
    if not run.startswith("./run_experiments_"):
        return None  # Invalid run name
    
    try:
        if run.startswith(script_start.format(type="rag")):
            current_type = "rag"
            args_input_string = run.replace(script_start.format(type=current_type), "").strip()
            args = parse_args_string(current_type, args_input_string)
            dataset = args.dataset
            dataset_percentage = args.dataset_percentage
            setup = args.setup
            split = args.split
            top_k_passages = args.top_k_passages
            use_instructions = args.use_instructions

            method = current_type
            args.method = method
            config = {
                "dataset_percentage": dataset_percentage,
                "dataset": dataset,
                "method": method,
                "setup": setup,
                "split": split,
                "top_k_passages": top_k_passages,
                "use_instructions": use_instructions,
            }
            preprocessor = ModelInputPreprocessor(config, silent=True)
            all_results = []
            exists, finished, all_results = load_experiment(args, silent=True)
            if args.override or not finished or len(all_results) < len(preprocessor.processed_dataset):
                return run_item

        elif run.startswith(script_start.format(type="cad")):
            current_type = "cad"
            args_input_string = run.replace(script_start.format(type=current_type), "").strip()
            args = parse_args_string(current_type, args_input_string)
            dataset = args.dataset
            dataset_percentage = args.dataset_percentage
            setup = args.setup
            split = args.split
            strategy = args.strategy
            top_k_passages = args.top_k_passages
            use_instructions = args.use_instructions

            only_count_valid = False
            method = current_type
            alphas = [0.3] if strategy == 'constant' else [None]
            args.alphas = alphas
            if strategy == 'adacad':
                method = "adacad"
            args.method = method
            args.only_count_valid = only_count_valid
            config = {
                "dataset_percentage": dataset_percentage,
                "dataset": dataset,
                "method": method,
                "setup": setup,
                "split": split,
                "top_k_passages": top_k_passages,
                "use_instructions": use_instructions,
            }
            preprocessor = ModelInputPreprocessor(config, silent=True)    
            for alpha in alphas:
                args.alpha = alpha
                exists, finished, all_results = load_experiment(args, silent=True)
                if args.override or not finished or len(all_results) < len(preprocessor.processed_dataset):
                    return run_item
        elif run.startswith(script_start.format(type="knnlm")):
            current_type = "knnlm"
            args_input_string = run.replace(script_start.format(type=current_type), "").strip()
            args = parse_args_string(current_type, args_input_string)
            dataset = args.dataset
            dataset_percentage = args.dataset_percentage
            setup = args.setup
            split = args.split
            strategy = args.strategy
            top_k_passages = args.top_k_passages
            use_instructions = args.use_instructions
            variant = args.variant
            method = current_type
            if "context" in variant:
                method = f"{method}-context"
            if "adacad" in variant:
                method = f"{method}-adacad"
            if "plus" in variant:
                method = f"{method}-plus"
            method = f"{method}-{strategy}"
            if strategy == 'constant':
                lambas = [0.5]
            args.method = method
            config = {
                "dataset_percentage": dataset_percentage,
                "dataset": dataset,
                "method": method,
                "setup": setup,
                "split": split,
                "top_k_passages": top_k_passages,
                "use_instructions": use_instructions,
            }
            preprocessor = ModelInputPreprocessor(config, silent=True)
            if strategy == 'constant':
                for lamba in lambas:
                    args.lamba = lamba
                    exists, finished, all_results = load_experiment(args, silent=True)
                    if args.override or not finished or len(all_results) < len(preprocessor.processed_dataset):
                        return run_item
            else:
                args.lamba = None
            exists, finished, all_results = load_experiment(args, silent=True)
            if args.override or not finished or len(all_results) < len(preprocessor.processed_dataset):
                return run_item
        return filtered_run
    except Exception as e:
        if "supported" in str(e):
            return None
        raise e


def start_filtering(all_runs):
    grouped_runs = {}
    for run in all_runs:
        script_name = run["run"].split()[0]
        if script_name not in grouped_runs:
            grouped_runs[script_name] = []
        grouped_runs[script_name].append(run)
    for script_name, runs in grouped_runs.items():
        print(f"[!] {script_name}: {len(runs)}")
    results = []
    # for script_name, runs in grouped_runs.items():
    #     with mp.Pool(mp.cpu_count()) as pool:
    #         results.extend(list(tqdm(pool.imap_unordered(process_run, runs), total=len(runs), desc=f"Processing {script_name} Runs")))
    for script_name, runs in grouped_runs.items():
        for run in tqdm(runs, desc=f"Processing {script_name} Runs"):
            result = process_run(run)
            if result is not None:
                results.append(result)
    
    filtered_runs = [result for result in results if result is not None]
    # filtered_runs.sort(key=lambda x: x["run"])
    print(f"[!] to run: {len(filtered_runs)} experiments")
    return filtered_runs


if __name__ == "__main__":
    with open("all_runs.json", "r") as file:
        all_runs = json.load(file)
    print(f"[!] total runs: {len(all_runs)}")
    filtered_runs = start_filtering(all_runs)
    with open("all_runs.json", "w") as file:
        json.dump(filtered_runs, file, indent=4)
    all_to_run = "\n".join([f"{run['info']}|{run['window']}|{run['run']}" for run in filtered_runs])
    with open("to_run.txt", "w") as file:
        file.write(all_to_run)
    sys.exit(0)