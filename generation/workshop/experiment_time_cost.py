import gc
import itertools
import os
import sys

import numpy as np
from tqdm import tqdm

from generation.baselines.knnlm.knnlm import KNNLM


if '/srv/elkhyo/lexquo' not in sys.path:
    sys.path.insert(0, '/srv/elkhyo/lexquo')
from generation.baselines.cad.cad import CAD
from generation.workshop.dataloader import ModelInputPreprocessor
from generation.workshop.experiment_utils import (
    build_args_parser,
    assign_args_modifications,
    print_args
    )
from generation.baselines.rag.rag import RAG
from oauth2client.service_account import ServiceAccountCredentials
from slack_notifier import send_slack_notification
import gspread
import torch

service_account_json_file = os.path.join('../../', 'thesis_service_account.json')
if not os.path.exists(service_account_json_file):
    raise FileNotFoundError(f"[!] service account JSON file not found: {service_account_json_file}")


def add_experiment(args, data):
    sheet_columns_args = [
        "model",
        "method",
        "dataset",
        "use_instructions",
        "setup",
        "top_k_passages",
        "dataset_percentage",
        "split"
    ]
    sheet_columns_data = [
        "generation_length.max",
        "generation_length.actual",
        "tokenized_lengths.prompt.mean",
        "tokenized_lengths.context.mean",
        "tokenized_lengths.references.mean",
        "time_cost.mean.token",
        "time_cost.overall.token",
        "time_cost.mean.build_datastores",
        "time_cost.overall.build_datastores",
        "time_cost.mean.knn",
        "time_cost.overall.knn",
    ]
    
    new_row = [args[key] for key in sheet_columns_args]
    for key in sheet_columns_data:
        keys = key.split('.')
        value = data
        for k in keys:
            if value is None:
                break
            value = value.get(k, None)
            if value is None:
                break
        new_row.append(value)

    short_setup = ''.join([part[0] for part in args["setup"].split("_")])
    new_row[1] = new_row[1].upper()
    new_row[2] = new_row[2].upper()
    new_row[3] = str(new_row[3]).upper()
    new_row[4] = short_setup.upper()

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(service_account_json_file, scope)
    client = gspread.authorize(creds)

    spreadsheet_url = "https://docs.google.com/spreadsheets/d/1bE5AbY1hrqlR-_v-ohLCgHA6hFvRCTpH4lrRPqXm9UU/edit?usp=sharing"
    sheet = client.open_by_url(spreadsheet_url)
    worksheet = sheet.worksheet("Generation — Time-Cost")

    worksheet.append_row(new_row, value_input_option="USER_ENTERED")
    
def prepare_datasets(model, config):
    datasets = [
        "clerc",
        "echr_qa",
        "oal_qa",
        # must be last because of args modifications
        "cuad",
        "obli_qa",
    ]
    working_datasets = {}
    for dataset in datasets:
        config["dataset"] = dataset
        config["top_k_passages"] = 10 if dataset in ["obli_qa", "cuad"] else 3
        dataset_preprocessor = ModelInputPreprocessor(config)
        work_dataset, _ = dataset_preprocessor.process_dataset(tokenizer=model.tokenizer,
                                                               max_tokens=model.model.config.max_position_embeddings)
        working_datasets[dataset] = work_dataset
    return working_datasets

# python experiment_time_cost.py --model mistralai/Mistral-7B-Instruct-v0.3 --setup bm25_relevant_passages_oracle_documents --split test --use_instructions 1 --decoding_strategy greedy --dataset_percentage 0.05 --batch_size 1 --device 2


def setup_results():
    all_tokenized_lengths = {
        "prompt": [],
        "context": [],
        "references": [],
    }
    all_time_costs = {
        "overall": {},
        "mean": {},
    }
    all_generation_lengths = {
        "actual": [],
    }
    final_report = {
        "generation_length": {},
        "tokenized_lengths": {},
        "time_cost": all_time_costs,
    }
    return all_tokenized_lengths, all_time_costs, all_generation_lengths, final_report

def finalize_results(max_new_tokens, all_tokenized_lengths, all_time_costs, all_generation_lengths, final_report):
    final_report["tokenized_lengths"] = {
        "prompt": {
            "mean": np.mean(all_tokenized_lengths["prompt"]),
        },
        "context": {
            "mean": np.mean(all_tokenized_lengths["context"]),
        },
        "references": {
            "mean": np.mean(all_tokenized_lengths["references"]) if len(all_tokenized_lengths["references"]) > 0 else 0.0,
        }
    }
    for key, value in all_time_costs.items():
        for sub_key, sub_value in value.items():
            final_report["time_cost"][key][sub_key] = np.mean(sub_value) if len(sub_value) > 0 else 0.0
    final_report["generation_length"] = {
        "max": max_new_tokens,
        "actual": np.mean(all_generation_lengths["actual"]),
    }
    return final_report

def run_experiment_rag():
    args = build_args_parser(method="rag")
    model_name = args.model
    setup = args.setup
    split = args.split
    device = args.device
    top_k_passages = args.top_k_passages
    use_instructions = args.use_instructions
    dataset_percentage = args.dataset_percentage
    method = 'rag'
    args.method = method
    config = {
        "dataset_percentage": dataset_percentage,
        "method": method,
        "setup": setup,
        "split": split,
        "top_k_passages": top_k_passages,
        "use_instructions": use_instructions,
    }
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model = RAG(model_name=model_name, device=device.index, compile=False)
    working_datasets = prepare_datasets(model, config)
    for key, dataset in tqdm(working_datasets.items()):
        args.dataset = key
        args = assign_args_modifications(args)
        args.top_k_passages = 10 if key in ["obli_qa", "cuad"] else 3
        print_args(args)
        max_new_tokens = args.max_new_tokens
        assert max_new_tokens is not None
        decoding_strategy = args.decoding_strategy
        repetition_penalty = args.repetition_penalty
        batch_size = args.batch_size
        
        all_tokenized_lengths, all_time_costs, all_generation_lengths, final_report = setup_results()
        for batch in tqdm(dataset.iter(batch_size=batch_size), desc=f"Processing {key} dataset", total=len(dataset) // batch_size):
            context_prefixes = batch['context_prefix']
            contexts = batch['context']
            prompts = batch['prompt']
            contexts = [f"{context_prefix}\n\n{context}" for context_prefix, context in zip(context_prefixes, contexts)]
            _, report = model.generate(
                prompts=prompts,
                contexts=contexts,
                max_length=max_new_tokens,
                decoding_strategy=decoding_strategy,
                use_repetition_penalty=repetition_penalty > 1.0,
                repetition_penalty_value=repetition_penalty,
                generate_time_report=True
                )
            all_tokenized_lengths["prompt"].append(report["tokenized_lengths"]["prompt"])
            all_tokenized_lengths["context"].append(report["tokenized_lengths"]["context"])
            for key, value in report["time_cost"].items():
                for sub_key, sub_value in value.items():
                    if key not in all_time_costs:
                        all_time_costs[key] = {}
                    if sub_key not in all_time_costs[key]:
                        all_time_costs[key][sub_key] = []
                    all_time_costs[key][sub_key].append(sub_value)
            all_generation_lengths["actual"].append(report["generation_length"]["actual"])
        final_report = finalize_results(max_new_tokens, all_tokenized_lengths, all_time_costs, all_generation_lengths, final_report)
        add_experiment(vars(args), final_report)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    
def run_experiment_cad():
    strategies = ['constant', 'adacad']
    args = build_args_parser(method="cad")
    model_name = args.model
    device = args.device
    method = 'cad'
    args.method = method
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model = CAD(model_name=model_name, device=device.index, compile=False)
    setup = args.setup
    split = args.split
    method = args.method
    top_k_passages = args.top_k_passages
    use_instructions = args.use_instructions
    dataset_percentage = args.dataset_percentage
    config = {
        "dataset_percentage": dataset_percentage,
        "method": method,
        "setup": setup,
        "split": split,
        "top_k_passages": top_k_passages,
        "use_instructions": use_instructions,
    }
    working_datasets = prepare_datasets(model, config)
    for strategy in strategies:
        args.strategy = strategy
        if strategy == 'constant':
            alpha = 0.3
        else:
            alpha = None
        if strategy == 'adacad':
            method = "adacad"
        args.method = method
        for key, dataset in tqdm(working_datasets.items()):
            args.dataset = key
            args = assign_args_modifications(args)
            args.top_k_passages = 10 if key in ["obli_qa", "cuad"] else 3
            print_args(args)
            max_new_tokens = args.max_new_tokens
            assert max_new_tokens is not None
            decoding_strategy = args.decoding_strategy
            repetition_penalty = args.repetition_penalty
            batch_size = args.batch_size
            
            all_tokenized_lengths, all_time_costs, all_generation_lengths, final_report = setup_results()
            for batch in tqdm(dataset.iter(batch_size=batch_size), desc=f"Processing {key} dataset", total=len(dataset) // batch_size):
                context_prefixes = batch['context_prefix']
                contexts = batch['context']
                prompts = batch['prompt']
                contexts = [f"{context_prefix}\n\n{context}" for context_prefix, context in zip(context_prefixes, contexts)]
                _, report = model.generate(
                    prompts=prompts,
                    contexts=contexts,
                    max_length=max_new_tokens,
                    alpha=alpha,
                    method=method,
                    decoding_strategy=decoding_strategy,
                    use_repetition_penalty=repetition_penalty > 1.0,
                    repetition_penalty_value=repetition_penalty,
                    generate_time_report=True
                    )
                all_tokenized_lengths["prompt"].append(report["tokenized_lengths"]["prompt"])
                all_tokenized_lengths["context"].append(report["tokenized_lengths"]["context"])
                for key, value in report["time_cost"].items():
                    for sub_key, sub_value in value.items():
                        if key not in all_time_costs:
                            all_time_costs[key] = {}
                        if sub_key not in all_time_costs[key]:
                            all_time_costs[key][sub_key] = []
                        all_time_costs[key][sub_key].append(sub_value)
                all_generation_lengths["actual"].append(report["generation_length"]["actual"])
            final_report = finalize_results(max_new_tokens, all_tokenized_lengths, all_time_costs, all_generation_lengths, final_report)
            add_experiment(vars(args), final_report)
    del model
    torch.cuda.empty_cache()
    gc.collect()



def run_experiment_knnlm():
    args = build_args_parser(method="knnlm")
    model_name = args.model
    device = args.device
    method = 'knnlm'
    args.method = method
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model = KNNLM(model_name=model_name, device=device.index, compile=False)
    setup = args.setup
    split = args.split
    method = args.method
    strategy = args.strategy
    top_k_passages = args.top_k_passages
    use_instructions = args.use_instructions
    dataset_percentage = args.dataset_percentage
    config = {
        "dataset_percentage": dataset_percentage,
        "method": method,
        "setup": setup,
        "split": split,
        "top_k_passages": top_k_passages,
        "use_instructions": use_instructions,
    }
    working_datasets = prepare_datasets(model, config)

    entropy_sigmoid_threshold = args.entropy_sigmoid_threshold
    entropy_strategy = args.entropy_strategy
    lambda_smoothing_factor = args.lambda_smoothing_factor
    if strategy == 'constant':
        lamba = 0.5
        args.lamba = lamba
    layer_index = -1
    k = 10
    args.method = method
    print_args(args)
    strategies = [
        'constant',
        'entropy'
        ]
    variants = [
        'context',
        'context_adacad',
        'context_plus' 
        'context_adacad_plus'
        ]
    for strategy, variant in itertools.product(strategies, variants):
        method = "knnlm"
        if strategy == 'constant' and variant != 'context':
            continue
        if "context" in variant:
            method = f"{method}-context"
        if "adacad" in variant:
            method = f"{method}-adacad"
        if "plus" in variant:
            method = f"{method}-plus"
        method = f"{method}-{strategy}"
        args.method = method
        for key, dataset in tqdm(working_datasets.items()):
            args.dataset = key
            args = assign_args_modifications(args)
            args.top_k_passages = 10 if key in ["obli_qa", "cuad"] else 3
            print_args(args)
            try:
                use_faiss = args.use_faiss
            except AttributeError:
                use_faiss = False
            max_new_tokens = args.max_new_tokens
            assert max_new_tokens is not None
            decoding_strategy = args.decoding_strategy
            repetition_penalty = args.repetition_penalty
            batch_size = args.batch_size
            
            all_tokenized_lengths, all_time_costs, all_generation_lengths, final_report = setup_results()
            for batch in tqdm(dataset.iter(batch_size=batch_size), desc=f"Processing {key} dataset", total=len(dataset) // batch_size):
                context_prefixes = batch['context_prefix']
                contexts = batch['context']
                prompts = batch['prompt']
                contexts = [f"{context_prefix}\n\n{context}" for context_prefix, context in zip(context_prefixes, contexts)]
                references = contexts.copy()
                if "plus" in variant:
                    references = batch['meta.oracle_documents']
                _, report = model.generate(
                    prompts=prompts,
                    contexts=contexts,
                    references=references,
                    max_length=max_new_tokens,
                    lamba=lamba,
                    strategy=strategy,
                    k=k,
                    variant=variant,
                    entropy_strategy=entropy_strategy,
                    entropy_sigmoid_threshold=entropy_sigmoid_threshold,
                    lambda_smoothing_factor=lambda_smoothing_factor,
                    datastore_from_layer_index=layer_index,
                    decoding_strategy=decoding_strategy,
                    use_repetition_penalty=repetition_penalty > 1.0,
                    repetition_penalty_value=repetition_penalty,
                    use_faiss=use_faiss,
                    generate_time_report=True
                    )
                all_tokenized_lengths["prompt"].append(report["tokenized_lengths"]["prompt"])
                all_tokenized_lengths["context"].append(report["tokenized_lengths"]["context"])
                all_tokenized_lengths["references"].append(report["tokenized_lengths"]["references"])
                for key, value in report["time_cost"].items():
                    for sub_key, sub_value in value.items():
                        if key not in all_time_costs:
                            all_time_costs[key] = {}
                        if sub_key not in all_time_costs[key]:
                            all_time_costs[key][sub_key] = []
                        all_time_costs[key][sub_key].append(sub_value)
                all_generation_lengths["actual"].append(report["generation_length"]["actual"])
            final_report = finalize_results(max_new_tokens, all_tokenized_lengths, all_time_costs, all_generation_lengths, final_report)
            add_experiment(vars(args), final_report)
    del model
    torch.cuda.empty_cache()
    gc.collect()

# run_experiment_rag()
# run_experiment_cad()
run_experiment_knnlm()
send_slack_notification("[!] Time-Cost Experiment completed!")