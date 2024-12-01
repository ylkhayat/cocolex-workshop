print(f"[!] loading dependencies!")
from datasets import load_dataset
from evaluate import load
print(f"[!] loading evaluation!")
from generation.evaluation.align_score.src.alignscore import AlignScore
from generation.evaluation.unieval.metric.evaluator import get_evaluator
from generation.evaluation.unieval.utils import convert_to_json
from oauth2client.service_account import ServiceAccountCredentials
from tabulate import tabulate
import argparse
import gspread
import json
import numpy as np
import os

num_proc = os.cpu_count() - 2
align_score_model_to_use = 'https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt'
service_account_json_file = '../../thesis_service_account.json'
if not os.path.exists(service_account_json_file):
    raise FileNotFoundError(f"[!] service account JSON file not found: {service_account_json_file}")


def build_args_parser(method):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_percentage", type=float, default=0.005)
    parser.add_argument("--dataset", type=str, default="clerc")
    parser.add_argument("--decoding_strategy", type=str, choices=['top_p', 'top_k', 'greedy'], default='greedy')
    parser.add_argument("--device", type=int, default=0)
    
    if method == 'rag':
        parser.add_argument("--repetition_penalty", type=float, default=1.3)
    if 'cad' in method:
        parser.add_argument("--repetition_penalty", type=float, default=1.0)
    if 'knnlm' in method:
        parser.add_argument("--entropy_sigmoid_threshold", type=float, default=0.5)
        parser.add_argument("--entropy_strategy", type=str, choices=['exp', 'exp_norm', 'sig'], default='exp_norm')
        parser.add_argument("--lambda_smoothing_factor", type=float, default=0.3)
        parser.add_argument("--repetition_penalty", type=float, default=1.5)
    

    
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--method", type=str)
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--setup", type=str, default="bm25_oracle_passages_oracle_documents")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--strategy", type=str, choices=['constant', 'entropy', 'adacad'], default='constant')
    parser.add_argument("--top_k_passages", type=int, default=3)
    parser.add_argument("--use_instructions", type=int, default=0)
    parser.add_argument("--variant", type=str, choices=['normal', 'context', 'plus', 'context_plus'], default="normal")
    args = parser.parse_args()
    args.use_instructions = args.use_instructions == 1
    return args

def print_args(args):
    args_dict = {key: value for key, value in vars(args).items()}
    print(tabulate(args_dict.items(), headers=["Argument", "Value"], tablefmt="fancy_grid"))


def reshape_and_save_experiment_results(scores_results, args):
    split = args["split"]
    if "split" not in args:
        split = "train"
    model = args["model"]
    method = args["method"]
    dataset = args["dataset"]
    new_results = scores_results["results"]
    top_k_passages = args["top_k_passages"]
    setup = args["setup"]
    oracle_top_k = f"{setup}_{top_k_passages}"
    
    excluded_keys = {"split", "model", "method", "setup", "top_k_passages"}
    scores = {"bert_score", "rouge", "align_score", "unieval"}
    params = {k: v for k, v in args.items() if k not in excluded_keys}
    scores = {k: v for k, v in scores_results.items() if k in scores}
    param_str = "_".join([f"{key[0]}-{value}" for key, value in params.items()])

    model_cleaned = model.replace("/", "_")
    results_output_path = f"../basement/{dataset}/{model_cleaned}/{split}/{oracle_top_k}/results/{method}__{param_str}.jsonl"
    meta_output_path = f"../basement/{dataset}/{model_cleaned}/{split}/{oracle_top_k}/meta/{method}__{param_str}.json"
    os.makedirs(os.path.dirname(results_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(meta_output_path), exist_ok=True)
    try:
        with open(results_output_path, "w") as f:
            for item in new_results:
                f.write(json.dumps(item) + "\n")
        with open(meta_output_path, "w") as f:
            data_without_results = args.copy()
            data_without_results['params'] = params
            data_without_results['scores'] = scores
            for key in params:
                data_without_results.pop(key)
            f.write(json.dumps(data_without_results, indent=4))
    except Exception as e:
        print(f"[!] error saving results: {e}")
    return results_output_path, meta_output_path

def average_scores(score_list):
    rounded_score = round(sum(score_list) / len(score_list) * 100, 2)
    return rounded_score

def evaluate(results, device):
    predictions = [result['gen'] for result in results]
    references = [result['meta']['gold_text'] for result in results]
    prefixes = [result['meta']['previous_text'] for result in results]

    bertscore = load("bertscore")
    bertscores = bertscore.compute(predictions=predictions, references=references, lang="en", device=device)
    precision = np.mean(bertscores["precision"])
    recall = np.mean(bertscores["recall"])
    f1 = np.mean(bertscores["f1"])
    rouge = load('rouge')
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    task = 'summarization'
    unieval_evaluator = get_evaluator(task)
    data = convert_to_json(output_list=predictions, src_list=prefixes, ref_list=references)
    eval_scores = unieval_evaluator.evaluate(data)
    sum_scores = {key: 0 for key in eval_scores[0].keys()}
    for score in eval_scores:
        for key, value in score.items():
            sum_scores[key] += value
    unieval_mean_scores = {key: round(sum_scores[key] / len(eval_scores), 4) for key in sum_scores}

    evaluation_mode = "nli"
    print(f"[!] using alignscore model evaluation mode: {evaluation_mode}")
    alignscorer = AlignScore(model='roberta-large', batch_size=32, device=device, ckpt_path=align_score_model_to_use, evaluation_mode=evaluation_mode)
    alignscores = alignscorer.score(contexts=references, claims=predictions)
    align_scores = average_scores(alignscores)
    
    return {
        "bert_score": {
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "rouge": rouge_scores,
        "unieval": unieval_mean_scores,
        "align_score": align_scores
    }

def add_experiment(data,
                   args):
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
        "align_score", 
        "unieval.coherence", 
        "unieval.consistency",
        "unieval.fluency",
        "unieval.relevance",
        "bert_score.precision",
        "bert_score.recall",
        "bert_score.f1",
        "rouge.rouge1",
        "rouge.rouge2",
        "rouge.rougeL",
        "rouge.rougeLsum",
    ]

    new_row = []
    for key in sheet_columns_args:
        keys = key.split('.')
        value = args
        for k in keys:
            value = value.get(k, None)
            if value is None:
                break
        new_row.append(value)
    for key in sheet_columns_data:
        keys = key.split('.')
        value = data
        for k in keys:
            value = value.get(k, None)
            if value is None:
                break
        new_row.append(value)
    short_setup = ''.join([part[0] for part in args["setup"].split("_")])
    new_row[1] = new_row[1].upper()
    new_row[2] = new_row[2].upper()
    new_row[3] = str(new_row[3]).upper()
    new_row[4] = short_setup.upper()
    required_keys = [
        "model", "method", "dataset", "setup", "top_k_passages", "dataset_percentage", "split" ]
    for key in required_keys:
        if key not in args:
            raise KeyError(f"Missing required key in data: {key}")
    for key in ["coherence", "consistency", "fluency", "relevance"]:
        if key not in data["unieval"]:
            raise KeyError(f"Missing required key in data['unieval']: {key}")
    for key in ["precision", "recall", "f1"]:
        if key not in data["bert_score"]:
            raise KeyError(f"Missing required key in data['bert_score']: {key}")
    for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        if key not in data["rouge"]:
            raise KeyError(f"Missing required key in data['rouge']: {key}")
        
    description = ", ".join([f"{key}: {value}" for key, value in args.items() if key not in sheet_columns_args])
    new_row.append(description)
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(service_account_json_file, scope)
    client = gspread.authorize(creds)
    spreadsheet_url = "https://docs.google.com/spreadsheets/d/1bE5AbY1hrqlR-_v-ohLCgHA6hFvRCTpH4lrRPqXm9UU/edit?usp=sharing"
    sheet = client.open_by_url(spreadsheet_url)
    worksheet = sheet.get_worksheet(2)
    worksheet.append_row(new_row, value_input_option="USER_ENTERED")
    print("[!] added experiment metric!")
    

def build_context_prompt(prompt, contexts, tokenizer, use_instruction):
    ref_text = '\n\n'.join(contexts)
    retrieved_ids = [doc.split('\n')[0] for doc in contexts]
    context_prefix = "Below are reference cases provided for factual accuracy. When generating content, you must reference and cross-check the relevant details with the provided reference texts by their reference IDs. (e.g., " + ', '.join(retrieved_ids) + ").\nThese references take precedence over inferred or assumed information. Your output must clearly align with the facts in these cases."
    # if use_instruction and 'apply_chat_template' in dir(tokenizer):
    #     context_chat_parts = [
    #         { "role": "system", "content": context_prefix },
    #         { "role": "user", "content": ref_text }
    #     ]
    #     context = tokenizer.apply_chat_template(context_chat_parts, tokenize=False)
    #     context = context.replace(tokenizer.bos_token, '').replace(tokenizer.eos_token, '')
    # else:
    context = ref_text
        
    # prompt_prefix = "As a legal professional, continue to draft the following case in a formal, concise manner, maintaining the style and tone of a legal write-up. Your response must:\n1. Be professional and precise, adhering to legal writing conventions.\n2. Remain within 100 to 400 words.\n3. Explicitly cite the provided reference IDs in the text where applicable to ensure factual accuracy and consistency.\n4.Exclude redundant language, assumptions, personal opinions, or information not supported by the provided references.\n5.Focus exclusively on the requested continuation without diverging into irrelevant details or commentary."
    # prompt_prefix = "You are a legal professional. Continue to draft the following case in a formal, concise manner, maintaining the style and tone of a legal write-up. Be professional and precise, adhering to legal writing conventions. Explicitly cite the provided reference IDs in the text where applicable to ensure factual accuracy and consistency. Exclude redundant language, assumptions, personal opinions, or information not supported by the provided references."
    # prompt_prefix = "Continue to write the provided draft case using the style of the write-up. Your response should:\n1. Be concise and within 100 to 400 words.\n2. Explicitly cite the reference IDs in the text where applicable to ensure factual consistency.\n3. Avoid redundant language, assumptions, or information not found in the references."
    # prompt_prefix_ref_ids = "You must explicitly use the reference cases and mention their reference ids, i.e. " + ', '.join(retrieved_ids) + ". "
    prompt = (
        'Continue to write the following case in the style of my writeup. Your answer should range from 100 to 400 words. ' + 
        # prompt_prefix_ref_ids + 
        'Make your answer concise, and avoid redundant languages and assumptions. Below is what I have written so far:\n\n' +
        prompt
        )
    if use_instruction and 'apply_chat_template' in dir(tokenizer):
        prompt_chat_parts = [
            { "role": "system", "content": "You are a helpful legal professional." },
            { "role": "user", "content": prompt },
        ]
        prompt = tokenizer.apply_chat_template(prompt_chat_parts, tokenize=False)
    return context_prefix, context, prompt
    
def preprocess_function(record, 
                        top_k,
                        tokenizer,
                        use_instruction=True):
    prev_text = record['previous_text']
    gold_text = record['gold_text']
    oracle_documents = record['citations']
    retrieved_docs = record['top_10_passages'][:top_k]
    context_prefix, context, prompt = build_context_prompt(prev_text, retrieved_docs, tokenizer, use_instruction)
    return {
        "prompt": prompt,
        "context": context,
        "context_prefix": context_prefix,
        "meta": {
            "gold_text": gold_text,
            "previous_text": prev_text,
            "oracle_documents": oracle_documents,
            "top_k_passages": retrieved_docs
        }
    }


def setup_dataset(config, tokenizer, return_original_dataset_length=False):
    dataset = config['dataset']
    dataset_percentage = config['dataset_percentage']
    setup = config['setup']
    split = config['split']
    top_k_passages = config['top_k_passages']
    use_instructions = config['use_instructions']
    
    assert dataset is not None, "dataset must be defined in the config"
    assert dataset_percentage is not None, "dataset_percentage must be defined in the config"
    assert setup is not None, "setup must be defined in the config"
    assert split is not None, "split must be defined in the config"
    assert top_k_passages is not None, "top_k_passages must be defined in the config"
    
    dataset_repo_name = "CLERC-generation-workshop"
    if dataset == "echr":
        dataset_repo_name = "ECHR-generation-workshop"
    dataset_repo_name = f"ylkhayat/{dataset_repo_name}"
    dataset = load_dataset(dataset_repo_name, data_dir=setup, split=split)
    length_of_dataset = int(len(dataset) * dataset_percentage)
    print(f"[!] dataset: {dataset_repo_name}")
    print(f"[!] num of records: {length_of_dataset}")
    processed_dataset = dataset.select(range(int(len(dataset) * dataset_percentage)))
    top_k_passages = config.get('top_k_passages', 5)
    processed_dataset = processed_dataset.map(
        lambda record: preprocess_function(record,
                                           top_k=top_k_passages,
                                           tokenizer=tokenizer,
                                           use_instruction=use_instructions),
        batched=False
        )
    if return_original_dataset_length:
        return processed_dataset, len(dataset)
    return processed_dataset
    
