print(f"[!] loading dependencies!")
import copy
from evaluate import load
print(f"[!] loading evaluation!")
from generation.evaluation.align_score.src.alignscore import AlignScore
from generation.evaluation.unieval.metric.evaluator import get_evaluator
from generation.evaluation.unieval.utils import convert_to_json
from oauth2client.service_account import ServiceAccountCredentials
from transformers import pipeline
from tabulate import tabulate
import argparse
import gspread
import json
import numpy as np
import os

num_proc = os.cpu_count() - 2
align_score_model_to_use = 'https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt'
service_account_json_file = os.path.join(os.path.dirname(__file__), '../../', 'thesis_service_account.json')
if not os.path.exists(service_account_json_file):
    raise FileNotFoundError(f"[!] service account JSON file not found: {service_account_json_file}")


def build_args_parser(method):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_percentage", type=float, default=0.005)
    parser.add_argument("--dataset", type=str, default="clerc")
    parser.add_argument("--decoding_strategy", type=str, choices=['top_p', 'top_k', 'greedy'], default='greedy')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--override", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.5)
    
    if 'knnlm' in method:
        parser.add_argument("--entropy_sigmoid_threshold", type=float, default=0.5)
        parser.add_argument("--entropy_strategy", type=str, choices=['exp', 'exp_norm', 'sig'], default='exp_norm')
        parser.add_argument("--lambda_smoothing_factor", type=float, default=0.3)
    
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--method", type=str)
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--setup", type=str, default="bm25_oracle_passages_oracle_documents")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--strategy", type=str, choices=['constant', 'entropy', 'adacad'], default='constant')
    parser.add_argument("--top_k_passages", type=int, default=3)
    parser.add_argument("--use_instructions", type=int, default=0)
    parser.add_argument("--variant", type=str, choices=['normal', 'context', 'plus', 'context_plus'], default="normal")
    args = parser.parse_args()
    args.use_instructions = args.use_instructions == 1
    args.override = args.override == 1
    return args


def should_run_experiment(args):
    new_args = copy.deepcopy(args)
    override = new_args.override
    exists = False
    if 'cad' in args.method:
        alphas = [None] if args.strategy == 'adacad' else args.alphas
        for i in alphas:
            new_args.alpha = i
            results_output_path, meta_output_path = build_path(new_args)
            if os.path.exists(results_output_path) and os.path.exists(meta_output_path):
                print(f"[!] experiment already exists for alpha {i}, skipping...")
                exists = True
                break
    else:
        results_output_path, meta_output_path = build_path(new_args)
        if os.path.exists(os.path.abspath(results_output_path)) and os.path.exists(os.path.abspath(meta_output_path)):
            exists = True
    return override or not exists

excluded_keys = {"split", "model", "method", "setup", "top_k_passages", "override", "device", "only_count_valid"}

def build_path(args):
    try:
        args = vars(args)
    except:
        pass
    if "split" not in args:
        split = "train"
    else: 
        split = args['split']
    model = args['model']
    method = args['method']
    dataset = args['dataset']
    top_k_passages = args['top_k_passages']
    setup = args['setup']
    params = {k: v for k, v in args.items() if k not in excluded_keys}
    param_str = "_".join([f"{key[:2]}-{value}" for key, value in params.items()])

    model_cleaned = model.replace("/", "_")
    common_output_path = f"../basement/{dataset}/{split}/{setup}/{top_k_passages}/{model_cleaned}"
    results_output_path = f"{common_output_path}/results/{method}__{param_str}.jsonl"
    meta_output_path = f"{common_output_path}/meta/{method}__{param_str}.json"
    return results_output_path, meta_output_path

def print_args(args):
    args_dict = {key: value for key, value in vars(args).items()}
    print(tabulate(args_dict.items(), headers=["Argument", "Value"], tablefmt="fancy_grid"))


def reshape_and_save_experiment_results(scores_results, args):
    new_results = scores_results["results"]
    scores = {"bert_score", "rouge", "align_score", "unieval"}
    params = {k: v for k, v in args.items() if k not in excluded_keys}
    scores = {k: v for k, v in scores_results.items() if k in scores}

    results_output_path, meta_output_path = build_path(args)
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




def evaluate(results, device, evaluation_mode="nli_sp"):
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
    def sent_tokenize(text):
        pipe = pipeline(
            'token-classification',
            model= 'rcds/distilbert-SBD-en-judgements-laws',
            aggregation_strategy="simple",
            device = device
        )
        sentences = pipe(text)
        sentences = [sent['word'] for sent in sentences]
        return sentences
    evaluation_mode = evaluation_mode
    print(f"[!] using alignscore model evaluation mode: {evaluation_mode}")
    alignscorer = AlignScore(model='roberta-large',
                             batch_size=32,
                             device=device,
                             ckpt_path=align_score_model_to_use,
                             evaluation_mode=evaluation_mode,
                             sent_tokenize=sent_tokenize)
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
        
    description = ", ".join([f"{key}: {value}" for key, value in args.items() if key not in sheet_columns_args and key not in excluded_keys])
    new_row.append(description)
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(service_account_json_file, scope)
    client = gspread.authorize(creds)
    spreadsheet_url = "https://docs.google.com/spreadsheets/d/1bE5AbY1hrqlR-_v-ohLCgHA6hFvRCTpH4lrRPqXm9UU/edit?usp=sharing"
    sheet = client.open_by_url(spreadsheet_url)
    worksheet = sheet.get_worksheet(0)
    worksheet.append_row(new_row, value_input_option="USER_ENTERED")
    print("[!] added experiment metric!")
