print(f"[!] loading dependencies!")
import copy
from evaluate import load
print(f"[!] loading evaluation!")
from generation.evaluation.align_score.src.alignscore import AlignScore
from generation.evaluation.unieval.metric.evaluator import get_evaluator
from generation.evaluation.unieval.utils import convert_to_json
from generation.evaluation.GapHalu.eval_gap_halu import generate_and_parse_dataset
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
    parser.add_argument('--check_only',type=int, default=0, help='Only check if the experiment needs to run.')
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
        parser.add_argument("--lambda_smoothing_factor", type=float, default=0.5)
    
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--method", type=str)
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--setup", type=str, default="bm25_oracle_passages_oracle_documents")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--strategy", type=str, choices=['constant', 'entropy', 'adacad'], default='constant')
    parser.add_argument("--top_k_passages", type=int, default=10)
    parser.add_argument("--use_instructions", type=int, default=0)
    parser.add_argument("--variant", type=str, choices=['normal', 'context', 'context_adacad', 'plus', 'context_plus', 'context_adacad_plus'], default="normal")
    args = parser.parse_args()
    
    if 'echr_qa' in args.dataset:
        args.max_new_tokens = 300
    elif 'clerc' in args.dataset:
        args.max_new_tokens = 200
    elif 'obli_qa' in args.dataset:
        args.max_new_tokens = 200
        args.top_k_passages = 10
    elif 'cuad' in args.dataset:
        args.max_new_tokens = 50
        args.top_k_passages = 10
    elif 'echr' in args.dataset:
        args.max_new_tokens = 300
        
    args.use_instructions = args.use_instructions == 1
    args.override = args.override == 1
    return args


def load_results(args):
    new_args = copy.deepcopy(args)
    _, meta_output_path, _ = build_path(new_args)
    print(f"[!] loading meta from {meta_output_path}")
    has_meta = os.path.exists(meta_output_path)
    if has_meta:
        with open(meta_output_path, "r") as f:
            meta = json.load(f)
        return meta
    return None


def load_experiment(args):
    minimum_accepted_generation_length = int(args.max_new_tokens * 0.1)
    new_args = copy.deepcopy(args)
    exists = False
    results = []
    results_output_path, meta_output_path, legacy_results_output_path = build_path(new_args)
    print(f"[!] loading experiment from {results_output_path}")
    has_results = os.path.exists(results_output_path)
    has_legacy_results = os.path.exists(legacy_results_output_path)
    print(f"[!] found meta: {meta_output_path}")
    has_meta = os.path.exists(meta_output_path)
    duplicates = 0
    invalid = 0
    existing_docids = set()
    if has_legacy_results:
        print("[!] found legacy results, accumulating...")
        with open(legacy_results_output_path, "r") as f:
            for line in f:
                record = json.loads(line)
                record_gen = record['gen']
                if len(record_gen) <= minimum_accepted_generation_length:
                    invalid += 1
                    continue
                if record['meta']['docid'] not in existing_docids:
                    results.append(record)
                    existing_docids.add(record['meta']['docid'])
                else:
                    duplicates += 1
    if has_results:
        exists = True
        duplicates = 0
        with open(results_output_path, "r") as f:
            for line in f:
                record = json.loads(line)
                record_gen = record['gen']
                if len(record_gen) < minimum_accepted_generation_length:
                    invalid += 1
                    continue
                if record['meta']['docid'] not in existing_docids:
                    results.append(record)
                    existing_docids.add(record['meta']['docid'])
                else:
                    duplicates += 1
                
    print(f"[!] found {len(results)} results, and pruned {duplicates} duplicates, and {invalid} invalid records")
    print(f"[!] writing new results to {results_output_path}")
    write_results(results, results_output_path)
    return exists, has_meta, results

common_excluded_keys = {"split", "model", "method", "setup", "top_k_passages", "override", "device", "only_count_valid", "check_only"}
excluded_keys  = common_excluded_keys | {"decoding_strategy"}
results_path_excluded_keys = common_excluded_keys | {"is_truncated" , "dataset_percentage"}
meta_path_excluded_keys = common_excluded_keys | {"is_truncated"}

def build_path(args):
    local_results_path_excluded_keys = results_path_excluded_keys
    local_meta_path_excluded_keys = meta_path_excluded_keys
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
    if "knnlm" in method and "entropy" not in args["strategy"]:
        extra_keys = {"entropy_sigmoid_threshold", "lambda_smoothing_factor"}
        local_results_path_excluded_keys = local_results_path_excluded_keys | extra_keys
        local_meta_path_excluded_keys = local_meta_path_excluded_keys | extra_keys
    dataset = args['dataset']
    top_k_passages = args['top_k_passages']
    setup = args['setup']
    results_params = {k: v for k, v in args.items() if k not in local_results_path_excluded_keys}
    results_params_str = "_".join([f"{key[:2]}-{value}" for key, value in results_params.items()])
    meta_params = {k: v for k, v in args.items() if k not in local_meta_path_excluded_keys}
    meta_params_str = "_".join([f"{key[:2]}-{value}" for key, value in meta_params.items()])

    model_cleaned = model.replace("/", "_")
    common_output_path = f"../basement/{dataset}/{split}/{setup}/{top_k_passages}/{model_cleaned}"
    results_output_path = f"{common_output_path}/generations/{method}__{results_params_str}.jsonl"
    legacy_results_output_path = f"{common_output_path}/results/{method}__{meta_params_str}.jsonl"
    meta_output_path = f"{common_output_path}/meta/{method}__{meta_params_str}.json"
    os.makedirs(os.path.dirname(results_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(meta_output_path), exist_ok=True)
    return results_output_path, meta_output_path, legacy_results_output_path

def print_args(args):
    args_dict = {key: value for key, value in vars(args).items()}
    print(tabulate(args_dict.items(), headers=["Argument", "Value"], tablefmt="fancy_grid"))


def write_results(results, results_output_path, reference_dataset=None):
    if reference_dataset is not None:
        reference_docids = reference_dataset['docid']
        results.sort(key=lambda x: reference_docids.index(x['meta']['docid']))
    with open(results_output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    return results

def save_metadata(scores_results, meta_output_path, args):
    scores = {"bert_score", "rouge", "align_score", "unieval"}
    params = {k: v for k, v in args.items() if k not in excluded_keys}
    scores = {k: v for k, v in scores_results.items() if k in scores}
    try:
        with open(meta_output_path, "w") as f:
            data_without_results = args.copy()
            data_without_results['params'] = params
            data_without_results['scores'] = scores
            for key in params:
                data_without_results.pop(key)
            f.write(json.dumps(data_without_results, indent=4))
    except Exception as e:
        print(f"[!] error saving results: {e}")

def average_scores(score_list):
    rounded_score = round(sum(score_list) / len(score_list) * 100, 2)
    return rounded_score


def get_resources(results, reference_dataset):
    docids = [result['meta']['docid'] for result in results]
    reference_dataset = reference_dataset.filter(lambda record: record['docid'] in docids)
    return reference_dataset.to_dict()
    
    
def evaluate(results, device, reference_dataset, args, has_new_results=True, align_score_evaluation_mode="nli"):
    try:
        evaluation = load_results(args)
        if evaluation is None:
            evaluation = {}
        else:
            evaluation = evaluation['scores']
    except Exception:
        evaluation = {}
        
    unfiltered_results_length = len(results)
    results = [result for result in results if len(result['gen'].strip()) > 20]
    print(f"[!] filtering results, filtered {unfiltered_results_length - len(results)} invalid results")
    # ipdb.set_trace()
    method = args.method
    print(f"[!] evaluation for method: {method}")
    resources = get_resources(results, reference_dataset)
    resources_docids = resources['docid']
    results.sort(key=lambda x: resources_docids.index(x['meta']['docid']))
    all_docids = [result['meta']['docid'] for result in results]
    assert all_docids == resources_docids, "The order of docids does not match between results and reference dataset"
    # drop the id and merge all bigger documents
    oracle_documents = ['\n'.join([document[1] for document in resource]) for resource in resources['meta.oracle_documents']]
    top_k_passages = ['\n'.join([document for document in resource]) for resource in resources['meta.top_k_passages']]
    predictions = [result['gen'] for result in results]
    references = [result['meta']['gold_text'] for result in results]
    prefixes = [result['meta']['previous_text'] for result in results]
    
    
    
    assert len(predictions) == len(references) == len(prefixes) == len(oracle_documents) == len(top_k_passages), "Lengths do not match"
    if (has_new_results or 
        "bert_score" not in evaluation or 
        "precision" not in evaluation["bert_score"] or
        "recall" not in evaluation["bert_score"] or
        "f1" not in evaluation["bert_score"]):
        bertscore = load("bertscore")
        bertscores = bertscore.compute(predictions=predictions, references=references, lang="en", device=device)
        evaluation["bert_score"] = {}
        evaluation["bert_score"]["precision"] = np.mean(bertscores["precision"])
        evaluation["bert_score"]["recall"] = np.mean(bertscores["recall"])
        evaluation["bert_score"]["f1"] = np.mean(bertscores["f1"])
    else:
        print("[!] skip bert score")
        
    if has_new_results or "rouge" not in evaluation:
        rouge = load('rouge')
        evaluation["rouge"] = rouge.compute(predictions=predictions, references=references)
    else:
        print("[!] skip rouge score")
    
    if has_new_results or "unieval" not in evaluation:
        task = 'summarization'
        unieval_evaluator = get_evaluator(task, device=device)
        data = convert_to_json(output_list=predictions, src_list=prefixes, ref_list=references)
        eval_scores = unieval_evaluator.evaluate(data)
        sum_scores = {key: 0 for key in eval_scores[0].keys()}
        for score in eval_scores:
            for key, value in score.items():
                sum_scores[key] += value
        evaluation["unieval"] = {key: round(sum_scores[key] / len(eval_scores), 4) for key in sum_scores}
    else:
        print("[!] skip unieval")
        
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
    align_score_evaluation_mode_set = {
        "correctness": align_score_evaluation_mode,
        "faithfulness": align_score_evaluation_mode
        }
    if args.dataset == "cuad" or args.dataset == "obli_qa":
        align_score_evaluation_mode_set["correctness"] = "bin"
        align_score_evaluation_mode_set["faithfulness"] = "bin"
        
    has_align_score_correctness = False
    has_align_score_faithfulness_documents = False
    has_align_score_faithfulness_passages = False
    if not has_new_results and "align_score" in evaluation:
        if "correctness" in evaluation["align_score"]:
            has_align_score_correctness = True
        if "faithfulness" in evaluation["align_score"]:
            if "documents" in evaluation["align_score"]["faithfulness"]:
                has_align_score_faithfulness_documents = True
            if "passages" in evaluation["align_score"]["faithfulness"]:
                has_align_score_faithfulness_passages = True
    need_any_align_score = not has_align_score_correctness or not has_align_score_faithfulness_documents or not has_align_score_faithfulness_passages
    if "align_score" not in evaluation:
        evaluation["align_score"] = {}
        evaluation["align_score"]["meta"] = {}
    if need_any_align_score:
        print(f"[!] using alignscore model evaluation mode: {align_score_evaluation_mode}")
        alignscorer = {}
        for current_align_score_evaluation_mode in set(align_score_evaluation_mode_set.values()):
            print(f"[!] loading alignscore model for {current_align_score_evaluation_mode}")
            alignscorer[current_align_score_evaluation_mode] = AlignScore(model='roberta-large',
                                                                        batch_size=64,
                                                                        device=device,
                                                                        ckpt_path=align_score_model_to_use,
                                                                        evaluation_mode=current_align_score_evaluation_mode,
                                                                        sent_tokenize=sent_tokenize)
    if not has_align_score_correctness:
        print(f"[!] using '{align_score_evaluation_mode_set['correctness']}' for correctness")
        print("[!] using generated text for correctness")
        evaluation["align_score"]["meta"]["correctness"] = align_score_evaluation_mode_set["correctness"]
        proper_align_scorer = alignscorer[align_score_evaluation_mode_set["correctness"]]
        correctness = proper_align_scorer.score(contexts=references, claims=predictions)
        evaluation["align_score"]["correctness"] = average_scores(correctness)
    
    def faithfulness_filterer(current_references):
        if args.dataset == "cuad":
            print("[!] filtering for faithfulness")
            filtered_data = [
                (current_reference, prediction, oracle_document, top_k_passage) for current_reference, prediction, oracle_document, top_k_passage in zip(current_references, predictions, oracle_documents, top_k_passages)
                if "No relevant information" not in current_reference
            ]
            if filtered_data:
                filtered_references, filtered_predictions, filtered_oracle_documents, filtered_top_k_passages = map(list, zip(*filtered_data))
            else:
                filtered_references, filtered_predictions, filtered_oracle_documents, filtered_top_k_passages = [], [], [], []
            print(f"[!] neglecting {len(current_references) - len(filtered_references)} records for faithfulness which had 'No relevant...'")
            return filtered_references, filtered_predictions, filtered_oracle_documents, filtered_top_k_passages
        print("[!] no filtering for faithfulness")
        return current_references, predictions, oracle_documents, top_k_passages
    
    if not has_align_score_faithfulness_documents and not has_align_score_faithfulness_passages:
        evaluation["align_score"]["faithfulness"] = {}
        
        
    _, filtered_predictions, filtered_oracle_documents, filtered_top_k_passages = faithfulness_filterer(references)
    assert len(filtered_predictions) == len(filtered_oracle_documents) == len(filtered_top_k_passages), "Lengths do not match"
    
    if not has_align_score_faithfulness_documents:
        print("[!] using oracle documents for faithfulness")
        print(f"[!] using '{align_score_evaluation_mode_set['faithfulness']}' for faithfulness")
        evaluation["align_score"]["meta"]["faithfulness"] = align_score_evaluation_mode_set["faithfulness"]
        proper_align_scorer = alignscorer[align_score_evaluation_mode_set["faithfulness"]]
        assert proper_align_scorer is not None, "Align scorer is None"
        try:
            faithfulness_documents = proper_align_scorer.score(contexts=filtered_oracle_documents, claims=filtered_predictions)
            evaluation["align_score"]["faithfulness"]["documents"] = average_scores(faithfulness_documents)
        except Exception as e:
            print(f"[!] error in faithfulness documents: {e}")
            evaluation["align_score"]["faithfulness"]["documents"] = 0.0
    if not has_align_score_faithfulness_passages:
        try:
            assert proper_align_scorer is not None, "Align scorer is None"
            print("[!] using top k passages for faithfulness")
            faithfulness_passages = proper_align_scorer.score(contexts=filtered_top_k_passages, claims=filtered_predictions)
            evaluation["align_score"]["faithfulness"]["passages"] = average_scores(faithfulness_passages)
        except Exception as e:
            print(f"[!] error in faithfulness passages: {e}")
            evaluation["align_score"]["faithfulness"]["passages"] = 0.0
    
    return evaluation

def add_experiment(data, args):
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
        "align_score.correctness", 
        "align_score.faithfulness.passages", 
        "align_score.faithfulness.documents", 
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
    add_from_args = {"align_score.meta.correctness", "align_score.meta.faithfulness"}
    
    extra_args = []
    for key in add_from_args:
        keys = key.split('.')
        value = data
        for k in keys:
            value = value.get(k, None)
            if value is None:
                break
        extra_args.append({key: value})
        
    description += ", ".join([f"{key}: {value}" for key, value in extra_args])
            
    new_row.append(description)
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(service_account_json_file, scope)
    client = gspread.authorize(creds)
    spreadsheet_url = "https://docs.google.com/spreadsheets/d/1bE5AbY1hrqlR-_v-ohLCgHA6hFvRCTpH4lrRPqXm9UU/edit?usp=sharing"
    sheet = client.open_by_url(spreadsheet_url)
    worksheet = sheet.worksheet(f'Generation — {args["dataset_percentage"]}')
    # worksheet = sheet.get_worksheet(0)
    worksheet.append_row(new_row, value_input_option="USER_ENTERED")
    print("[!] added experiment metric!")
    