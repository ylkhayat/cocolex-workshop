import sys
import traceback
if '/srv/elkhyo/lexquo' not in sys.path:
    sys.path.insert(0, '/srv/elkhyo/lexquo')
from generation.workshop.dataloader import ModelInputPreprocessor
from generation.workshop.experiment_utils import (
    add_experiment,
    build_args_parser,
    build_path,
    evaluate,
    load_experiment,
    print_args,
    save_metadata,
    write_results
    )
from generation.baselines.knnlm.knnlm import KNNLM
from slack_notifier import send_slack_notification
from tqdm import tqdm
import torch

args = build_args_parser(method="knnlm")

batch_size = args.batch_size
dataset = args.dataset
dataset_percentage = args.dataset_percentage
decoding_strategy = args.decoding_strategy
device = args.device
entropy_sigmoid_threshold = args.entropy_sigmoid_threshold
entropy_strategy = args.entropy_strategy
lambda_smoothing_factor = args.lambda_smoothing_factor
max_new_tokens = args.max_new_tokens
model_name = args.model
repetition_penalty = args.repetition_penalty
setup = args.setup
split = args.split
strategy = args.strategy
top_k_passages = args.top_k_passages
use_instructions = args.use_instructions
variant = args.variant


method = "knnlm"
if "context" in variant:
    method = f"{method}-context"
if "adacad" in variant:
    method = f"{method}-adacad"
if "plus" in variant:
    method = f"{method}-plus"
method = f"{method}-{strategy}"
if strategy == 'constant':
    lamdas = [0.5]
    args.lamdas = lamdas
layers = [-1]
args.method = method
print_args(args)

try:
    def carry_experiment(lamba,
                        strategy,
                        k,
                        layer_index):
        global device
        args.lamba = lamba
        exists, finished, all_results = load_experiment(args)
        if not args.override and finished:
            print("[!] experiment already exists, skipping...")
            sys.exit(1 if args.check_only else 0)
        if args.check_only:
            sys.exit(0)
        device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        knnlm_model = KNNLM(model_name=model_name, device=device.index)
        config = {
            "dataset_percentage": dataset_percentage,
            "dataset": dataset,
            "method": method,
            "setup": setup,
            "split": split,
            "top_k_passages": top_k_passages,
            "max_tokens": knnlm_model.model.config.max_position_embeddings,
            "use_instructions": use_instructions,
        }
        preprocessor = ModelInputPreprocessor(tokenizer=knnlm_model.tokenizer)
        work_dataset = preprocessor.process_dataset(config)
        
        needed_docids = work_dataset['docid'] # needed finished + needed not finished
        current_results = [result for result in all_results if result['meta']['docid'] in needed_docids] # needed finished + not needed finished
        computed_docids = [result['meta']['docid'] for result in current_results] # needed finished
        print(f"[!] used {len(current_results)} relevent records")
        results_output_path, meta_output_path, _ = build_path(args)
        try:
            start_index = 0
            is_truncated_global = False
            filted_work_dataset = work_dataset.filter(lambda record: record['docid'] not in computed_docids)
            print(f"[!] filtered {len(work_dataset) - len(filted_work_dataset)} records")
            record_counter = 0
            for batch in tqdm(filted_work_dataset.iter(batch_size=batch_size), desc="Batches", total=len(filted_work_dataset) // batch_size):
                if any(batch['is_truncated']):
                    is_truncated_global = True
                docids = batch['docid']
                prefixes = batch['previous_text']
                refs = batch['gold_text']
                contexts = batch['context']
                context_prefixes = batch['context_prefix']
                
                references = contexts.copy()
                if "plus" in variant:
                    references = batch['meta.oracle_documents']
                    
                contexts = [f"{context_prefix}\n\n{context}" for context_prefix, context in zip(context_prefixes, contexts)]
                prompts = batch['prompt']
                outputs = knnlm_model.generate(
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
                    repetition_penalty_value=repetition_penalty
                    )
                generated_texts = knnlm_model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for index, (docid, generated_text, gold_text, prev_text, prompt, context) in enumerate(zip(docids, generated_texts, refs, prefixes, prompts, contexts)):
                    new_object = {
                        "meta": {}
                    }
                    new_object["meta"]['docid'] = docid
                    new_object["meta"]['index'] = (split, start_index + index)
                    new_object["meta"]['gold_text'] = gold_text
                    new_object["meta"]['previous_text'] = prev_text
                    new_object["meta"]['prompt'] = prompt
                    new_object["meta"]['context'] = context
                    new_object["gen"] = generated_text
                    current_results.append(new_object)
                    all_results.append(new_object)
                    record_counter += 1
                if record_counter % 10 == 0:
                    write_results(all_results, results_output_path)
            write_results(all_results, results_output_path)
            knnlm_model.model.to(torch.device('cpu'))
            args.is_truncated = is_truncated_global
            experiment_results = evaluate(current_results, device, work_dataset, args.method)
            experiment_results['knn_k'] = k
            experiment_results['lamba'] = lamba
            experiment_results['layer_index'] = layer_index
            start_index += batch_size
            current_args = vars(args)
            save_metadata(experiment_results, meta_output_path, current_args)
            add_experiment(experiment_results, current_args)
            send_slack_notification(f"Experiment completed: {results_output_path}!")
        except Exception as e:
            print(f"[!] Error: {e}")
            send_slack_notification(f"Error in experiment: {results_output_path}!")
            sys.exit(1)
    if strategy == 'constant':
        for lamba in tqdm(lamdas, desc="Lamba"):
            for layer_index in layers:
                carry_experiment(lamba, "constant", 10, layer_index)
    else:
        for layer_index in layers:
            carry_experiment(None, "entropy", 10, layer_index)
except Exception as e:
    print(f"[!] Error: {e}")
    traceback.print_exc()
    sys.exit(1)