import sys

import ipdb
if '/srv/elkhyo/lexquo' not in sys.path:
    sys.path.insert(0, '/srv/elkhyo/lexquo')
from generation.workshop.dataloader import ModelInputPreprocessor
from generation.workshop.experiment_utils import (
    add_experiment, 
    build_args_parser, 
    evaluate, print_args, 
    reshape_and_save_experiment_results, 
    should_run_experiment
    )

from tqdm import tqdm
from slack_notifier import send_slack_notification
from generation.baselines.rag.rag import RAG
import torch

args = build_args_parser(method="rag")

batch_size = args.batch_size
dataset = args.dataset
dataset_percentage = args.dataset_percentage
decoding_strategy = args.decoding_strategy
device = args.device
max_new_tokens = args.max_new_tokens
model_name = args.model
repetition_penalty = args.repetition_penalty
setup = args.setup
split = args.split
strategy = args.strategy
top_k_passages = args.top_k_passages
use_instructions = args.use_instructions
variant = args.variant

method = 'rag'
args.method = method
print_args(args)

if not should_run_experiment(args):
    print("[!] experiment already exists, skipping...")
    sys.exit(0)
    
device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
rag_model = RAG(model_name=model_name, device=device.index)
config = {
    "dataset_percentage": dataset_percentage,
    "dataset": dataset,
    "method": method,
    "setup": setup,
    "split": split,
    "top_k_passages": top_k_passages,
    "max_tokens": rag_model.model.config.max_position_embeddings,
    "use_instructions": use_instructions,
}
preprocessor = ModelInputPreprocessor(tokenizer=rag_model.tokenizer)
work_dataset = preprocessor.process_dataset(config)
results = []
try:
    start_index = 0
    is_truncated_global = False
    for batch in tqdm(work_dataset.iter(batch_size=batch_size), desc="Batches", total=len(work_dataset) // batch_size):
        prefixes = batch['previous_text']
        docids = batch['docid']
        refs = batch['gold_text']
        context_prefixes = batch['context_prefix']
        contexts = batch['context']
        if any(batch['is_truncated']):
            is_truncated_global = True
        prompts = batch['prompt']
        prompts = [f"{context_prefix}\n\n{context}{rag_model.tokenizer.eos_token}{prompt}" for context_prefix, context, prompt in zip(context_prefixes, contexts, prompts)]
        outputs = rag_model.generate(
            prompts=prompts,
            max_length=max_new_tokens,
            decoding_strategy=decoding_strategy,
            use_repetition_penalty=repetition_penalty > 1.0,
            repetition_penalty_value=repetition_penalty
            )
        generated_texts = rag_model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for index, (docid, generated_text, gold_text, prev_text, prompt) in enumerate(zip(docids, generated_texts, refs, prefixes, prompts)):
            new_object = {
                "meta": {}
            }
            new_object["meta"]['docid'] = docid
            new_object["meta"]['index'] = (split, start_index + index)
            new_object["meta"]['gold_text'] = gold_text
            new_object["meta"]['previous_text'] = prev_text
            new_object["meta"]['prompt'] = prompt
            new_object["gen"] = generated_text
            results.append(new_object)
    args.is_truncated = is_truncated_global
    experiment_results = evaluate(results, device)
    experiment_results['results'] = results
    results_output_path, meta_output_path = reshape_and_save_experiment_results(experiment_results, vars(args))
    start_index += batch_size
    add_experiment(experiment_results, vars(args))
    send_slack_notification(f"[!] Experiment completed: {results_output_path}!")
except Exception as e:
    print(f"[!] Error: {e}")
    send_slack_notification(f"[x] Experiment failed!")
    sys.exit(1)