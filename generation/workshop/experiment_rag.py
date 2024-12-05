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

results = []
exists, finished, results = load_experiment(args)
if not args.override and finished:
    print("[!] experiment already exists, skipping...")
    sys.exit(1 if args.check_only else 0)
if args.check_only:
    sys.exit(0)
truncate_results_found = len(results)

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

needed_docids = work_dataset['docid'] # needed finished + needed not finished
results = [result for result in results if result['meta']['docid'] in needed_docids] # needed finished + not needed finished
computed_docids = [result['meta']['docid'] for result in results] # needed finished
initial_length = len(results)
print(f"[!] filtered {initial_length - len(results)} records")
try:
    results_output_path, meta_output_path, _ = build_path(args)
    start_index = 0
    is_truncated_global = False
    filted_work_dataset = work_dataset.filter(lambda record: record['docid'] not in computed_docids) # needed not finished
    for batch in tqdm(filted_work_dataset.iter(batch_size=batch_size), desc="Batches", total=len(filted_work_dataset) // batch_size):
        if any(batch['is_truncated']):
            is_truncated_global = True
        docids = batch['docid']
        prefixes = batch['previous_text']
        docids = batch['docid']
        refs = batch['gold_text']
        context_prefixes = batch['context_prefix']
        contexts = batch['context']
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
        write_results(results, results_output_path)
    rag_model.model.to(torch.device('cpu'))
    args.is_truncated = is_truncated_global
    experiment_results = evaluate(results, device, work_dataset, args.method)
    start_index += batch_size
    current_args = vars(args)
    save_metadata(experiment_results, meta_output_path, current_args)
    add_experiment(experiment_results, current_args)
    send_slack_notification(f"[!] Experiment completed: {results_output_path}!")
except Exception as e:
    print(f"[!] Error: {e}")
    traceback.print_exc()
    send_slack_notification(f"[x] Experiment failed!")
    sys.exit(1)