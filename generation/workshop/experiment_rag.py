import sys
if '/srv/elkhyo/lexquo' not in sys.path:
    sys.path.insert(0, '/srv/elkhyo/lexquo')
from experiment_utils import (
    add_experiment,
    build_args_parser,
    evaluate,
    print_args,
    reshape_and_save_experiment_results,
    setup_dataset
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

config = {
    "dataset": dataset,
    "dataset_percentage": dataset_percentage,
    "setup": setup,
    "split": split,
    "top_k_passages": top_k_passages,
    "use_instructions": use_instructions
}

device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
rag_model = RAG(model_name=model_name, device=device.index)
clerc_dataset = setup_dataset(config,
                              tokenizer=rag_model.tokenizer)

results = []
try:
    start_index = 0
    for batch in tqdm(clerc_dataset.iter(batch_size=batch_size), desc="Processing batches", total=len(clerc_dataset) // batch_size):
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
    experiment_results = evaluate(results, device)
    experiment_results['results'] = results
    results_output_path, meta_output_path = reshape_and_save_experiment_results(experiment_results, vars(args))
    start_index += batch_size
    add_experiment(experiment_results, vars(args))
    send_slack_notification(f"[!] Experiment completed: {results_output_path}!")
except Exception as e:
    print(f"[!] Error: {e}")
    send_slack_notification(f"[x] Experiment failed!")
    raise e
