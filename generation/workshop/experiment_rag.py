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
from slack_notifier import send_slack_notification
from generation.baselines.rag.rag import RAG
from transformers import pipeline
import os
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
    "dataset_percentage": dataset_percentage,
    "setup": setup,
    "split": split,
    "top_k_passages": top_k_passages,
    "use_instructions": use_instructions
}

device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

legal_advisor = pipeline("text-generation",
                         model=model_name,
                         device_map=device if device != 'auto' else 'auto',
                         torch_dtype=torch.float16,
                         max_new_tokens=max_new_tokens,
                         batch_size=batch_size)

clerc_dataset = setup_dataset(config,
                              tokenizer=legal_advisor.tokenizer)
legal_advisor.tokenizer.pad_token = legal_advisor.tokenizer.eos_token
legal_advisor.model.generation_config.pad_token_id = legal_advisor.tokenizer.eos_token_id

results = []
try:
    start_index = 0
    prompts = clerc_dataset['prompt']
    contexts = clerc_dataset['context']
    context_prefixes = clerc_dataset['context_prefix']
    inputs = [f"{context_prefix}\n\n{context}{legal_advisor.tokenizer.eos_token}{prompt}"
                for context_prefix, context, prompt in zip(context_prefixes, contexts, prompts)]
    generations = legal_advisor(
        inputs,
        do_sample=False,
        batch_size=batch_size,
        return_full_text=False,
        repetition_penalty=repetition_penalty,
    )
    generated_texts = [generated[-1]['generated_text'] for generated in generations]
    for index, (docid, generated_text, gold_text, prev_text) in enumerate(zip(clerc_dataset['docid'], generated_texts, clerc_dataset['gold_text'], clerc_dataset['previous_text'])):
        new_object = {
            "meta": {}
        }
        new_object["meta"]['docid'] = docid
        new_object["meta"]['index'] = (split, start_index + index)
        new_object["meta"]['gold_text'] = gold_text
        new_object["meta"]['previous_text'] = prev_text
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
