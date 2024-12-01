import sys
if '/srv/elkhyo/lexquo' not in sys.path:
    sys.path.insert(0, '/srv/elkhyo/lexquo')
from experiments.generation.baselines.knnlm.knnlm import KNNLM
from experiments.generation.workshop.experiment_utils import add_experiment, build_args_parser, evaluate, print_args, reshape_and_save_experiment_results, setup_dataset
from experiments.slack_notifier import send_slack_notification
from tqdm import tqdm
import os
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
if "plus" in variant:
    method = f"{method}-plus"
method = f"{method}-{strategy}"
if strategy == 'constant':
    lamdas = [0.5]
    args.lamdas = lamdas
layers = [-1]
args.method = method
print_args(args)

device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

config = {
    "dataset_percentage": dataset_percentage,
    "setup": setup,
    "split": split,
    "top_k_passages": top_k_passages,
    "use_instructions": use_instructions
}

knnlm_model = KNNLM(model_name=model_name, device=device.index)
clerc_dataset = setup_dataset(config,
                              tokenizer=knnlm_model.tokenizer)


def carry_experiment(lamba,
                     strategy,
                     k,
                     layer_index):
    results = []
    os.makedirs("./basement", exist_ok=True)
    try:
        start_index = 0
        for batch in tqdm(clerc_dataset.iter(batch_size=batch_size), desc="Processing batches", total=len(clerc_dataset) // batch_size):
            prefixes = batch['previous_text']
            docids = batch['docid']
            refs = batch['gold_text']
            contexts = batch['context']
            context_prefixes = batch['context_prefix']
            prompts = batch['prompt']
            if "context" in variant:
                prompts = [f"{context_prefix}\n\n{context}{knnlm_model.tokenizer.eos_token}{prompt}" for context_prefix, context, prompt in zip(context_prefixes, contexts, prompts)]
            if "plus" in variant:
                # contexts = [['\n'.join(document_parts) for document_parts in meta['oracle_documents']] for meta in batch['meta']]
                contexts = [meta['oracle_documents'] for meta in batch['meta']]
            outputs = knnlm_model.generate(
                prompts=prompts,
                contexts=contexts,
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
            for index, (docid, generated_text, gold_text, prev_text) in enumerate(zip(docids, generated_texts, refs, prefixes)):
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
        experiment_results['knn_k'] = k
        experiment_results['lamba'] = lamba
        experiment_results['layer_index'] = layer_index
        experiment_results['results'] = results
        start_index += batch_size
        results_output_path, _= reshape_and_save_experiment_results(experiment_results, vars(args))
        add_experiment(experiment_results, vars(args))
        send_slack_notification(f"Experiment completed: {results_output_path}!")
    except Exception as e:
        print(f"[!] Error: {e}")
        send_slack_notification(f"Error in experiment: {results_output_path}!")
        raise e
    
if strategy == 'constant':
    for lamba in tqdm(lamdas, desc="Lamba"):
        for layer_index in layers:
            carry_experiment(lamba, "constant", 10, layer_index)
else:
    for layer_index in layers:
        carry_experiment(None, "entropy", 10, layer_index)