import sys
if '/srv/elkhyo/lexquo' not in sys.path:
    sys.path.insert(0, '/srv/elkhyo/lexquo')
from generation.baselines.cad.cad import CAD
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
from slack_notifier import send_slack_notification
from tqdm import tqdm
import torch
import traceback

args = build_args_parser(method="cad")

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

only_count_valid = False
method = 'cad'
if strategy == 'constant':
    alphas = [0.3]
else:
    alphas = [None]
args.alphas = alphas
if strategy == 'adacad':
    method = "adacad"
args.method = method
args.only_count_valid = only_count_valid
print_args(args)

device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
try:
    def carry_experiment(alpha):
        args.alpha = alpha
        config = {
            "dataset_percentage": dataset_percentage,
            "dataset": dataset,
            "method": method,
            "setup": setup,
            "split": split,
            "top_k_passages": top_k_passages,
            "use_instructions": use_instructions,
        }
        preprocessor = ModelInputPreprocessor(config)
        exists, finished, all_results = load_experiment(args)
        if not args.override and finished and len(all_results) >= len(preprocessor.processed_dataset):
            print("[!] experiment already exists, skipping...")
            sys.exit(1 if args.check_only else 0)
        if args.check_only:
            sys.exit(0)
        results_output_path, meta_output_path, _ = build_path(args)
        cad_model = CAD(model_name=model_name, device=device.index)
        work_dataset, original_dataset = preprocessor.process_dataset(tokenizer=cad_model.tokenizer, max_tokens=cad_model.model.config.max_position_embeddings)
        needed_docids = work_dataset['docid'] # needed finished + needed not finished
        current_results = [result for result in all_results if result['meta']['docid'] in needed_docids] # needed finished + not needed finished
        computed_docids = [result['meta']['docid'] for result in current_results] # needed finished
        print(f"[!] used {len(current_results)} relevent records")
        try:
            is_truncated_global = False
            filted_work_dataset = work_dataset.filter(lambda record: record['docid'] not in computed_docids)
            print(f"[!] filtered {len(work_dataset) - len(filted_work_dataset)} records")
            record_counter = 0
            has_new_results = False
            for start_index, batch in enumerate(tqdm(filted_work_dataset.iter(batch_size=batch_size), desc="Batches", total=len(filted_work_dataset) // batch_size), start=0):
                if any(batch['is_truncated']):
                    is_truncated_global = True
                docids = batch['docid']
                prefixes = batch['previous_text']
                refs = batch['gold_text']
                prompts = batch['prompt']
                contexts = batch['context']
                context_prefixes = batch['context_prefix']
                contexts = [f"{context_prefix}\n\n{context}" for context_prefix, context in zip(context_prefixes, contexts)]
                assert len(contexts) == len(prompts)
                outputs = cad_model.generate(
                    prompts=prompts,
                    contexts=contexts,
                    max_length=max_new_tokens,
                    alpha=alpha,
                    method=method,
                    decoding_strategy=decoding_strategy,
                    use_repetition_penalty=repetition_penalty > 1.0,
                    repetition_penalty_value=repetition_penalty
                    )
                sent_lengths = [len(output) for output in outputs]
                if only_count_valid:
                    valid_sents = [sent_length > max_new_tokens//2 for sent_length in sent_lengths]
                    valid_outputs = [output for output, valid_sent in zip(outputs, valid_sents) if valid_sent]
                    if len(valid_outputs) == 0:
                        continue
                else:
                    valid_outputs = outputs
                generated_texts = cad_model.tokenizer.batch_decode(valid_outputs, skip_special_tokens=True)
                has_new_results = True
                for index, (docid, generated_text, gold_text, prev_text, prompt, context) in enumerate(zip(docids, generated_texts, refs, prefixes, prompts, contexts)):
                    new_object = {
                        "meta": {}
                    }
                    new_object["meta"]['docid'] = docid
                    new_object["meta"]['index'] = (split, (start_index * batch_size) + index)
                    new_object["meta"]['gold_text'] = gold_text
                    new_object["meta"]['previous_text'] = prev_text
                    new_object["meta"]['prompt'] = prompt
                    new_object["meta"]['context'] = context
                    new_object["gen"] = generated_text
                    current_results.append(new_object)
                    all_results.append(new_object)
                    record_counter += 1
                if record_counter % 10 == 0:
                    all_results = write_results(all_results, results_output_path, reference_dataset=original_dataset)
            all_results = write_results(all_results, results_output_path, reference_dataset=original_dataset)
            cad_model.model.to(torch.device('cpu'))
            args.is_truncated = is_truncated_global
            experiment_results = evaluate(current_results, device, work_dataset, args,
                                          has_new_results=has_new_results)
            if alpha is not None:
                experiment_results['alpha'] = alpha
            current_args = vars(args)
            save_metadata(experiment_results, meta_output_path, current_args)
            add_experiment(experiment_results, current_args)
            send_slack_notification(f"[!] Experiment completed: {results_output_path}!")
        except Exception as e:
            print(f"[!] Error: {e}")
            send_slack_notification(f"[x] Error in experiment: {results_output_path}!")
            sys.exit(1)
    for alpha in tqdm(alphas, desc="Alpha"):
        carry_experiment(alpha)
except Exception as e:
    print(f"[!] Error: {e}")
    traceback.print_exc()
    sys.exit(1)
    