from generation.workshop.time_monitor import GenerationTimeMonitor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Union, List
import numpy as np
import random
import sys
import torch
import torch.nn.functional as F
import transformers

print(f"Python Version : {sys.version}")
print(f"Torch Version : {torch.__version__}")
print(f"Transformers Version : {transformers.__version__}")

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(1002)

class RAG:
    def __init__(self, model_name: str, device: Union[int,str] = 0, compile: bool = True):
        device_map = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=device_map, 
            use_cache=True, 
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.float16
        )
        if compile:
            self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            padding_side='left', 
            truncation_side='left'
        )
        self.device = device_map
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        
    def _top_p_sampling(self, 
                        logits: torch.Tensor, 
                        top_p: float = 0.9, 
                        filter_value: float = -float("Inf"), 
                        min_tokens_to_keep: int = 1
                        ) -> torch.Tensor :
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

        return logits

    def _top_k_sampling(self, 
                        logits: torch.Tensor, 
                        top_k: int = 20, 
                        filter_value: float = -float("Inf"), 
                        min_tokens_to_keep: int = 1
                        ) -> torch.Tensor :
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits

    def predict_next_token(self, 
                           logits: torch.Tensor, 
                           decoding_strategy: str, 
                           top_p: float, 
                           top_k: int, 
                           use_repetition_penalty: bool, 
                           repetition_penalty_value: float, 
                           generated_tokens: List[set] = None
                           ) -> torch.Tensor :

        if use_repetition_penalty:
            assert repetition_penalty_value >= 1.0, "Repetition penalty must be >= 1."
            mask = torch.zeros_like(logits)
            for i, token_set in enumerate(generated_tokens):
                mask[i, list(token_set)] = 1.0
            penalty = torch.where(mask == 1.0, repetition_penalty_value, 1.0)
            logits *= torch.where(logits < 0, penalty, 1.0 / penalty)
        
        if decoding_strategy == 'top_p':
            assert top_p is not None, "top_p must be provided for top_p sampling"
            logits = self._top_p_sampling(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()
        elif decoding_strategy == 'top_k':
            assert top_k is not None, "top_k must be provided for top_k sampling"
            logits = self._top_k_sampling(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()
        elif decoding_strategy == 'greedy':
            next_token = torch.argmax(logits, dim=-1)
        return next_token
    
    def generate(self, 
                 prompts: List[str], 
                 contexts: Optional[List[str]] = None, 
                 max_length: int = 256,
                 decoding_strategy: str = 'top_p',
                 top_p_value: float = 0.9,
                 top_k_value: int = 20,
                 use_repetition_penalty: bool = False, 
                 repetition_penalty_value: float = 1.0,
                 temperature: float = 1.0,
                 min_length_ratio: float = 0.1,
                 generate_time_report: bool = False
                ):
        if generate_time_report:
            assert len(prompts) == len(contexts) == 1, "Time report mode only supports batch size 1"
            print("[!] time report mode is on")
        self.model.eval()
        min_length = int(min_length_ratio * max_length)
        inputs_with_contexts = [f"{context}{self.tokenizer.eos_token}{prompt}" for context, prompt in zip(contexts, prompts)]
        tokenized_inputs_with_contexts = self.tokenizer(
            inputs_with_contexts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.model.config.max_position_embeddings
        )
        tokenized_inputs_with_contexts = {key: value.to(self.model.device) for key, value in tokenized_inputs_with_contexts.items()}
        input_ids = tokenized_inputs_with_contexts['input_ids']
        attention_mask = tokenized_inputs_with_contexts['attention_mask']
        cache_position = torch.arange(input_ids.shape[1], dtype=torch.int64, device=self.device)

        model_kwargs = {
            "use_cache": True,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None
        }

        cur_len = 0
        batch_size = len(input_ids)
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        generated_tokens = [[] for _ in range(batch_size)]

        if generate_time_report:
            monitor = GenerationTimeMonitor(
                tokenized_context_length=self.tokenizer(contexts[0], return_tensors="pt")['input_ids'].shape[1],
                tokenized_prompt_length=self.tokenizer(prompts[0], return_tensors="pt")['input_ids'].shape[1],
                tokenized_reference_length=None,
                max_length=max_length
            )

        with torch.no_grad():
            while cur_len < max_length:
                
                if generate_time_report:
                    monitor.start_record("token")
                    
                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                outputs = self.model(**model_inputs, return_dict=True)
                next_token_logits = outputs.logits[:, -1, :]

                model_kwargs["attention_mask"] = torch.cat(
                    [model_kwargs["attention_mask"], torch.ones((batch_size, 1), device=self.device)], 
                    dim=-1
                )
                model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
                model_kwargs["past_key_values"] = outputs.past_key_values

                next_token = self.predict_next_token(
                    logits=next_token_logits, 
                    decoding_strategy=decoding_strategy, 
                    top_p=top_p_value, 
                    top_k=top_k_value, 
                    use_repetition_penalty=use_repetition_penalty, 
                    repetition_penalty_value=repetition_penalty_value, 
                    generated_tokens=[set(tokens) for tokens in generated_tokens]
                )

                if generate_time_report:
                    monitor.stop_record("token")
                    
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                cur_len += 1

                for i, token in enumerate(next_token.tolist()):
                    if unfinished_sents[i] == 1:
                        generated_tokens[i].append(token)
                    if unfinished_sents[i] == 1 and token == self.tokenizer.eos_token_id and cur_len > min_length:
                        unfinished_sents[i] = 0
                        sent_lengths[i] = cur_len

                if unfinished_sents.max() == 0:
                    break
        if generate_time_report:
            report_json = monitor.get_report(generation_length=max(sent_lengths).item())
            return generated_tokens, report_json
        
        return generated_tokens