from more_itertools import chunked
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Union, List, Literal, Optional
import gc
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import os
import torch.nn.functional as F


class KNNLM:
    def __init__(self, model_name: str, device: Union[int,str] = 0):
        print(f"[!] optimized KNNLM is initialized with model: {model_name}")
        device_map = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, use_cache=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left', use_fast=True)
        self.device = device_map
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        print(f"[!] added [PAD] token to the tokenizer {self.tokenizer.pad_token_id}")
        
    def chunk_tokens(self,
                     tokens, 
                     max_length: int,
                     overlap: float) -> List[List[str]]:
        if max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if not (0 <= overlap < 1):
            raise ValueError("overlap must be between 0 (inclusive) and 1 (exclusive)")
        chunks = []
        step = int(max_length * (1 - overlap))
        last_index = 0
        for i in range(0, len(tokens) - max_length + step, step):
            if i == 0:
                split = tokens[i:i+max_length]
                splits = (split, 0)
                last_index = i+max_length
            else:
                split = tokens[last_index-step: last_index+step]
                splits = (split, len(split) - step - 1) # -1 for the last key token
                last_index = last_index+step
            chunks.append(splits)
        return chunks

    def prepare_contexts_for_datastore(self,
                                   contexts: List[List[str]],
                                   max_length: int = 512,
                                   overlap: float = 0.5) -> List[Tuple[torch.Tensor, int]]:
        processed_chunks = []
        for context in tqdm(contexts, desc="Preparing contexts"):
            context_text = context[1]
            tokenized_contexts = self.tokenizer(context_text, return_tensors="pt", truncation=False)
            chunks = self.chunk_tokens(tokenized_contexts["input_ids"][0], max_length, overlap)
            processed_chunks.extend(chunks)
        return processed_chunks  
    
    def construct_datastore_plus(self,
                                    context_texts: List[List[str]],
                                    overlap:int,
                                    layer_index=-1):
        assert overlap >= 0.0 and overlap <= 1.0, "Overlap must be between [0, 1]"
        assert isinstance(context_texts, list), "Contexts must be a list of lists"
        print(f"[!] constructing plus datastore from layer '{layer_index}'")
        begin_time = time.process_time()
        batch_datastores = []
        for contexts in context_texts:
            keys = []
            values = []
            batch_size = 8
            for batch_contexts in tqdm(chunked(contexts, batch_size), desc="Processing contexts"):
                splits = self.prepare_contexts_for_datastore(batch_contexts)
                for split, pick_from in splits:
                    split = split.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        attention_mask = torch.ones_like(split, device=self.device)
                        outputs = self.model(split, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[layer_index][:, pick_from:-1, :].detach().cpu().numpy()
                    next_tokens = split[:, pick_from+1:].detach().cpu().numpy()
                    for j in range(hidden_states.shape[0]):
                        keys.extend(hidden_states[j])
                        values.extend(next_tokens[j])
            print(f"[!] collected keys: {len(keys)}")
            batch_datastores.append({
                'keys': np.array(keys),
                'values': np.array(values)
            })
        elapsed_time = time.process_time() - begin_time
        print(f"[!] datastore construction took {elapsed_time:.2f} seconds")
        return batch_datastores
    
    

    # more safe to use
    def construct_datastore_individually(self,
                                    context_texts: List[str],
                                    layer_index=-1):
        print(f"[!] constructing datastore: {layer_index}")
        begin_time = time.process_time()
        batch_datastores = []
        for text in context_texts:
            single_input = self.tokenizer(text,
                                          return_tensors="pt",
                                          padding=True,
                                          truncation=True, 
                                          max_length=self.model.config.max_position_embeddings
                                          ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**single_input, return_dict=True, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_index][:, :-1, :]
            next_tokens = single_input['input_ids'][:, 1:]
            batch_datastores.append({
                'keys': hidden_states[0].detach().cpu().numpy(),
                'values': next_tokens[0].detach().cpu().numpy()
            })
        elapsed_time = time.process_time() - begin_time
        print(f"[!] datastore construction took {elapsed_time:.2f} seconds")
        return batch_datastores
    

    def compute_knn_probs(self, batch_datastores, query, k=10, temperature=1.0):
        batch_size = query.shape[0]
        knn_probs_list = []
        pad_idx = self.tokenizer.pad_token_id
        
        for i in range(batch_size):
            current_batch_datastore = batch_datastores[i]
            keys = current_batch_datastore['keys'].reshape(-1, current_batch_datastore['keys'].shape[-1])
            values = current_batch_datastore['values'].reshape(-1)
            query_flat = query[i].reshape(-1, query.shape[-1]).cpu().numpy()
                
            nneighbors = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
            nneighbors.fit(keys)

            distances, indices = nneighbors.kneighbors(query_flat)
            logits = 50 / distances
            
            neighbor_values = values[indices]
            knn_logits = np.zeros((query_flat.shape[0], self.model.config.vocab_size))
            for j in range(query_flat.shape[0]):
                for l in range(k):
                    token_id = neighbor_values[j, l]
                    knn_logits[j, token_id] += logits[j, l]
            knn_logits[knn_logits == 0.0] = -10000.0
            knn_probs = np.exp(knn_logits) / np.exp(knn_logits).sum(axis=-1, keepdims=True)
            knn_probs_list.append(knn_probs)
        knn_probs_list = torch.tensor(np.concatenate(knn_probs_list, axis=0), device=self.device)
        return knn_probs_list
        
    
    def _top_p_sampling(self, 
                    probs: torch.Tensor, 
                    top_p: float = 0.9, 
                    min_tokens_to_keep: int = 1
                    ) -> torch.Tensor:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs


    def _top_k_sampling(self, 
                        probs: torch.Tensor, 
                        top_k: int = 20, 
                        min_tokens_to_keep: int = 1
                        ) -> torch.Tensor:
        top_k = min(max(top_k, min_tokens_to_keep), probs.size(-1))
        kth_values = torch.topk(probs, top_k)[0][..., -1, None]
        indices_to_remove = probs < kth_values
        probs[indices_to_remove] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True)

        return probs
    
    
    
    def predict_next_token(self, 
                        probs: torch.Tensor, 
                        decoding_strategy: str, 
                        top_p: float, 
                        top_k: int, 
                        use_repetition_penalty: bool, 
                        repetition_penalty_value: float, 
                        generated_tokens: List[set] = None
                        ) -> torch.Tensor:

        if use_repetition_penalty:
            assert repetition_penalty_value >= 1.0, "Repetition penalty must be >= 1."
            mask = torch.zeros_like(probs)
            for i, token_set in enumerate(generated_tokens):
                mask[i, list(token_set)] = 1.0
            penalty = torch.where(mask == 1.0, repetition_penalty_value, 1.0)
            probs = probs / penalty
            probs = probs / probs.sum(dim=-1, keepdim=True)

        if decoding_strategy == 'top_p':
            assert top_p is not None, "top_p must be provided for top_p sampling"
            probs = self._top_p_sampling(probs, top_p)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        elif decoding_strategy == 'top_k':
            assert top_k is not None, "top_k must be provided for top_k sampling"
            probs = self._top_k_sampling(probs, top_k)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        elif decoding_strategy == 'greedy':
            next_token = torch.argmax(probs, dim=-1)

        return next_token
    
    def generate(self, 
                prompts: List[str], 
                contexts: Optional[List[str]] = None, 
                lamba: float = 0.5,
                strategy: str = 'constant',
                max_length: int = 256,
                variant: str = 'normal',
                entropy_strategy: str = 'exp_norm', 
                entropy_sigmoid_threshold: float = 0.5,
                lambda_smoothing_factor: float = 0.3,
                decoding_strategy: str = 'top_p',
                top_p_value: float = 0.9,
                top_k_value: int = 20,
                k: int = 10,
                datastore_from_layer_index: int = -1,
                use_repetition_penalty: bool = False, 
                repetition_penalty_value: float = 1.0,
                temperature: float = 1.0,
                ) -> List[List[int]]:
        self.model.eval()
        
        if 'plus' in variant:
            batch_datastores = self.construct_datastore_plus(contexts,
                                                                  overlap=0.5,
                                                                  layer_index=datastore_from_layer_index)
        else:
            batch_datastores = self.construct_datastore_individually(contexts,
                                                            layer_index=datastore_from_layer_index)     
        tokenized_inputs = self.tokenizer(prompts,
                                          return_tensors="pt",
                                          padding=True,
                                          truncation=True,
                                          max_length=self.model.config.max_position_embeddings)
        tokenized_inputs = {key: value.to(self.model.device) for key, value in tokenized_inputs.items()}
        input_ids = tokenized_inputs['input_ids']
        cache_position = torch.arange(tokenized_inputs['input_ids'].shape[1], dtype=torch.int64, device=self.device)

        attention_mask = tokenized_inputs['attention_mask']
        
        cur_len = 0
        batch_size = len(input_ids)
        previous_lambda = torch.zeros((batch_size, 1), device=self.device)
        if strategy == 'entropy':
            print(f"[!] initial lamba: {previous_lambda}")
            print(f"[!] entropy strategy: {entropy_strategy}")        
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)
        generated_tokens = [[] for _ in range(batch_size)] 
        model_kwargs = {
            "use_cache": True,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None
        }
        entropies_history = [previous_lambda]
        lamba_history = [previous_lambda]
        desc = "KNNLM'ing" if strategy == 'constant' else "KNNLM'ing with Lambda Entropy"
        with torch.no_grad():
            pbar = tqdm(total=max_length, desc=desc, position=0)
            if strategy == 'constant':
                pbar.set_postfix({"Lambda": lamba})
            while cur_len < max_length:
                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                outputs = self.model(**model_inputs,
                                     output_hidden_states=True)
                next_token_logits = outputs.logits[:, -1, :] # (batch_size, vocab_size)
                query_to_knn = outputs.hidden_states[datastore_from_layer_index][:, -1:, :]
                model_kwargs["attention_mask"] = torch.cat([model_kwargs["attention_mask"], torch.ones((batch_size, 1), device=self.device)], dim=-1)
                model_kwargs["past_key_values"] = outputs.past_key_values
                model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
                knn_next_token_probs = self.compute_knn_probs(batch_datastores, query_to_knn, k=k, temperature=temperature)
                original_dtype = next_token_logits.dtype
                original_next_token_probs = F.softmax(next_token_logits / temperature, dim=-1).float()
                if strategy == 'entropy':
                    original_next_token_probs = torch.clamp(original_next_token_probs, min=1e-10)
                    entropy = -torch.sum(original_next_token_probs * torch.log(original_next_token_probs), dim=-1).unsqueeze(-1)
                    if entropy_strategy == 'exp':
                        lamba = torch.exp(-entropy).to(original_dtype)
                    elif entropy_strategy == 'exp_norm':
                        max_entropy = torch.log(torch.tensor(original_next_token_probs.size(-1), dtype=original_next_token_probs.dtype))
                        normalized_entropy = entropy / max_entropy
                        lamba = torch.exp(-normalized_entropy).to(original_dtype) 
                    elif entropy_strategy == 'sig':
                        lamba = 1 / (1 + torch.exp(entropy - entropy_sigmoid_threshold))
                    lamba = lambda_smoothing_factor * lamba + (1 - lambda_smoothing_factor) * previous_lambda
                    entropies_history.append(entropy)
                    lamba_history.append(lamba)
                    previous_lambda = lamba
                    
                    # print(f"[!] lamba: {lamba}")
                    assert torch.all(lamba >= 0.0) and torch.all(lamba <= 1.0), "Lambda must be between [0, 1]"
                next_token_probs = (1 - lamba) * knn_next_token_probs.to(original_dtype) + lamba * original_next_token_probs.to(original_dtype)

                next_token = self.predict_next_token(probs=next_token_probs, 
                                                    decoding_strategy=decoding_strategy, 
                                                    top_p=top_p_value, 
                                                    top_k=top_k_value, 
                                                    use_repetition_penalty=use_repetition_penalty, 
                                                    repetition_penalty_value=repetition_penalty_value, 
                                                    generated_tokens=[set(tokens) for tokens in generated_tokens])

                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

                
                cur_len += 1
                for i, token in enumerate(next_token.tolist()):
                    if unfinished_sents[i] == 1:
                        generated_tokens[i].append(token)
                    if unfinished_sents[i] == 1 and token == self.tokenizer.eos_token_id:
                        unfinished_sents[i] = 0
                        sent_lengths[i] = cur_len
                if unfinished_sents.max() == 0:
                    break
                pbar.update(1)
        pbar.close()
        
        # plot_figure = None
        # if strategy == 'entropy':
        #     entropies_history = torch.cat(entropies_history, dim=-1).cpu().numpy()
        #     lamba_history = torch.cat(lamba_history, dim=-1).cpu().numpy()
        #     timesteps = np.arange(lamba_history.shape[1])
        #     plt.figure(figsize=(8, 6))
        #     for i, (batch_lambda, batch_entropies) in enumerate(zip(lamba_history, entropies_history)):
        #         label_suffix = 'W Context' if i == 1 else 'W/O Context'
        #         plt.plot(timesteps, batch_lambda, marker='o', linestyle='-', label=f'Batch {label_suffix} - Lambda {i+1}', linewidth=2, color=f'C{i}')
        #         plt.plot(timesteps, batch_entropies, marker='x', linestyle='--', label=f'Batch {label_suffix} - Entropy {i+1}', linewidth=1, color=f'C{i}')
        #     os.makedirs('plots', exist_ok=True)
        #     plt.title(f"Entropy Function: {entropy_strategy}, Lambda Smoothing: {lambda_smoothing_factor}, Sigmoid Threshold: {entropy_sigmoid_threshold}")
        #     plt.xlabel("Timestep")
        #     plt.ylabel("Lambda Value")
        #     plt.grid(True, linestyle='--', alpha=0.6)
        #     plt.legend()
        #     plot_figure = plt.gcf()
        #     plt.close(plot_figure)

        gc.collect()
        torch.cuda.empty_cache()

        # Return the generated tokens
        return generated_tokens
        # return generated_tokens, plot_figure



