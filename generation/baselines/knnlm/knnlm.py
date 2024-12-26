from generation.baselines.cad.cad import CAD
from generation.workshop.time_monitor import GenerationTimeMonitor
from more_itertools import chunked
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Union, List, Optional
import faiss
import gc
import numpy as np
import torch
import torch.nn.functional as F


class KNNLM:
    def __init__(self, model_name: str, device: Union[int,str] = 0, compile: bool = True):
        device_map = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, use_cache=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        if compile:
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
        if len(tokens) <= max_length:
            chunks.append((tokens, 0))
        else:
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

    def prepare_references_for_datastore(self,
                                   references: List[List[str]],
                                   max_length: int = 512,
                                   overlap: float = 0.5) -> List[Tuple[torch.Tensor, int]]:
        processed_chunks = []
        # for context in tqdm(references, desc="Preparing references"):
        for context in references:
            assert len(context) == 2, "Each reference must be a tuple of (id, text)"
            context_text = context[1]
            tokenized_references = self.tokenizer(context_text, return_tensors="pt", truncation=False)
            chunks = self.chunk_tokens(tokenized_references["input_ids"][0], max_length, overlap)
            processed_chunks.extend(chunks)
        return processed_chunks  
    
    def construct_datastore_plus(self,
                                    context_texts: List[List[str]],
                                    overlap:float,
                                    layer_index=-1,
                                    k=10):
        assert overlap >= 0.0 and overlap <= 1.0, "Overlap must be between [0, 1]"
        assert isinstance(context_texts, list), "references must be a list of lists"
        batch_datastores = []
        for references in context_texts:
            keys = []
            values = []
            batch_size = 8
            for batch_references in chunked(references, batch_size):
                splits = self.prepare_references_for_datastore(batch_references)
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
            if self.use_faiss:
                keys = np.array(keys).reshape(-1, np.array(keys).shape[-1]).astype('float32')
                nneighbors = faiss.IndexFlatL2(keys.shape[-1])
                faiss.normalize_L2(keys)
                if torch.cuda.is_available():
                    nneighbors = faiss.index_cpu_to_all_gpus(nneighbors, ngpu=faiss.get_num_gpus())
                    # res = faiss.StandardGpuResources()
                    # nneighbors = faiss.index_cpu_to_gpu(res, self.device.index, nneighbors)
                nneighbors.add(keys)
            else:
                keys = np.array(keys).reshape(-1, np.array(keys).shape[-1])
                nneighbors = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean', n_jobs=-1)
                nneighbors.fit(keys)
            values = np.array(values).reshape(-1)
            batch_datastores.append({
                'store': nneighbors,
                'values': np.array(values)
            })
        return batch_datastores
    
    

    # more safe to use
    def construct_datastore_individually(self,
                                    context_texts: List[str],
                                    layer_index=-1,
                                    k=10):
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
            if self.use_faiss:
                keys = hidden_states[0].detach().cpu().numpy().astype('float32')
                nneighbors = faiss.IndexFlatL2(keys.shape[-1])
                faiss.normalize_L2(keys)
                if torch.cuda.is_available():
                    nneighbors = faiss.index_cpu_to_all_gpus(nneighbors, ngpu=faiss.get_num_gpus())
                    # res = faiss.StandardGpuResources()
                    # nneighbors = faiss.index_cpu_to_gpu(res, self.device.index, nneighbors)
                nneighbors.add(keys)
            else:
                keys = hidden_states[0].detach().cpu().numpy()
                nneighbors = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean', n_jobs=-1)
                nneighbors.fit(keys)
            values = next_tokens[0].detach().cpu().numpy()
            batch_datastores.append({
                'store': nneighbors,
                'values': values
            })
        return batch_datastores
    

    def compute_knn_probs(self, batch_datastores, query, k=10, temperature=1.0):
        batch_size = query.shape[0]
        knn_probs_list = []
        pad_idx = self.tokenizer.pad_token_id
        
        for i in range(batch_size):
            current_batch_datastore = batch_datastores[i]
            nneighbors = current_batch_datastore['store']
            values = current_batch_datastore['values'].reshape(-1)
            if self.use_faiss:
                query_flat = query[i].reshape(-1, query.shape[-1]).cpu().numpy().astype('float32')
                faiss.normalize_L2(query_flat)
                distances, indices = nneighbors.search(query_flat, k=k)
                distances = np.clip(distances, 1e-10, None)
                logits = 1 / distances
            else:
                query_flat = query[i].reshape(-1, query.shape[-1]).cpu().numpy()
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
                contexts: List[str],
                references: Optional[Union[List[str], List[List[str]]]] = None, 
                alpha: float = 0.5,
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
                min_length_ratio: float = 0.1,
                use_faiss: bool = False,
                generate_time_report: bool = False
                ) -> List[List[int]]:
        if generate_time_report:
            assert len(prompts) == len(contexts) == 1, "Time report mode only supports batch size 1"
            monitor = GenerationTimeMonitor()

        self.model.eval()
        self.use_faiss = use_faiss
        min_length = int(min_length_ratio * max_length)

        if generate_time_report:
            monitor.start_record("build_datastores")
        if 'plus' in variant:
            batch_datastores = self.construct_datastore_plus(references,
                                                             overlap=0.5,
                                                             layer_index=datastore_from_layer_index,
                                                             k=k)
        else:
            batch_datastores = self.construct_datastore_individually(references,
                                                                     layer_index=datastore_from_layer_index,
                                                                     k=k)
        if generate_time_report:
            monitor.stop_record("build_datastores")

        tokenized_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.model.config.max_position_embeddings)
        tokenized_inputs = {key: value.to(self.model.device) for key, value in tokenized_inputs.items()}
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        cache_position = torch.arange(tokenized_inputs['input_ids'].shape[1], dtype=torch.int64, device=self.device)
        model_kwargs = {
            "use_cache": True,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None
        }


        inputs_with_contexts = [f"{context}{self.tokenizer.eos_token}{prompt}" for context, prompt in zip(contexts, prompts)]
        tokenized_inputs_with_contexts = self.tokenizer(inputs_with_contexts, return_tensors="pt", padding=True, truncation=True, max_length=self.model.config.max_position_embeddings)
        tokenized_inputs_with_contexts = {key: value.to(self.model.device) for key, value in tokenized_inputs_with_contexts.items()}
        input_ids_with_contexts = tokenized_inputs_with_contexts['input_ids']
        attention_mask_with_contexts = tokenized_inputs_with_contexts['attention_mask']
        cache_position_with_contexts = torch.arange(tokenized_inputs['input_ids'].shape[1], dtype=torch.int64, device=self.device)
        model_kwargs_with_contexts = {
            "use_cache": True,
            "attention_mask": attention_mask_with_contexts,
            "cache_position": cache_position_with_contexts,
            "past_key_values": None
        }

        cur_len = 0
        batch_size = len(input_ids)
        previous_lambda = torch.zeros((batch_size, 1), device=self.device)

        alpha_tr = torch.full((batch_size, 1), alpha, device=self.device)
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)
        generated_tokens = [[] for _ in range(batch_size)] 

        entropies_history = [previous_lambda]
        lamba_history = [previous_lambda]


        if generate_time_report:
            average_tokenized_reference_length = sum(
                len(self.tokenizer(reference[1], return_tensors="pt")['input_ids'][0])
                for reference in references[0]
            ) / len(references)
            monitor.set_lengths(
                tokenized_context_length=self.tokenizer(contexts[0], return_tensors="pt")['input_ids'].shape[1],
                tokenized_prompt_length=self.tokenizer(prompts[0], return_tensors="pt")['input_ids'].shape[1],
                tokenized_reference_length=average_tokenized_reference_length,
                max_length=max_length
            )
        with torch.no_grad():
            while cur_len < max_length:
                if generate_time_report:
                    monitor.start_record("token")

                if "context" not in variant or "adacad" in variant:
                    model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                    outputs = self.model(**model_inputs,
                                         output_hidden_states=True,
                                         return_dict=True)
                    next_token_logits = outputs.logits[:, -1, :]
                    model_kwargs["attention_mask"] = torch.cat([model_kwargs["attention_mask"], torch.ones((batch_size, 1), device=self.device)], dim=-1)
                    model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
                    model_kwargs["past_key_values"] = outputs.past_key_values

                if "context" in variant or "adacad" in variant:
                    model_inputs_with_contexts = self.model.prepare_inputs_for_generation(input_ids_with_contexts, **model_kwargs_with_contexts)
                    outputs_with_contexts = self.model(**model_inputs_with_contexts,
                                                       output_hidden_states=True,
                                                       return_dict=True)
                    next_token_logits_with_contexts = outputs_with_contexts.logits[:, -1, :]
                    model_kwargs_with_contexts["attention_mask"] = torch.cat([model_kwargs_with_contexts["attention_mask"], torch.ones((batch_size, 1), device=self.device)], dim=-1)
                    model_kwargs_with_contexts["cache_position"] = model_kwargs_with_contexts["cache_position"][-1:] + 1
                    model_kwargs_with_contexts["past_key_values"] = outputs_with_contexts.past_key_values


                outputs_hidden_states = outputs_with_contexts.hidden_states
                final_token_logits = next_token_logits_with_contexts

                if "adacad" in variant:
                    alpha_tr = CAD.compute_jsd_per_batch(next_token_logits_with_contexts, next_token_logits)
                    outputs_hidden_states = outputs_with_contexts.hidden_states
                    final_token_logits = (1 + alpha_tr) * next_token_logits_with_contexts - alpha_tr * next_token_logits
                else:
                    if "context" not in variant:
                        outputs_hidden_states = outputs.hidden_states
                        final_token_logits = next_token_logits

                query_to_knn = outputs_hidden_states[datastore_from_layer_index][:, -1:, :]
                if generate_time_report:
                    monitor.start_record("knn")
                knn_next_token_probs = self.compute_knn_probs(batch_datastores, query_to_knn, k=k, temperature=temperature)
                if generate_time_report:
                    monitor.stop_record("knn")


                original_dtype = final_token_logits.dtype
                original_next_token_probs = F.softmax(final_token_logits / temperature, dim=-1).float()
                if strategy == 'entropy':
                    original_next_token_probs = torch.clamp(original_next_token_probs, min=1e-10)
                    entropy = -torch.sum(original_next_token_probs * torch.log(original_next_token_probs), dim=-1).unsqueeze(-1)
                    if entropy_strategy == 'exp':
                        lamba = torch.exp(-entropy).to(original_dtype)
                    elif entropy_strategy == 'exp_norm':
                        normalizer = torch.log(torch.tensor(original_next_token_probs.size(-1), dtype=original_next_token_probs.dtype))
                        normalized_entropy = entropy / normalizer
                        lamba = torch.exp(-normalized_entropy).to(original_dtype)
                    elif entropy_strategy == 'sig':
                        lamba = 1 / (1 + torch.exp(entropy - entropy_sigmoid_threshold))
                    lamba = torch.clamp(lamba, min=0.2, max=0.8) # to avoid extreme values
                    lamba = lambda_smoothing_factor * lamba + (1 - lambda_smoothing_factor) * previous_lambda
                    entropies_history.append(entropy)
                    lamba_history.append(lamba)
                    previous_lambda = lamba
                    assert torch.all(lamba >= 0.0) and torch.all(lamba <= 1.0), "Lambda must be between [0, 1]"


                next_token_probs = (1 - lamba) * knn_next_token_probs.to(original_dtype) + lamba * original_next_token_probs.to(original_dtype)
                next_token = self.predict_next_token(probs=next_token_probs, 
                                                    decoding_strategy=decoding_strategy, 
                                                    top_p=top_p_value, 
                                                    top_k=top_k_value, 
                                                    use_repetition_penalty=use_repetition_penalty, 
                                                    repetition_penalty_value=repetition_penalty_value, 
                                                    generated_tokens=[set(tokens) for tokens in generated_tokens])

                if generate_time_report:
                    monitor.stop_record("token")

                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                input_ids_with_contexts = torch.cat([input_ids_with_contexts, next_token.unsqueeze(-1)], dim=-1)


                cur_len += 1
                for i, token in enumerate(next_token.tolist()):
                    if unfinished_sents[i] == 1:
                        generated_tokens[i].append(token)
                    if unfinished_sents[i] == 1 and token == self.tokenizer.eos_token_id and cur_len > min_length:
                        unfinished_sents[i] = 0
                        sent_lengths[i] = cur_len
                if unfinished_sents.max() == 0:
                    break

        gc.collect()
        torch.cuda.empty_cache()

        if generate_time_report:
            report_json = monitor.get_report(generation_length=max(sent_lengths).item())
            return generated_tokens, report_json

        return generated_tokens


