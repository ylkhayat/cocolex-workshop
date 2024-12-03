# %%
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Literal, Union, List, Optional
import gc
import numpy as np
import random
import sys
import torch
import torch.nn.functional as F
import transformers

print(f"Python Version : {sys.version}")
print(f"Torch Version : {torch.__version__}")
print(f"Transformers Version : {transformers.__version__}")

# %%
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(1002)

# %%
class CAD:
    def __init__(self, model_name: str, device: Union[int,str] = 0):
        device_map = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, use_cache=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', truncation_side='left')
        self.device = device_map
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        # only for batch processing
        # if self.tokenizer.pad_token is None:
        #     special_tokens_dict = {'pad_token': '[PAD]'}
        #     self.tokenizer.add_special_tokens(special_tokens_dict)
        #     self.model.resize_token_embeddings(len(self.tokenizer))
        #     self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        #     print(f"[!] added [PAD] token to the tokenizer {self.tokenizer.pad_token_id}")
        
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
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep - 1 because we add the first one below)
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

        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None] # * logit 값이 Top-k의 토큰 중 가장 작은 값보다 작은 토큰의 인덱스 반환 
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

        # * Repetitin Penalty 참고 코드 : https://huggingface.co/transformers/v2.11.0/_modules/transformers/modeling_utils.html#PreTrainedModel.enforce_repetition_penalty_
        if use_repetition_penalty:
            assert repetition_penalty_value >= 1.0, "Repetition penalty must be >= 1."
            mask = torch.zeros_like(logits)
            for i, token_set in enumerate(generated_tokens):
                mask[i, list(token_set)] = 1.0
            penalty = torch.where(mask == 1.0, repetition_penalty_value, 1.0) # generated_tokens에 있는 토큰들은 penalty를 repetition_penalty_value로, 없는 토큰들은 1.0(현상 유지)으로 설정
            logits *= torch.where(logits < 0, penalty, 1.0/penalty) # if logit is smaller than 0, multiply with penalty, else divide by penalty
        
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

    def get_jsd(self, p, q):
        original_dtype = p.dtype
        p = p.to(torch.float32)
        q = q.to(torch.float32)

        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        if ((p + q) == 0).any():
            m = (0.5 * (p + q)).clamp_min(1e-9).log()
        else:
            m = (0.5 * (p + q)).log()
        if torch.any(p <= 0):
            p = p.clamp_min(1e-9)
        if torch.any(q <= 0):
            q = q.clamp_min(1e-9)

        result = 0.5 * (
            F.kl_div(m, p, reduction='batchmean', log_target=False) +
            F.kl_div(m, q, reduction='batchmean', log_target=False)
        )

        return result.to(original_dtype)
    
    def compute_jsd_per_batch(self, p, q):
        batch_size = p.size(0)
        jsd_values = []
        for i in range(batch_size):
            jsd = self.get_jsd(p[i], q[i])
            jsd_values.append(jsd)
        return torch.tensor(jsd_values, device=p.device, dtype=p.dtype)

    def calculate_eos_weight(self, entropy, entropy_with_contexts, beta=1.0):
        """
        Calculates a weight or penalty for the EOS token logits based on entropies.
        Parameters:
            entropy: Entropy of logits without context.
            entropy_with_contexts: Entropy of logits with context.
            beta: Scaling factor for sensitivity to entropy differences.
        Returns:
            A weight or penalty value for the EOS token logits.
        """
        entropy_diff = entropy_with_contexts - entropy
        weight = torch.tanh(beta * entropy_diff)  # Output is in the range [-1, 1]
        return weight
    
    def generate(self, 
                prompts: List[str], 
                contexts: Optional[List[str]] = None, 
                alpha: float = 0.5,
                method: Literal['cad', 'adacad'] = 'cad',
                max_length: int = 256,
                decoding_strategy: str = 'top_p',
                top_p_value: float = 0.9,
                top_k_value: int = 20,
                use_repetition_penalty: bool = False, 
                repetition_penalty_value: float = 1.0,
                temperature: float = 1.0
                ) -> List[List[int]]:
        self.model.eval()
        eos_token_id = self.tokenizer.eos_token_id
        
        # Tokenize 'prompts' and create attention masks
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
        # inputs_with_contexts = [f"{context}\n{prompt}" for context, prompt in zip(contexts, prompts)]
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

        # Initialize variables for generation loop
        cur_len = 0
        batch_size = len(input_ids)
        unfinished_sents = input_ids_with_contexts.new(batch_size).fill_(1)
        sent_lengths = input_ids_with_contexts.new(batch_size).fill_(max_length)
        min_length = max_length // 2  # 25% of the max length

        generated_tokens = [[] for _ in range(batch_size)] # e.g., [[4132, 102, 29402], [2378, 7893, 23001]]

        # Generate tokens
        with torch.no_grad():
            # pbar = tqdm(total=max_length, desc="CAD'ing" if method == 'cad' else "ADACAD'ing", position=0)
            while cur_len < max_length:
                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                outputs = self.model(**model_inputs,
                                     return_dict=True)
                next_token_logits = outputs.logits[:, -1, :]
                model_kwargs["attention_mask"] = torch.cat([model_kwargs["attention_mask"], torch.ones((batch_size, 1), device=self.device)], dim=-1)
                model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
                model_kwargs["past_key_values"] = outputs.past_key_values

                model_inputs_with_contexts = self.model.prepare_inputs_for_generation(input_ids_with_contexts, **model_kwargs_with_contexts)
                outputs_with_contexts = self.model(**model_inputs_with_contexts,
                                                   return_dict=True)
                next_token_logits_with_contexts = outputs_with_contexts.logits[:, -1, :]
                model_kwargs_with_contexts["attention_mask"] = torch.cat([model_kwargs_with_contexts["attention_mask"], torch.ones((batch_size, 1), device=self.device)], dim=-1)
                model_kwargs_with_contexts["cache_position"] = model_kwargs_with_contexts["cache_position"][-1:] + 1
                model_kwargs_with_contexts["past_key_values"] = outputs_with_contexts.past_key_values

                if method == 'adacad':
                    alpha_tr = self.compute_jsd_per_batch(next_token_logits_with_contexts, next_token_logits)
                    alpha = torch.clamp(alpha_tr, min=0.0, max=1.0).unsqueeze(-1)
                    # pbar.set_postfix({"Alpha": alpha.mean().item()})
                # eos_token_logits = next_token_logits[:, self.tokenizer.eos_token_id]
# 
                next_token_logits = (1 + alpha) * next_token_logits_with_contexts - alpha * next_token_logits
                # if cur_len < min_length:
                    # probs = F.softmax(next_token_logits, dim=-1)
                    # eos_prob = probs[:, eos_token_id]
                    # print(f"[!] eos probs — {eos_prob.item()}")
                    # next_token_logits[:, eos_token_id] = -np.inf  
            
                # next_token_logits[:, self.tokenizer.eos_token_id] = eos_token_logits
                
                # if torch.argmax(next_token_logits, dim=-1).item() == self.tokenizer.eos_token_id:
                #     probabilities = F.softmax(next_token_logits, dim=-1)
                #     entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)
                #     print(entropy)
                #     custom_temperature = 0.3
                #     custom_alpha = 0.8
                #     prob = torch.softmax(outputs.logits[:, -1, :] / custom_temperature, dim=-1)
                #     prob = torch.clamp(prob, min=1e-9)
                #     prob_with_contexts = torch.softmax(outputs_with_contexts.logits[:, -1, :] / custom_temperature, dim=-1)
                #     prob_with_contexts = torch.clamp(prob_with_contexts, min=1e-9)
                #     entropy = -torch.sum(prob * torch.log(prob), dim=-1)
                #     entropy_with_contexts = -torch.sum(prob_with_contexts * torch.log(prob_with_contexts), dim=-1)
                #     mean_entropy = (entropy + entropy_with_contexts) / 2
                #     eos_prob = prob[:, self.tokenizer.eos_token_id]
                #     eos_prob_with_contexts = prob_with_contexts[:, self.tokenizer.eos_token_id]
                #     eos_entropy = -torch.sum(prob[:, self.tokenizer.eos_token_id] * torch.log(prob[:, self.tokenizer.eos_token_id]))
                #     eos_entropy_with_contexts = -torch.sum(prob_with_contexts[:, self.tokenizer.eos_token_id] * torch.log(prob_with_contexts[:, self.tokenizer.eos_token_id]))
                #     # print(f"[!] model alpha: {alpha}")
                #     # print(f"[!] eos probs — {eos_prob.item()} | {eos_prob_with_contexts.item()}")
                #     # print(f"[!] eos entropies — {eos_entropy.item()} | {eos_entropy_with_contexts.item()}")
                #     # print(f"[!] entropies — {entropy.item()} | {entropy_with_contexts.item()} | mean: {mean_entropy.item()}")
                #     # if mean_entropy.item() >= 0.1:
                #     next_token_logits[:, self.tokenizer.eos_token_id] = mean_entropy * next_token_logits[:, self.tokenizer.eos_token_id]                if False:
                    # custom_temperature = 0.3
                    # prob = torch.softmax(outputs.logits[:, -1, :] / custom_temperature, dim=-1)
                    # prob = torch.clamp(prob, min=1e-9)
                    # prob_with_contexts = torch.softmax(outputs_with_contexts.logits[:, -1, :] / custom_temperature, dim=-1)
                    # prob_with_contexts = torch.clamp(prob_with_contexts, min=1e-9)
                    # entropy = -torch.sum(prob * torch.log(prob), dim=-1)
                    # entropy_with_contexts = -torch.sum(prob_with_contexts * torch.log(prob_with_contexts), dim=-1)
                    # weight = self.calculate_eos_weight(entropy, entropy_with_contexts)
                    # next_token_logits[:, self.tokenizer.eos_token_id] = weight * next_token_logits[:, self.tokenizer.eos_token_id]

                
                next_token = self.predict_next_token(logits=next_token_logits, 
                                                    decoding_strategy=decoding_strategy, 
                                                    top_p=top_p_value, 
                                                    top_k=top_k_value, 
                                                    use_repetition_penalty=use_repetition_penalty, 
                                                    repetition_penalty_value=repetition_penalty_value, 
                                                    generated_tokens=[set(tokens) for tokens in generated_tokens])
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                input_ids_with_contexts = torch.cat([input_ids_with_contexts, next_token.unsqueeze(-1)], dim=-1)

                cur_len += 1

                for i, token in enumerate(next_token.tolist()):
                    if unfinished_sents[i] == 1:
                        # if token == self.tokenizer.pad_token_id:
                        #     continue
                        generated_tokens[i].append(token)
                    if unfinished_sents[i] == 1 and token == self.tokenizer.eos_token_id:
                        unfinished_sents[i] = 0
                        sent_lengths[i] = cur_len

                if unfinished_sents.max() == 0:
                    break
            #     pbar.update(1)
            # pbar.close()

        # Return the generated tokens
        return generated_tokens


