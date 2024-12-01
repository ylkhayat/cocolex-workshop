from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple
from typing import Union, List, Optional
import torch
import torch.nn.functional as F

class EBD:
    def __init__(self,
                 model_name: str,
                 device: Union[int,str] = 0):
        device_map = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, use_cache=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.device = device_map
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def construct_context_based_inputs(self,
                                       prompts: List[str],
                                       context_prefix: str = None,
                                       contexts: List[List[Tuple[Optional[str], str]]] = None) -> Tuple[List[str], List[int]]:
        """
        Construct input strings with contexts for the model
        Args:
            prompts: List of prompts to generate completions for
            context_prefix: Prefix to add to the context before each prompt
            contexts: List of lists of tuples containing context IDs and context texts
        Returns:
            List of input strings with contexts
        """
        
        inputs_with_contexts = []
        inputs_with_contexts_to_prompt_index = []
        for prompt_index, prompt in enumerate(prompts):
            if contexts is not None:
                context_list = contexts[prompt_index]
                for context_id, context_text in context_list:
                    if len(context_text) > 0:
                        if context_id is not None:
                            context_prefix = context_text.format(context_id)
                        inputs_with_contexts.append(f"{context_prefix}\n{context_text} {self.tokenizer.eos_token} {prompt}")
                        inputs_with_contexts_to_prompt_index.append(prompt_index)
            else:
                inputs_with_contexts.append(prompt)
                inputs_with_contexts_to_prompt_index.append(prompt_index)
        return inputs_with_contexts
        
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

    def compute_maximum_entropy_layer_probs(self,
                                            input_ids: torch.Tensor,
                                            attention_mask: torch.Tensor,
                                            temperature: float = 1.0) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(input_ids,
                                 attention_mask=attention_mask,
                                 return_dict=True,
                                 output_hidden_states=True)
        batch_layer_entropies = []
        for hidden_state_l in outputs.hidden_states[0:]:  # hidden_state_l: (batch_size, seq_len, hidden_size)
            current_rep_hidden_states = hidden_state_l[:, -1:, :]  
            current_rep_logits = torch.matmul(current_rep_hidden_states, self.model.lm_head.weight.t())  # (batch_size, vocab_size)
            probs = F.softmax(current_rep_logits / temperature, dim=-1) 
            current_rep_entropy = -torch.sum(probs * torch.log(probs), dim=-1)  
            batch_layer_entropies.append(current_rep_entropy)
        entropies_tensor = torch.stack(batch_layer_entropies)
        maximum_entropy_layer_indices = torch.max(entropies_tensor, dim=0).indices.squeeze(-1)
        maximum_entropy_layer_probs = []
        for prompt_index, maximum_entropy_layer_index in enumerate(maximum_entropy_layer_indices):
            maximum_entropy_layer = outputs.hidden_states[maximum_entropy_layer_index][:, -1:, :]
            maximum_entropy_layer_logits = torch.matmul(maximum_entropy_layer, self.model.lm_head.weight.t())
            maximum_entropy_layer_probs.append(maximum_entropy_layer_logits[prompt_index])
        return torch.stack(maximum_entropy_layer_probs).squeeze(1)
    
    def compute_le_ens_scores(self,
                              input_ids_with_context: torch.Tensor,
                              attention_mask_with_context: torch.Tensor,
                              context_lengths: List[int]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(input_ids_with_context,
                                 attention_mask=attention_mask_with_context,
                                 return_dict=True,
                                 output_hidden_states=True)
        next_token_logits = outputs.logits[:, -1, :]
        batched_logits_per_prompt = next_token_logits.split(context_lengths)
        assert batched_logits_per_prompt[0].shape[0] == context_lengths[0]
        
        temperature = 1.0
        le_ens_scores = []
        for logits in batched_logits_per_prompt:
            probs = F.softmax(logits / temperature, dim=-1)
            le_ens_score_entropy = -torch.sum(probs * torch.log(probs), dim=-1)
            le_ens_scores.append(torch.sum(le_ens_score_entropy.unsqueeze(1) * torch.log(probs), dim=0))
        return torch.stack(le_ens_scores)
        
        
        
    def generate(self, 
                prompts: List[str],
                context_prefix: str = None,
                contexts: Optional[List[str]] = None, 
                beta: float = 0.5,
                max_length: int = 256,
                decoding_strategy: str = 'top_p',
                top_p_value: float = 0.9,
                top_k_value: int = 20,
                use_repetition_penalty: bool = False, 
                repetition_penalty_value: float = 1.0,
                ) -> List[List[int]]:

        tokenized_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.model.config.max_position_embeddings)
        tokenized_inputs = {key: value.to(self.model.device) for key, value in tokenized_inputs.items()}
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        
        
        inputs_with_contexts = self.construct_context_based_inputs(prompts,
                                                                   context_prefix,
                                                                   contexts)
        tokenized_inputs_with_contexts = self.tokenizer(inputs_with_contexts, return_tensors="pt", padding=True, truncation=True, max_length=self.model.config.max_position_embeddings)
        tokenized_inputs_with_contexts = {key: value.to(self.model.device) for key, value in tokenized_inputs_with_contexts.items()}
        input_ids_with_context = tokenized_inputs_with_contexts['input_ids']
        attention_mask_with_context = tokenized_inputs_with_contexts['attention_mask']
        
        cur_len = 0
        batch_size = len(input_ids)
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        generated_tokens = [[] for _ in range(batch_size)] # e.g., [[4132, 102, 29402], [2378, 7893, 23001]]
        context_lengths = [len(context) for context in contexts]

                
        with torch.no_grad():
            pbar = tqdm(total=max_length, desc="EBD'ing", position=0)
            while cur_len < max_length:

                maximum_entropy_layer_probs = self.compute_maximum_entropy_layer_probs(input_ids,
                                                                                       attention_mask)
                le_ens_scores = self.compute_le_ens_scores(input_ids_with_context,
                                                           attention_mask_with_context,
                                                           context_lengths)
                next_token_logits = F.softmax((1 + beta) * le_ens_scores - beta * torch.log(maximum_entropy_layer_probs), dim=-1)
                next_token = self.predict_next_token(logits=next_token_logits, 
                                                    decoding_strategy=decoding_strategy, 
                                                    top_p=top_p_value, 
                                                    top_k=top_k_value, 
                                                    use_repetition_penalty=use_repetition_penalty, 
                                                    repetition_penalty_value=repetition_penalty_value, 
                                                    generated_tokens=[set(tokens) for tokens in generated_tokens])

                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=self.device)], dim=-1)
                repeated_next_token = torch.cat([
                    next_token[i].repeat(context_lengths[i]) for i in range(len(next_token))
                ], dim=0)
                input_ids_with_context = torch.cat([input_ids_with_context, repeated_next_token.unsqueeze(-1)], dim=-1)
                attention_mask_with_context = torch.cat([attention_mask_with_context, torch.ones((input_ids_with_context.shape[0], 1), device=self.device)], dim=-1)
                
                cur_len += 1

                # Update generated tokens and check for completion
                for i, token in enumerate(next_token.tolist()):
                    if unfinished_sents[i] == 1:
                        generated_tokens[i].append(token)
                    if unfinished_sents[i] == 1 and token == self.tokenizer.eos_token_id:
                        unfinished_sents[i] = 0
                        sent_lengths[i] = cur_len

                # Check for sentences that are finished
                if self.tokenizer.eos_token_id is not None:
                    eos_in_sents = next_token == self.tokenizer.eos_token_id
                    is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                    sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                    unfinished_sents.mul_((~eos_in_sents).long())

                # Break if all sentences are finished : stop when there is a EOS token in each sentence, or if we exceed the maximul length
                if unfinished_sents.max() == 0:
                    break
                pbar.update(1)
        return generated_tokens
