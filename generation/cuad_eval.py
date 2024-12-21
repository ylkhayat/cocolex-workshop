def model_generate(context, question, max_new_tokens, generate_method, device, full_context):
        
        prompt_context = """{text} \n\n Answer the following question based on above paragraph. Please output Not present in the contract, if the answer to question is not present in contract. \n\n{question} \nAnswer:""".format(text=context, question=question)
        prompt_only = """\n\n Answer the following question based on above contract. Please output Not present in the contract, if the answer to question is not present in contract. \n\n{question} \nAnswer:""".format(question=question)
        start_marker = '\nAnswer:'

      #   prompt_context = """{text} \n\n Answer the following question based on above paragraph. \n\n{question} \nAnswer:""".format(text=context, question=question)
      #   prompt_only = """\n\n Answer the following question based on above paragraph.  \n\n{question} \nAnswer:""".format(question=question)
      #   start_marker = '\nAnswer:'
        
        inputs = tokenizer(prompt_context, return_tensors="pt")
        input_without_context = tokenizer(prompt_only, return_tensors="pt")

        input_len = [len(each) for each in input_without_context.input_ids]

        for key in inputs:
            inputs[key] = inputs[key].to(device)

        knn_k = 10
        knn_temperature = 0.5
        lambda_  = 0.3
        knn_flag = False  
        adaknn_flag = False
        adacad_flag= False
        alpha = 0.0
        alpha_max = 0.5
        extend_flag = False


        if generate_method in ['default', 'cad', 'Adacad', 'knn', 'Adaknn', 'Extend_Adaknn']:
            if generate_method == 'cad':
                  alpha = 1.0
            elif generate_method == 'Adacad':
                  alpha = None
                  adacad_flag= True
            elif generate_method == 'knn':
                  knn_flag = True
            elif generate_method == 'Adaknn':
                  knn_flag = True
                  adaknn_flag = True
                  adacad_flag = True
            elif generate_method == 'Extend_Adaknn':
                  knn_flag = True
                  adaknn_flag = True
                  adacad_flag = True
                  extend_flag = True
                  
                    
            out = model.generate(
                            input_texts=[prompt_only],
                            use_context_aware=True,
                            contexts=[context],
                            max_length=50,
                            alpha=alpha,
                            decoding_strategy='greedy',
                            top_p_value=0.9,
                            use_repetition_penalty=True,
                            repetition_penalty_value=1.5,
                            device = device,
                            adacad_flag = adacad_flag,
                            knn_flag = knn_flag,
                            knn_k = knn_k,
                            knn_temperature = knn_temperature,
                            lambda_ = lambda_,
                            adaknn_flag = adaknn_flag,
                            alpha_max = alpha_max,
                            full_context = full_context,
                            extend_flag = extend_flag
                            )
            out_text = tokenizer.batch_decode(out)[0]
        else:
            out = model.greedy_search_pld(inputs.input_ids,
                                    input_len = input_len,
                                    attention_mask = inputs.attention_mask,
                                    stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=len(inputs.input_ids[0]) + max_new_tokens)]),
                                    draft_matching_window_size = 3,
                                    draft_num_candidate_tokens = 10,
                                    use_cache=True,
                                    pad_token_id=0,
                                    output_hidden_states=True,
                                    return_dict_in_generate=True)
            out_text = tokenizer.batch_decode(out.sequences,)[0]

        

        start_index = out_text.find(start_marker)
        if start_index != -1:
                out_text = out_text[start_index + len(start_marker):].strip()
                
        return out_text

from typing import Union, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

class CAD:
    def __init__(self, model_name: str, device: Union[int,str] = 0):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, use_cache=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if model_name.startswith('huggyllama') or model_name.startswith('mistralai') or model_name.startswith('meta-llama'):    
            special_tokens_dict = {'pad_token': '[PAD]'}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))

    def construct_datastore_single(self, context_inputs, context_attn, layer_index, take_from = 0):
        with torch.no_grad():
            outputs = self.model(context_inputs, attention_mask= context_attn, return_dict=True, output_hidden_states=True)
    
        
        hidden_states = outputs.hidden_states[layer_index][:, take_from:-1, :]  # Shape: (batch_size, context_len, hidden_dim)
        next_tokens = context_inputs[:, take_from + 1:]  # Shifted tokens as 'values'

        datastore = {
            'keys': hidden_states.detach().cpu().numpy(),  # Keys: hidden states
            'values': next_tokens.detach().cpu().numpy()   # Values: next tokens
        }
        return datastore

    def construct_datastore_whole(self, context_inputs, context_attn, layer_index):
        all_keys = []
        all_values = []
        chunk_size = 2048
        stride = 1024
        batch_size = 4  

        # Prepare chunks for batch processing
        chunk_batches = []
        attention_batches = []
        take_from_list = []  # Tracks the "take_from" offset for each chunk

        for start_idx in range(0, context_inputs.shape[1], stride):
            end_idx = start_idx + chunk_size

            chunk_batches.append(context_inputs[:, start_idx:end_idx])
            attention_batches.append(context_attn[:, start_idx:end_idx])
            take_from_list.append(0 if start_idx == 0 else stride)

            if len(chunk_batches) == batch_size or end_idx + stride > context_inputs.shape[1]:
                # Stack chunks into a single batch
                batch_inputs = torch.cat(chunk_batches, dim=0)  # Shape: (num_chunks_in_batch, chunk_size)
                batch_attn = torch.cat(attention_batches, dim=0)  # Shape: (num_chunks_in_batch, chunk_size)

                # Process batch using construct_datastore_single
                chunk_datastore = self.construct_datastore_single(batch_inputs, batch_attn, layer_index)

                # Split results back into chunks
                batch_hidden_states = chunk_datastore['keys']  # Shape: (num_chunks_in_batch, chunk_len, hidden_dim)
                batch_next_tokens = chunk_datastore['values']  # Shape: (num_chunks_in_batch, chunk_len)

                chunk_start_idx = 0
                for i, take_from in enumerate(take_from_list):
                    chunk_keys = batch_hidden_states[i, take_from:]  # Apply take_from offset
                    chunk_values = batch_next_tokens[i, take_from:]
                    all_keys.append(chunk_keys)
                    all_values.append(chunk_values)

                    chunk_start_idx += chunk_keys.shape[0]

                # Reset batch lists
                chunk_batches = []
                attention_batches = []
                take_from_list = []

        concatenated_keys = np.concatenate(all_keys, axis=0)  # Shape: (total_tokens, hidden_dim)
        concatenated_values = np.concatenate(all_values, axis=0)

        # Concatenate all keys and values
        datastore = {
        'keys': np.expand_dims(concatenated_keys, axis=0),  # Shape: (1, total_tokens, hidden_dim)
        'values': np.expand_dims(concatenated_values, axis=0)  # Shape: (1, total_tokens)
         }


        #print(datastore['keys'].shape, datastore['values'].shape)
        return datastore



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
    
    def compute_knn_probabilities(self, datastore, query, k=10, temperature=1.0, vocab_size=32001):
        keys = datastore['keys'].reshape(-1, datastore['keys'].shape[-1])  # Ensure 2D (num_entries, dim)
        values = datastore['values'].reshape(-1)  # Ensure 1D (num_entries,)

        query_flat = query.reshape(-1, query.shape[-1])  # Flatten queries for batch processing

        # Fit kNN model
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
        nbrs.fit(keys)

        distances, indices = nbrs.kneighbors(query_flat)
        logits = 50 / distances


        neighbor_values = values[indices].astype(int)  # Map indices to token IDs, ensure integer type

        # Aggregate logits into a vocabulary-sized tensor
        knn_logits = np.zeros((query_flat.shape[0], vocab_size))  # (num_queries, vocab_size)
        for j in range(query_flat.shape[0]):
                for l in range(k):
                    token_id = neighbor_values[j, l]
                    knn_logits[j, token_id] += logits[j, l]
        
        knn_logits[knn_logits == 0.0] = -10000


        # Compute softmax probabilities
        knn_probs = np.exp(knn_logits) / np.exp(knn_logits).sum(axis=-1, keepdims=True)
        return knn_probs



    def _top_k_sampling(self, 
                        logits: torch.Tensor, 
                        top_k: int = 20, 
                        filter_value: float = -float("Inf"), 
                        min_tokens_to_keep: int = 1
                        ) -> torch.Tensor :

        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
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
    
    def calculate_jsd(self, logits_with_contexts, logits):
        probs_with_contexts = F.softmax(logits_with_contexts, dim=-1)
        probs = F.softmax(logits, dim=-1)

        m = 0.5 * (probs_with_contexts + probs)  # Average distribution

        jsd = 0.5 * (torch.sum(probs_with_contexts * (torch.log(probs_with_contexts + 1e-10) - torch.log(m + 1e-10)), dim=-1) +
                             torch.sum(probs * (torch.log(probs + 1e-10) - torch.log(m + 1e-10)), dim=-1))
        return jsd
    

    def calculate_p_max_and_entropy(self,logits):
        probabilities = F.softmax(logits, dim=-1)
        p_max = torch.max(probabilities, dim=-1).values
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)  # Adding small value to avoid log(0)
        
        return p_max, entropy

    def generate(self, 
                input_texts: List[str], 
                contexts: Optional[List[str]] = None, 
                use_context_aware: bool = True,
                alpha:Optional[float] = 0.5,
                max_length: int = 256,
                decoding_strategy: str = 'top_p',
                top_p_value: float = 0.9,
                top_k_value: int = 20,
                use_repetition_penalty: bool = False, 
                repetition_penalty_value: float = 1.0,
                device:str = 'cuda',
                adacad_flag: bool = False,
                knn_flag: bool = False,
                knn_k: int = 5, 
                knn_temperature: float = 1.0 ,
                datastore_layer: int = -1,
                lambda_:float = 0.3,
                adaknn_flag:bool = False,
                alpha_max: float = 0.3,
                full_context: List[str] = None,
                extend_flag: bool = False
                ) -> List[List[int]]:

        # Tokenize 'input_texts' and create attention masks
        tokenized_inputs = self.tokenizer(input_texts, return_tensors="pt")
        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)

        if contexts:
            inputs_with_contexts = [context + self.tokenizer.eos_token + input_text for context, input_text in zip(contexts, input_texts)]
            tokenized_inputs_with_contexts = self.tokenizer(inputs_with_contexts, return_tensors="pt")
            input_ids_with_contexts = tokenized_inputs_with_contexts['input_ids'].to(device)
            attention_mask_with_contexts = tokenized_inputs_with_contexts['attention_mask'].to(device)

            if knn_flag and extend_flag is not True:
                tokenized_context = self.tokenizer(contexts, return_tensors="pt")
                input_ids_contexts = tokenized_context['input_ids'].to(device)
                attention_mask_contexts = tokenized_context['attention_mask'].to(device)
                datastore = self.construct_datastore_single(input_ids_contexts , attention_mask_contexts,  layer_index=datastore_layer)
            
            if knn_flag and extend_flag is True:
                inputs_context = self.tokenizer(full_context, return_tensors="pt",truncation=False, add_special_tokens=False)
                ids_context = inputs_context["input_ids"].to(device)
                attention_context = inputs_context['attention_mask'].to(device)

                datastore = self.construct_datastore_whole(ids_context, attention_context,layer_index=datastore_layer)



        # Initialize variables for generation loop
        cur_len = 0
        batch_size = len(input_ids)
        unfinished_sents = input_ids_with_contexts.new(batch_size).fill_(1)
        sent_lengths = input_ids_with_contexts.new(batch_size).fill_(max_length)

        generated_tokens = [[] for _ in range(batch_size)] # e.g., [[4132, 102, 29402], [2378, 7893, 23001]]
        
        
        # Generate tokens
        with torch.no_grad():
            while cur_len < max_length:
                # * Context-aware Decoding
                if contexts:
                    outputs_with_contexts = self.model(input_ids_with_contexts, attention_mask=attention_mask_with_contexts,return_dict=True, output_hidden_states=True)
                    next_token_logits_with_contexts = outputs_with_contexts.logits[:, -1, :]

                    if adaknn_flag is True:
                        p_max, entropy = self.calculate_p_max_and_entropy(next_token_logits_with_contexts)
                        transformed_entropy = 1 / (entropy + 1)
                        #transformed_entropy = torch.exp(-entropy)
                        #confidence_score = (p_max * transformed_entropy) ** 0.5
                        confidence_score = transformed_entropy

                        # normalizer = torch.log(torch.tensor(next_token_logits_with_contexts.size(-1)))
                        # normalized_entropy = entropy / normalizer
                        # lambda_= torch.exp(-normalized_entropy).detach().cpu().numpy()


                        #print(f' with context  {confidence_score}')
                        lambda_ = max(lambda_, confidence_score.detach().cpu().numpy())

                if use_context_aware:
                    outputs = self.model(input_ids, attention_mask=attention_mask,return_dict=True, output_hidden_states=True)
                    next_token_logits = outputs.logits[:, -1, :] # (batch_size, vocab_size)
                    

                    if adacad_flag is True:
                        alpha = self.calculate_jsd(next_token_logits_with_contexts, next_token_logits)
                        alpha = max(alpha, alpha_max)

                    next_token_logits_with_contexts = (1 + alpha) * next_token_logits_with_contexts - alpha * next_token_logits
                
                if knn_flag:                    
                    current_rep = outputs_with_contexts.hidden_states[datastore_layer][:, -1:, :]  # Shape: (batch_size, 1, hidden_dim)
                    
                    # current_rep_without = outputs.hidden_states[datastore_layer][:, -1:, :]
                    
                    # current_rep = (1 + alpha) * current_rep - alpha * current_rep_without
                    
                    query = current_rep.detach().cpu().numpy()

                    knn_probs = self.compute_knn_probabilities(datastore, query, k=knn_k)
                    lm_logits =  next_token_logits_with_contexts #outputs_with_contexts.logits[:, -1, :] 
                    
                    lm_probs = torch.softmax(lm_logits, dim=-1).detach().cpu().numpy()

                    #print(f'{alpha}, {lambda_}')
                    next_token_logits_with_contexts = torch.tensor((1-lambda_)* knn_probs + (lambda_) * lm_probs)
 

                # Predict next token according to decoding strategy
                next_token = self.predict_next_token(logits=next_token_logits_with_contexts, 
                                                    decoding_strategy=decoding_strategy, 
                                                    top_p=top_p_value, 
                                                    top_k=top_k_value, 
                                                    use_repetition_penalty=use_repetition_penalty, 
                                                    repetition_penalty_value=repetition_penalty_value, 
                                                    generated_tokens=[set(tokens) for tokens in generated_tokens]).to(device)
                

                # Handle EOS token and padding
                if self.tokenizer.eos_token_id is not None:
                    tokens_to_add = next_token * unfinished_sents + (self.tokenizer.pad_token_id) * (1 - unfinished_sents)
                else:
                    tokens_to_add = next_token

                # Update input_ids and attention masks for the next forward pass
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, unfinished_sents.unsqueeze(-1)], dim=-1)
                input_ids_with_contexts = torch.cat([input_ids_with_contexts, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask_with_contexts = torch.cat([attention_mask_with_contexts, unfinished_sents.unsqueeze(-1)], dim=-1)

                cur_len += 1

                # Update generated tokens and check for completion
                for i, token in enumerate(tokens_to_add.tolist()):
                    if unfinished_sents[i] == 1:
                        generated_tokens[i].append(token)

                # Check for sentences that are finished
                if self.tokenizer.eos_token_id is not None:
                    eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
                    is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                    sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                    unfinished_sents.mul_((~eos_in_sents).long())

                # Break if all sentences are finished : stop when there is a EOS token in each sentence, or if we exceed the maximul length
                if unfinished_sents.max() == 0:
                    break

        # Return the generated tokens
        return generated_tokens

import nltk
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')
dense_model = SentenceTransformer('all-MiniLM-L6-v2')

def process_and_rank_sentences(context, question, min_length=50, method="dense", top_k=5):
    sentences = nltk.sent_tokenize(context)
    merged_sentences = []
    original_indices = []  # Track original indices
    current_sentence = ""
    current_index = 0  # Keep track of the current sentence index

    for i, sentence in enumerate(sentences):
        if len(current_sentence) + len(sentence) < min_length:
            current_sentence += " " + sentence
        else:
            if current_sentence:
                merged_sentences.append(current_sentence.strip())
                original_indices.append(current_index)  # Record the starting index for this merged sentence
            current_sentence = sentence
            current_index = i
    if current_sentence:
        merged_sentences.append(current_sentence.strip())
        original_indices.append(current_index)  # Record the last sentence's index
    
    if method == "dense":
        question_embedding = dense_model.encode(question, convert_to_tensor=True)
        sentence_embeddings = dense_model.encode(merged_sentences, convert_to_tensor=True)
        
        scores = util.cos_sim(question_embedding, sentence_embeddings)[0]
        ranked_sentences = sorted(
            zip(merged_sentences, scores.tolist(), original_indices), key=lambda x: x[1], reverse=True
        )
        
    elif method == "bm25":
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in merged_sentences]
        bm25 = BM25Okapi(tokenized_sentences)
        
        scores = bm25.get_scores(nltk.word_tokenize(question))
        ranked_sentences = sorted(
            zip(merged_sentences, scores, original_indices), key=lambda x: x[1], reverse=True
        )

    top_k_sentences = sorted(ranked_sentences[:top_k], key=lambda x: x[2])

    merged_text = ' '.join([sentence for sentence, _, _ in top_k_sentences])
    
    return merged_text


from datasets import load_dataset
test_dataset = load_dataset("theatticusproject/cuad-qa",split='test')


from tqdm import tqdm

def populate_generations(test_dataset, sample, generate_method, retrieval, top_k, device):
    generated_ans = []
    context_used = []

    for id in tqdm(range(sample)):
        full_context = test_dataset[id]['context']
        question = test_dataset[id]['question']

        if retrieval is not None:
            context = process_and_rank_sentences(full_context, question, min_length=50, method=retrieval)
            # top_k_sentences = ranked_sentences[:top_k]
            # context = " ".join(sentence for sentence, _ in top_k_sentences)
            
        # gold_answer = " ".join(test_dataset[id]['answers']['text'])
        # if gold_answer=='':
        #     gold_answer = 'The answer is not present in the contract.'
        else:
            context = full_context
        
        max_new_tokens = 50

        ans_context =  model_generate(context, question, max_new_tokens, generate_method, device, full_context = full_context)

        # print(f'Generated Answer: {ans_context}')
        # print(f'Answer Gold: {gold_answer}')
        generated_ans.append(ans_context)
        context_used.append(context)
    return generated_ans, context_used

# from evaluation.align_score.src.alignscore import AlignScore
# scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda:0', ckpt_path='https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt', evaluation_mode='bin')

# def alignscore_scorer(contexts, claims):
#     scores = []
#     for context, claim in zip(contexts, claims):
#         scores.append(scorer.score(contexts=[context], claims=[claim])[0])
#     return scores

import numpy as np

def hit_rate(context, original_answers):
    hits = sum(1 for answer in original_answers if answer in context)
    hit_rate_score = hits / len(original_answers) if len(original_answers)!=0 else 1
    return hit_rate_score

# def score_function(test_dataset, generated_ans, input_contexts):
#     #input_contexts = [x['context'] for x in test_dataset]
#     gold_answers_list = [x['answers']['text'] for x in test_dataset]
#     gold_answers = [" ".join(x['answers']['text']) for x in test_dataset]
#     hit_rate_scores = [hit_rate(context, answer_list) for context, answer_list in zip(input_contexts, gold_answers_list)] 

#     gold_answers = ['Not present in the contract.' if answer == '' else answer for answer in gold_answers]
#     generated_ans = ['None' if answer == '' else answer for answer in generated_ans]

#     correct_scores =  alignscore_scorer(generated_ans, gold_answers)

#     faith_gen_ans, faith_context = zip(*[(gen, context) for gen, context in zip(generated_ans, input_contexts) if gen != 'None' and gen != 'Not present in the contract.</s>'])

#     faith_scores =  alignscore_scorer(faith_context, faith_gen_ans)

#     faith_scores_full = [np.nan] * len(generated_ans)
#     faith_index = 0

#     for i, gen in enumerate(generated_ans):
#         if gen != 'None' and gen != 'Not present in the contract.</s>':
#             faith_scores_full[i] = faith_scores[faith_index]
#             faith_index += 1
            
#     return correct_scores, faith_scores_full, hit_rate_scores

from datasets import load_dataset
test_dataset = load_dataset("theatticusproject/cuad-qa",split='test')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
device = torch.device('cuda:0')

model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1" #"lmsys/vicuna-7b-v1.5"  #"facebook/opt-1.3b" #
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model = CAD(model_name=model_name_or_path, device="cuda:0")

generate_method = 'Extend_Adaknn'
retrieval= 'dense'
top_k = 20

generated_answers, used_context = populate_generations(test_dataset, len(test_dataset), generate_method, retrieval, top_k, device)
# correct_scores, faith_scores, hit_rate_scores = score_function(test_dataset, generated_answers, used_context)


gold_answers = [" ".join(x['answers']['text']) for x in test_dataset]
gold_answers = ['Not present in the contract.' if answer == '' else answer for answer in gold_answers]

import json


data = {}
for id, (ref, act, correct_score, faith_score, hit_rate_score) in enumerate(zip(gold_answers, generated_answers, correct_scores, faith_scores, hit_rate_scores)):
    data[id] = {
        "reference": ref,
        "actual": act,
        "correct_score": correct_score,
        "faith_score": faith_score,
        "hit_rate_score": hit_rate_score
    }

# Write to a JSON file
with open(f'cuad_FINAL_{generate_method}_{retrieval}_{top_k}.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
