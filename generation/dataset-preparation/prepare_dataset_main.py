from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse
import bm25s
import faiss
import json
import numpy as np
import os
import torch

top_k = 10

# encoder_name = "jhu-clsp/LegalBERT-DPR-CLERC-ft"
encoder_name = "jhu-clsp/BERT-DPR-CLERC-ft"
tokenizer = AutoTokenizer.from_pretrained(encoder_name, truncation_side='left')
model = AutoModel.from_pretrained(encoder_name)
# model = torch.compile(model)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)

clean_encoder_name = encoder_name.replace("/", "_")
data_dir = "dense_relevant_passages_oracle_documents"
# data_dir = "dense_oracle_passages_oracle_documents"
data_dir = f"{data_dir}/{clean_encoder_name}"

print(f"[!] data_dir: {data_dir}")

num_proc = os.cpu_count()

current_workshop_hf_name = f"ECHR-generation-workshop"
current_dataset = load_dataset(f"ylkhayat/{current_workshop_hf_name}")

def normalize_embeddings(embeddings):
    return embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True)

# def embed_texts_pooler(texts):
#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=model.config.max_position_embeddings).to(device)
#     with torch.no_grad():
#         embeddings = model(**inputs).pooler_output
#     embeddings = normalize_embeddings(embeddings).cpu().numpy()
#     return embeddings

def embed_texts(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=model.config.max_position_embeddings).to(device)
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        token_embeddings = model(**inputs).last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
    embeddings = normalize_embeddings(embeddings).cpu().numpy()
    return embeddings


def search_with_embeddings(all_queries, all_passages_text, all_passages_text_with_ids):
    query_embeddings = embed_texts(all_queries)
    passage_embeddings = embed_texts(all_passages_text)
    dim = passage_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embeddings)
    distances, indices = index.search(query_embeddings, top_k)
    top_passages_batch = [[all_passages_text_with_ids[index] for index in index_arr] for index_arr in indices]
    return distances, indices, top_passages_batch


key = "previous_text" if "relevant_passages" in data_dir else "gold_text"
print(f"[!] key: {key}")
def retrieve_top_passages_batch(entries):
    
    all_queries = entries[key]
    all_passages_batch = entries['oracle_documents_passages']
    all_passages_text_with_ids = [f"{passage[0]}\n{passage[1]}" for passages in all_passages_batch for passage in passages]
    all_passages_text = [passage[1] for passages in all_passages_batch for passage in passages]
    _, _, top_passages_batch = search_with_embeddings(all_queries, all_passages_text, all_passages_text_with_ids)
    return {f"top_{top_k}_passages": top_passages_batch}

create = True
    

if create or f"top_{top_k}_passages" not in current_dataset.column_names['train'] or f"top_{top_k}_passages" not in current_dataset.column_names['test']:
    print(f"[*] adding top_{top_k}_passages to {current_workshop_hf_name}")
    new_current_dataset = DatasetDict({split: current_dataset[split] for split in current_dataset.keys()})
    new_current_dataset = new_current_dataset.map(
        lambda batch: retrieve_top_passages_batch(batch),
        batched=True,
        batch_size=4
    )
    new_current_dataset.push_to_hub(f"ylkhayat/{current_workshop_hf_name}", data_dir=data_dir)
else:
    print(f"[!] top_{top_k}_passages already exists in {current_workshop_hf_name}")
print(json.dumps(new_current_dataset['train'][0][f"top_{top_k}_passages"][0], indent=4))