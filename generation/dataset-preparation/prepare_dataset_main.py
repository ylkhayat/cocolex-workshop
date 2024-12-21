
from datasets import load_dataset, DatasetDict
from more_itertools import windowed
from sentence_transformers import SentenceTransformer
import bm25s
import faiss
import json
import nltk
import numpy as np
import os
import spacy
import Stemmer 
import torch


top_k = 50

default_create = True
chunk_size = 300
chunk_overlap = 0.3
lang = "en"
stemmer = Stemmer.Stemmer("english" if lang == "en" else "french")
print(f"[!] top_k: {top_k}")
print(f"[!] chunk_size: {chunk_size}")
print(f"[!] chunk_overlap: {chunk_overlap}")

num_proc = os.cpu_count() - 3

dataset = "obli_qa"

key_field = "docid"
if dataset == "clerc":
    original_dataset = load_dataset("jhu-clsp/CLERC", data_files={"train": f"generation/train.jsonl",  "test": f"generation/test.jsonl"})
    workshop_hf_name = f"CLERC-generation-workshop"
elif dataset == "echr":
    workshop_hf_name = f"ECHR-generation-workshop"
elif dataset == "echr_qa":
    workshop_hf_name = f"ECHR_QA-generation-workshop"
elif dataset == "obli_qa":
    workshop_hf_name = f"OBLI_QA-generation-workshop"
elif dataset == "cuad":
    workshop_hf_name = f"CUAD-generation-workshop"
else:
    raise ValueError("Invalid dataset")
current_chosen_dataset = load_dataset(f"ylkhayat/{workshop_hf_name}", data_dir="data")

nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")

def chunk_document_into_passages(document_id: str, text: str, max_len: int = 300, overlap: float = 0.0):
    if max_len <= 0:
        raise ValueError("max_len must be a positive integer")
    if not (0 <= overlap < 1):
        raise ValueError("overlap must be between 0 (inclusive) and 1 (exclusive)")
    chunks = []
    words = text.split(" ")
    # words = [word for word in words if len(word) > 0]
    chunked_words = windowed(words, max_len, fillvalue="", step=int(max_len * (1 - overlap)))
    for chunk in chunked_words:
        chunks.append([document_id, " ".join(chunk).strip()])
    return chunks


# def chunk_document_into_passages(document_id: str, text: str, max_len: int = 300, overlap: float = 0.0):
#     sentences = nltk.sent_tokenize(text)
#     merged_sentences = []
#     current_sentence = ""
#     for sentence in sentences:
#         if len(current_sentence) + len(sentence) < max_len:
#             current_sentence += " " + sentence
#         else:
#             if current_sentence:
#                 merged_sentences.append([document_id, current_sentence.strip()])
#             current_sentence = sentence
#     if current_sentence:
#         merged_sentences.append([document_id, current_sentence.strip()])
#     return merged_sentences


def chunk_citations(record, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
    chunks = []
    for citation_id, citation in record['citations']:
        chunks.extend(chunk_document_into_passages(citation_id, citation, max_len=chunk_size, overlap=chunk_overlap))
    return chunks

create = default_create
if create or 'oracle_documents_passages' not in current_chosen_dataset.column_names['train'] or 'oracle_documents_passages' not in current_chosen_dataset.column_names['test']:
    print(f"[*] adding oracle_documents_passages to {workshop_hf_name}")
    # Use parallel processing with num_proc if dataset is large. Increase num_proc as needed.
    current_chosen_dataset = current_chosen_dataset.map(
        lambda record: {'oracle_documents_passages': chunk_citations(record)},
        num_proc=num_proc
    )
    columns_to_select = [key_field, 'previous_text', 'gold_text', 'citations', 'oracle_documents_passages']
    # columns_to_select += [col for col in current_chosen_dataset['test'].column_names if col.startswith('top_')]
    current_chosen_dataset = current_chosen_dataset.select_columns(columns_to_select)
    current_chosen_dataset.push_to_hub(workshop_hf_name, data_dir="data")
else:
    print(f"[!] oracle_documents_passages already exists in {workshop_hf_name}")



# data_dir = "bm25_oracle_passages_oracle_documents"

# from datasets import DatasetDict


# create = False
# def retrieve_top_passages(entry):
#     query = entry['gold_text']
#     all_passages = entry['oracle_documents_passages']
#     all_passages_text = [f"{passage_arr[0]}\n{passage_arr[1]}" for passage_arr in all_passages]
#     corpus_tokens = bm25s.tokenize(all_passages_text, stopwords=lang, stemmer=stemmer)
#     retriever = bm25s.BM25()
#     retriever.index(corpus_tokens)
#     query_tokens = bm25s.tokenize(query, stemmer=stemmer)
#     local_top_k = top_k if top_k <= len(all_passages_text) else len(all_passages_text)
#     results, _ = retriever.retrieve(query_tokens, corpus=all_passages_text, k=local_top_k)
#     results = results.squeeze(0)
#     return {f"top_{top_k}_passages": results}  
# try:
#     if not create:
#         new_dataset = load_dataset(f"ylkhayat/{workshop_hf_name}", data_dir=data_dir)
# except:
#     print(f"[!] {workshop_hf_name} not found in {data_dir}")
#     create = True

# if create or f"top_{top_k}_passages" not in current_chosen_dataset.column_names['test']:
#     print(f"[*] adding top_{top_k}_passages to {workshop_hf_name}")
#     new_dataset = DatasetDict({split: current_chosen_dataset[split] for split in current_chosen_dataset.keys()})
#     new_dataset = new_dataset.map(retrieve_top_passages, num_proc=num_proc)
#     new_dataset.push_to_hub(f"ylkhayat/{workshop_hf_name}", data_dir=data_dir)
# else:
#     print(f"[!] top_{top_k}_passages already exists in {workshop_hf_name}")
# print(json.dumps(new_dataset['test'][0][f"top_{top_k}_passages"][0], indent=4)) 



data_dir = "bm25_relevant_passages_oracle_documents"

create = default_create
def retrieve_top_passages(entry):
    query = entry['previous_text']
    all_passages = entry['oracle_documents_passages']
    all_passages_original = [f"{passage_arr[0]}\n{passage_arr[1]}" for passage_arr in all_passages]
    all_passages_text = [passage_arr[1] for passage_arr in all_passages]
    corpus_tokens = bm25s.tokenize(all_passages_text, stopwords=lang, stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    query_tokens = bm25s.tokenize(query, stopwords=lang, stemmer=stemmer)
    
    local_top_k = top_k if top_k <= len(all_passages_text) else len(all_passages_text)

    results, _ = retriever.retrieve(query_tokens, k=local_top_k)
    results = results.squeeze(0)
    results = [all_passages_original[result] for result in results]
    return {f"top_k_passages": results}  

try:
    if not create:
        new_dataset = load_dataset(f"ylkhayat/{workshop_hf_name}", data_dir=data_dir)
except:
    print(f"[!] {workshop_hf_name} not found in {data_dir}")
    create = True
    
if create or f"top_{top_k}_passages" not in current_chosen_dataset.column_names['train'] or f"top_{top_k}_passages" not in current_chosen_dataset.column_names['test']:
    print(f"[*] adding top_{top_k}_passages to {workshop_hf_name}")
    new_dataset = DatasetDict({split: current_chosen_dataset[split] for split in current_chosen_dataset.keys()})
    new_dataset = new_dataset.map(retrieve_top_passages, num_proc=num_proc)
    new_dataset.push_to_hub(f"ylkhayat/{workshop_hf_name}", data_dir=data_dir)
else:
    print(f"[!] top_{top_k}_passages already exists in {workshop_hf_name}")
print(json.dumps(new_dataset['test'][0][f"top_k_passages"][0], indent=4)) 





encoder_name = 'sentence-transformers/all-MiniLM-L6-v2'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(encoder_name, device=device)

clean_encoder_name = encoder_name.replace("/", "_")
data_dir = "dense_relevant_passages_oracle_documents"
data_dir = f"{data_dir}/{clean_encoder_name}"

print(f"[!] data_dir: {data_dir}")

num_proc = os.cpu_count()
# def normalize_embeddings(embeddings):
#     return embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True)

# def embed_texts(texts):
#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=model.config.max_position_embeddings).to(device)
#     attention_mask = inputs['attention_mask']
#     with torch.no_grad():
#         token_embeddings = model(**inputs).last_hidden_state
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
#         sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
#         embeddings = sum_embeddings / sum_mask
#     embeddings = normalize_embeddings(embeddings).cpu().numpy()
#     return embeddings


def search_with_embeddings(all_queries, all_passages_text, all_passages_text_with_ids):
    # query_embeddings = embed_texts(all_queries)
    query_embeddings = model.encode(all_queries)
    # passage_embeddings = embed_texts(all_passages_text)
    passage_embeddings = model.encode(all_passages_text)
    dim = passage_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(passage_embeddings)
    local_top_k = top_k if top_k <= len(all_passages_text) else len(all_passages_text)
    distances, indices = index.search(np.expand_dims(query_embeddings, axis=0), local_top_k)
    top_passages_batch = [all_passages_text_with_ids[index] for index in indices[0]]
    return distances, indices, top_passages_batch


key = "previous_text" if "relevant_passages" in data_dir else "gold_text"
print(f"[!] key: {key}")
def retrieve_top_passages_batch(record):
    
    all_queries = record[key]
    all_passages = record['oracle_documents_passages']
    all_passages_text_with_ids = [f"{passage[0]}\n{passage[1]}" for passage in all_passages]
    all_passages_text = [passage[1] for passage in all_passages]
    _, _, top_passages_batch = search_with_embeddings(all_queries, all_passages_text, all_passages_text_with_ids)
    return {f"top_k_passages": top_passages_batch}

create = default_create
if create or f"top_{top_k}_passages" not in current_chosen_dataset.column_names['train'] or f"top_{top_k}_passages" not in current_chosen_dataset.column_names['test']:
    print(f"[*] adding top_{top_k}_passages to {workshop_hf_name}")
    new_current_chosen_dataset = DatasetDict({split: current_chosen_dataset[split] for split in current_chosen_dataset.keys()})
    new_current_chosen_dataset = new_current_chosen_dataset.map(retrieve_top_passages_batch,
        # batched=True,
        # batch_size=128
    )
    new_current_chosen_dataset.push_to_hub(f"ylkhayat/{workshop_hf_name}", data_dir=data_dir)
else:
    print(f"[!] top_{top_k}_passages already exists in {workshop_hf_name}")
print(json.dumps(new_current_chosen_dataset['train'][0][f"top_k_passages"][0], indent=4))