# %%
from datasets import load_dataset

dataset = load_dataset("umarbutler/open-australian-legal-qa")
documents_dataset = load_dataset("umarbutler/open-australian-legal-corpus")

# %%

# %%
from tqdm import tqdm

corpus_dict = {item['version_id']: item['text'] for item in tqdm(documents_dataset['corpus'])}

# %%
import json

print(json.dumps(dataset['train'][1], indent=2))

# %%
import spacy
import os

num_proc=os.cpu_count()

from more_itertools import windowed
nlp = spacy.load("en_core_web_sm")

def chunk_document(document):
    doc = nlp(document)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def process_record(record):
    gold_text = record['answer']
    source = record['source']
    document_id = source['version_id']
    source_document = corpus_dict[document_id]
    citations = [[document_id, source_document]]
    sentences = chunk_document(source_document)
    oracle_documents_passages = []
    for sentence in windowed(sentences, 5, fillvalue="", step=2):
        section = " ".join([s for s in sentence if (s is not None and len(s.strip()) > 0)])
        oracle_documents_passages.append([document_id, section])
    return {
        'previous_text': record['question'],
        'gold_text': gold_text,
        'citations': citations,
        'oracle_documents_passages': oracle_documents_passages,
    }

dataset = dataset.map(process_record, num_proc=num_proc)
dataset = dataset.map(lambda _, idx: {'docid': str(idx + 1)}, with_indices=True)
dataset = dataset.select_columns(['docid', 'previous_text', 'gold_text', 'citations', 'oracle_documents_passages'])

# %%
dataset.push_to_hub("ylkhayat/OAL_QA-generation-workshop", data_dir="data")


