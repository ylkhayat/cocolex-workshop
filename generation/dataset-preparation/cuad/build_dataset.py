# %%
from datasets import load_dataset

dataset = load_dataset("theatticusproject/cuad-qa")

# %%
import json


print(json.dumps(dataset['test'][16], indent=4))

# %%
import spacy

from more_itertools import windowed
nlp = spacy.load("en_core_web_sm")

def process_record(record):
    document_id = "REFERENCE"
    document = record['context']
    answers = record['answers']
    gold_text = "Highlights:\n" + "\n".join([f"'{answer}'" for answer in answers])
    citations = [document_id, document]

    def chunk_document(document):
        doc = nlp(document)
        sentences = [sent.text for sent in doc.sents]
        return sentences
    sentences = chunk_document(document)
    
    oracle_documents_passages = []
    for sentence in windowed(sentences, 5, step=2):
        oracle_documents_passages.append([document_id, sentence])
    
    return {
        'docid': str(record['id']),
        'previous_text': record['question'],
        'gold_text': gold_text,
        'citations': citations,
        'oracle_documents_passages': oracle_documents_passages,
    }

dataset = dataset.map(process_record, num_proc=43)

# %%
dataset.push_to_hub('ylkhayat/CUAD-generation-workshop', data_dir='data')

# %%
# - complete documents
# text is extracted from context

# - align score (gen and the answer) correctness
#  ( gen and top k) faithfulness
 
#  - we need to break the context into chunks (paragraphs) - normal spacy -> to get sentences and then put in paragraph 
 
 
 
#  - instruct to mention no answer explicitly (challenge)
 
#  - challenge correctness (for no answer, manually state that there is no answer for better scores)
#  - challenge faithfulness (exclude no answers OR base don generations (if exists evaluate faithfulness, otherwise exclude the record))


