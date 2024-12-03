import copy
import ipdb
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset


class TextSection:
    def __init__(self, name, text, priority, tokenizer):
        self.name = name
        self.text = text
        self.priority = priority
        self.tokenizer = tokenizer
        self.truncated = 0
        self.truncated_before = False
        self.tokenized_text = self.tokenizer(self.text, truncation=False)["input_ids"]
        self.initial_token_count = len(self.tokenized_text)

    def token_count(self):
        return len(self.tokenized_text)

    def truncate(self, num_tokens):
        if self.truncated_before:
            raise ValueError("[!] section already truncated!")
        if num_tokens >= len(self.tokenized_text):
            raise ValueError("[!] this should not happen!")
        self.truncated_before = True
        self.tokenizer.truncation_side = 'left'
        input_ids = self.tokenizer.encode(self.text, truncation=False, add_special_tokens=False)
        input_ids = input_ids[num_tokens:]
        self.tokenized_text = input_ids
        
        truncated_text = self.tokenizer.decode(self.tokenized_text, skip_special_tokens=True)
        if len(truncated_text) < len(self.text):
            self.truncated = len(self.tokenized_text) - len(truncated_text)
        if truncated_text != self.text:
            truncated_text = " ..." + truncated_text
        self.text = truncated_text
        return self.truncated
        
    def __repr__(self):
        truncated_percentage = (self.truncated / len(self.text)) * 100 if len(self.text) > 0 else 0
        return f"<Section {self.name}:{self.priority} ({'Truncated' if self.truncated_before else 'Complete'} — {self.truncated}, {truncated_percentage:.2f}% truncated)>"


class ModelInputPreprocessor:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def build_context_prompt(self,
                             prompt,
                             contexts,
                             method,
                             max_tokens,
                             use_instructions):
        assert isinstance(contexts, list) and len(contexts) > 0, "Contexts must be a non-empty list of strings."
        retrieved_ids = [doc.split('\n')[0] for doc in contexts]
        def _get_copy(section):
            return copy.deepcopy(section)
        prefix_linker_section = TextSection("prefix_linker", "\n\n", priority=0, tokenizer=self.tokenizer)
        context_prefix = (
            "Below are reference cases provided for factual accuracy. When generating content, you must "
            "reference and cross-check the relevant details with the provided reference texts by their "
            "reference IDs. (e.g., " + ', '.join(retrieved_ids) + "). Your output must align with these references."
        )
        context_prefix_section = TextSection("context_prefix", context_prefix, priority=0, tokenizer=self.tokenizer)
        context_sections = []
        for i, context in enumerate(contexts):
            context_sections.append(TextSection(f"ref_text_{i}", context, priority=1, tokenizer=self.tokenizer))
            if i < len(contexts) - 1:
                context_sections.append(TextSection("prefix_linker", "\n\n", priority=0, tokenizer=self.tokenizer))
                
        prompt_prefix = (
            'Continue to write the following case in the style of my writeup. Your answer should range '
            'from 100 to 400 words. Make your answer concise, and avoid redundant languages and assumptions. '
            'Below is what I have written so far:'
        )
        
        prompt_intro_section = TextSection("prompt_intro", prompt_prefix, priority=0, tokenizer=self.tokenizer)
        prompt_section = TextSection("prompt", prompt, priority=2, tokenizer=self.tokenizer)
        building_sections = []
        if method == "rag" or method == "cad" or "context" in method:
            building_sections.append([context_prefix_section, _get_copy(prefix_linker_section)])
            building_sections.append(context_sections)
        building_sections.append([prompt_intro_section, _get_copy(prefix_linker_section), prompt_section])
        
        all_sections = [section for group in building_sections for section in group]
        all_sections_token_count = [section.token_count() for section in all_sections]
        current_length = sum(all_sections_token_count)
        truncation = False
        if current_length > max_tokens:
            truncation = True
            extra_length = current_length - max_tokens
            minimum_tokens_per_section = 10
            weights = [(max(section.token_count() - minimum_tokens_per_section, 0) * section.priority / current_length)  if section.priority != 0 else 0 for section in all_sections]
            tokens_to_truncate = [int(extra_length * ratio) for ratio in weights]
            total_truncated = sum(tokens_to_truncate)
            remaining_truncations = extra_length - total_truncated
            if remaining_truncations > 0:
                weights = [(max(section.token_count() - minimum_tokens_per_section, 0) / current_length)  if section.priority == 0 else 0 for section in all_sections]
                tokens_to_truncate_last = [int(remaining_truncations * ratio) for ratio in weights]
                tokens_to_truncate = [truncate_sign_1 + truncate_sign_2 for truncate_sign_1, truncate_sign_2 in zip(tokens_to_truncate, tokens_to_truncate_last)]
        
            for section, truncate in zip(all_sections, tokens_to_truncate):
                if truncate > 0:
                    section.truncate(truncate)

        context_prefix = context_prefix_section.text
        ref_text = ''.join([section.text for section in context_sections])
        formatted_prompt = prompt_intro_section.text + prefix_linker_section.text + prompt_section.text
        
        if use_instructions and 'apply_chat_template' in dir(self.tokenizer):
            formatted_prompt = self._apply_chat_template(formatted_prompt)
            
        return context_prefix, ref_text, formatted_prompt, truncation

    def _apply_chat_template(self, prompt):
        prompt_chat_parts = [
            {"role": "system", "content": "You are a helpful legal professional."},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(prompt_chat_parts, tokenize=False)
    
    def preprocess_record(self, record, top_k, method, max_tokens, use_instructions):
        prev_text = record['previous_text']
        gold_text = record['gold_text']
        oracle_documents = record['citations']
        retrieved_docs = record['top_10_passages'][:top_k]

        context_prefix, contexts, prompt, is_truncated = self.build_context_prompt(prompt=prev_text,
                                                                     contexts=retrieved_docs,
                                                                     method=method,
                                                                     use_instructions=use_instructions,
                                                                     max_tokens=max_tokens)
        assert isinstance(is_truncated, bool), "is_truncated must be a boolean."
        return {
            "prompt": prompt,
            "context": contexts,
            "context_prefix": context_prefix,
            "is_truncated": is_truncated,
            "meta": {
                "gold_text": gold_text,
                "previous_text": prev_text,
                "oracle_documents": oracle_documents,
                "top_k_passages": retrieved_docs
            }
        }

    def process_dataset(self, config):
        required_keys = ['method', 'dataset', 'dataset_percentage', 'setup', 'split', 'top_k_passages', 'max_tokens']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"[!] Missing required config key: {key}")
            
        self.max_tokens = config['max_tokens']
        self.method = config['method']
        self.dataset = config['dataset']
        self.dataset_percentage = config['dataset_percentage']
        self.setup = config['setup']
        self.split = config['split']
        self.top_k_passages = config['top_k_passages']
        self.use_instructions = config['use_instructions'] or False
        
        dataset_repo_name = "CLERC-generation-workshop"
        if self.dataset == "echr":
            dataset_repo_name = "ECHR-generation-workshop"
        dataset_repo_name = f"ylkhayat/{dataset_repo_name}"
        current_dataset = load_dataset(dataset_repo_name, data_dir=self.setup, split=self.split)
        length_of_dataset = int(len(current_dataset) * self.dataset_percentage)

        print(f"[!] dataset: {dataset_repo_name}")
        print(f"[!] num of records: {length_of_dataset}")

        processed_dataset = current_dataset.select(range(length_of_dataset))
        processed_dataset = processed_dataset.map(
            lambda record: self.preprocess_record(record,
                                                  top_k=self.top_k_passages,
                                                  method=self.method,
                                                  max_tokens=self.max_tokens,
                                                  use_instructions=self.use_instructions),
            batched=False,
            num_proc=10
        )
        processed_dataset = processed_dataset.flatten()
        assert "meta.oracle_documents" in processed_dataset.column_names
        try:
            processed_dataset = processed_dataset.rename_column("appno", "docid")
        except:
            pass
        return processed_dataset
    
    

    