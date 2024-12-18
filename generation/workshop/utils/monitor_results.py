import os
import json

import ipdb
from tabulate import tabulate
from tqdm import tqdm

def find_generations_folder(start_dir):
    for root, dirs, files in os.walk(start_dir):
        if 'generations' in dirs:
            return os.path.join(root, 'generations')
    return None

def count_lines_in_jsonl_files(generations_folder):
    file_line_counts = {}
    for file_name in os.listdir(generations_folder):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(generations_folder, file_name)
            with open(file_path, 'r') as file:
                line_count = sum(1 for _ in file)
                file_line_counts[file_path] = line_count
    return file_line_counts

experiments_count = {}
invalid_experiments = {}
same_start_experiments = {}

start_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../basement'))
for root, dirs, files in tqdm(os.walk(start_dir)):
    for file_name in files:
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r') as file:
                key = file_path.split('basement/', 1)[-1].split('__', 1)[0]
                invalid_count = 0
                line_count = 0
                same_start_count = 0
                for line in file:
                    line_count += 1
                    data = json.loads(line)
                    if "clerc" in key and len(data['gen'].strip()) < 20:
                        invalid_count += 1
                    elif "echr_qa" in key and len(data['gen'].strip()) < 30:
                        invalid_count += 1
                    if data['gen'].strip()[:100] in data['meta']['previous_text']:
                        same_start_count += 1
                experiments_count[key] = line_count
                invalid_experiments[file_path.split('basement/', 1)[-1]] = invalid_count
                same_start_experiments[key] = same_start_count
                    
with open('output.txt', 'w') as output_file:
    sorted_experiments_count = sorted(experiments_count.items(), key=lambda item: item[1])
    output_file.write(tabulate(sorted_experiments_count, headers=['File', 'Line count'], tablefmt='fancy_grid'))
with open('output_invalid.txt', 'w') as output_file:
    sorted_invalid_experiments = sorted(
        [(key, count) for key, count in invalid_experiments.items() if count > 0],
        key=lambda item: item[1],
        reverse=True
    )
    output_file.write(tabulate(sorted_invalid_experiments, headers=['File', 'Invalid count'], tablefmt='fancy_grid'))
with open('output_same_start.txt', 'w') as output_file:
    sorted_same_start_experiments = sorted(
        [(key, count) for key, count in same_start_experiments.items() if count > 0],
        key=lambda item: item[1],
        reverse=True
    )
    output_file.write(tabulate(sorted_same_start_experiments, headers=['File', 'Same start count'], tablefmt='fancy_grid'))