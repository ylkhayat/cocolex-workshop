import os
import json

def find_generations_folder(start_dir):
    for root, dirs, files in os.walk(start_dir):
        if 'generations' in dirs:
            return os.path.join(root, 'generations')
    return None

def count_lines_in_jsonl_files(generations_folder):
    total_lines = 0
    for file_name in os.listdir(generations_folder):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(generations_folder, file_name)
            with open(file_path, 'r') as file:
                total_lines += sum(1 for _ in file)
    return total_lines

start_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../basement'))
generations_folder = find_generations_folder(start_dir)

if generations_folder:
    total_lines = count_lines_in_jsonl_files(generations_folder)
    print(f"Total number of lines in JSONL files: {total_lines}")
else:
    print("Generations folder not found.")