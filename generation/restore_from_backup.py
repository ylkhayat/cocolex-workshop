import os
import shutil
from tqdm import tqdm

def copy_larger_files(
    src_dir_1="/srv/elkhyo/lexquo/generation/_backup/27_12",
    src_dir_2="/srv/elkhyo/lexquo/generation/_backup/28_12_corrupted",
    dest_dir="/srv/elkhyo/lexquo/generation/basement_updated"
):
    # This dictionary will map:
    #   rel_path -> (file_size, full_path_of_selected_file)
    file_map = {}

    # Counters for statistics
    replaced_with_larger = 0  # how many times an existing file in file_map got replaced by a larger one
    single_files_copied = 0   # how many files were newly added without conflict

    # Helper function to process a single directory
    # We merge its files into the file_map, picking larger files if there's a conflict
    def process_directory(src_dir, file_map, count_as_single=True):
        nonlocal replaced_with_larger, single_files_copied
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                full_path = os.path.join(root, f)
                # Calculate the relative path so we can compare conflicts
                rel_path = os.path.relpath(full_path, start=src_dir)
                size = os.path.getsize(full_path)

                if rel_path not in file_map:
                    # If this file doesn't exist in the map yet, add it
                    file_map[rel_path] = (size, full_path)
                    if count_as_single:
                        single_files_copied += 1
                else:
                    # Conflict scenario: same relative path file has already been recorded
                    existing_size, existing_path = file_map[rel_path]
                    if size > existing_size:
                        # Replace with the larger file
                        file_map[rel_path] = (size, full_path)
                        replaced_with_larger += 1

    process_directory(src_dir_1, file_map, count_as_single=True)
    process_directory(src_dir_2, file_map, count_as_single=True)

    # 3. Now copy the selected files into the destination directory
    for rel_path, (size, src_path) in tqdm(file_map.items(), desc="Copying files"):
        dst_path = os.path.join(dest_dir, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)

    print(f"[!] replaced with larger ones: {replaced_with_larger}")
    print(f"[!] single files added: {single_files_copied}")
    print(f"[!] total files in basement: {len(file_map)}")

if __name__ == "__main__":
    copy_larger_files()