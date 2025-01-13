import os
import pickle

def create_file_category_mapping(black_white_folder):
    mapping = {}
    for root, dirs, files in os.walk(black_white_folder):
        for file in files:
            file_basename = os.path.basename(file).lower().strip()
            relative_path = os.path.relpath(root, black_white_folder)
            if file_basename in mapping:
                mapping[file_basename].add(relative_path)
            else:
                mapping[file_basename] = set([relative_path])
    return mapping

def save_mapping(mapping, filename='file_category_mapping.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(mapping, f)

def main():
    black_white_folder = 'black_white'
    mapping = create_file_category_mapping(black_white_folder)
    save_mapping(mapping)
    print("Mappings files done.")

if __name__ == "__main__":
    main()
