import json
import hashlib
from collections import Counter
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

# Generate Hashes
def generate_hashes(data):
    return {
        "MD5": hashlib.md5(data.encode('UTF-8')).hexdigest(),
        "SHA1": hashlib.sha1(data.encode('UTF-8')).hexdigest(),
        "SHA256": hashlib.sha256(data.encode('UTF-8')).hexdigest(),
    }

# Calculate Entropy
def calculate_entropy(hash_value):
    counts = Counter(hash_value)
    length = len(hash_value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())

# Analyze Hash Characteristics
def analyze_hash(hash_value):
    return {
        "Length": len(hash_value),
        "Contains Numbers": any(char.isdigit() for char in hash_value),
        "Contains Letters": any(char.isalpha() for char in hash_value),
        "Contains Special Characters": any(not char.isalnum() for char in hash_value),
        "Entropy": calculate_entropy(hash_value),
    }

# Process a single input line
def process_line(line):
    dataset = []
    hash_data = generate_hashes(line)
    for algo, hash_value in hash_data.items():
        features = analyze_hash(hash_value)
        dataset.append({
            "Input Value": line,
            "Hash Algorithm": algo,
            "Hash Value": hash_value,
            **features,
        })
    return dataset

# Read a fixed number of lines from the file
def read_fixed_lines(filename, line_limit=100000):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        chunk = []
        for idx, line in enumerate(file):
            if idx >= line_limit:
                break
            chunk.append(line.strip())
        yield chunk  # Yield all lines up to the limit

# Save dataset to JSON
def save_to_json(data, output_filename):
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)

# Main function
def main(input_filename, output_filename, max_threads=8, line_limit=100000):
    dataset = []
    with ThreadPoolExecutor(max_threads) as executor:
        for chunk in tqdm(read_fixed_lines(input_filename, line_limit=line_limit)):
            results = executor.map(process_line, chunk)
            for result in results:
                dataset.extend(result)
    
    save_to_json(dataset, output_filename)
    print(f"Processed {line_limit} lines. Dataset saved to {output_filename}")

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))

    input_filename = os.path.join(data_dir, "../data/data.txt")  # Large input file
    output_filename = os.path.join(data_dir,"../data/hash_dataset.json")  # Output JSON file
    main(input_filename, output_filename)
