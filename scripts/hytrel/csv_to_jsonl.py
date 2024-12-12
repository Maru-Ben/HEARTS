import pandas as pd
import json
import os
import uuid
from pathlib import Path
import spacy
from tqdm import tqdm
import sys

def infer_column_type(sample_value):
    try:
        float(sample_value)
        return "real"
    except:
        return "text"

def process_value(value, nlp):
    tokens = [token.text for token in nlp(str(value))]
    ner_tags = [token.ent_type_ or "" for token in nlp(str(value))]
    return {
        "value": str(value),
        "tokens": tokens,
        "ner_tags": ner_tags
    }

def detect_separator(csv_file):
    """Detect whether file uses comma or semicolon as separator"""
    with open(csv_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if ';' in first_line:
            return ';'
        return ','

def csv_to_jsonl(csv_dir, output_dir, max_entries_per_file=100):
    nlp = spacy.load('en_core_web_sm')
    entries = []
    file_count = 0
    
    csv_files = list(Path(csv_dir).glob('*.csv'))
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            # Detect the separator
            separator = detect_separator(csv_file)
            
            # Try reading with detected separator
            if separator == ';':
                df = pd.read_csv(csv_file, sep=separator, index_col=None, skipinitialspace=True)
            else:
                df = pd.read_csv(csv_file, sep=separator, skipinitialspace=True)
            
            # Clean column names by stripping whitespace
            df.columns = df.columns.str.strip()
            
            # Process headers
            headers = []
            for col in df.columns:
                sample_val = df[col].iloc[0] if len(df) > 0 else ""
                headers.append({
                    "name": col,
                    "name_tokens": None,
                    "type": "text",  # default assumption
                    "sample_value": process_value(sample_val, nlp),
                    "sample_value_tokens": None,
                    "is_primary_key": False,
                    "foreign_key": None
                })
            
            # Convert data to list format
            data = df.values.tolist() if len(df) > 0 else []
            
            table_entry = {
                "id": str(uuid.uuid4()),
                "table": {
                    "caption": csv_file.stem,
                    "header": headers,
                    "data": data
                },
                "context_before": [],
                "context_after": []
            }
            
            entries.append(table_entry)
            
            # Write out in chunks
            if len(entries) >= max_entries_per_file:
                output_file = f"{output_dir}/chunk_{file_count}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for entry in entries:
                        json.dump(entry, f, ensure_ascii=False)
                        f.write('\n')
                entries = []
                file_count += 1
                
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            
    if entries:
        output_file = f"{output_dir}/chunk_{file_count}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python csv_to_jsonl.py <benchmark_name>")
        sys.exit(1)
        
    benchmark = sys.argv[1]

    # Check if it's a variant like *-p-col
    if benchmark.endswith("-p-col"):
        base_bench = benchmark.replace("-p-col","")
        csv_dir = f'./data/{base_bench}/datalake-p-col/'
        output_dir = f'./data/{base_bench}/chunks-p-col'
    else:
        csv_dir = f'./data/{benchmark}/datalake/'
        output_dir = f'./data/{benchmark}/chunks'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Processing {benchmark} benchmark...")
    print(f"Input directory: {csv_dir}")
    print(f"Output directory: {output_dir}")
    
    csv_to_jsonl(csv_dir, output_dir)
