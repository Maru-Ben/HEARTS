import os
import glob
import pandas as pd
import random
import hashlib
import argparse

def shuffle_columns_in_file(input_file, output_file):
    """
    Reads a CSV file, shuffles its columns randomly, and writes the result.
    The random seed is derived from the file's name so that each file gets its own random order.
    """
    # Generate a seed from the file's basename using MD5
    file_seed = int(hashlib.md5(os.path.basename(input_file).encode()).hexdigest(), 16) % (10**8)
    random.seed(file_seed)
    
    # Read the CSV into a DataFrame
    df = pd.read_csv(input_file)
    
    # Get list of columns and shuffle them
    cols = list(df.columns)
    random.shuffle(cols)
    
    # Reorder DataFrame columns and write to CSV
    df = df[cols]
    df.to_csv(output_file, index=False)
    print(f"Processed: {input_file} -> {output_file}")

def process_parent_folder(parent_folder):
    """
    For a given parent folder (e.g. 'data/santos'), check for a 'datalake' folder.
    If it exists, create a 'datalake-p-col' folder alongside it and process all CSV files.
    """
    datalake_dir = os.path.join(parent_folder, "datalake")
    output_dir = os.path.join(parent_folder, "datalake-p-col")
    
    if not os.path.exists(datalake_dir):
        print(f"Skipping {parent_folder}: No 'datalake' folder found.")
        return  # Skip if no datalake folder
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process every CSV file in the datalake folder
    for csv_file in glob.glob(os.path.join(datalake_dir, "*.csv")):
        output_file = os.path.join(output_dir, os.path.basename(csv_file))
        shuffle_columns_in_file(csv_file, output_file)

def main():
    parser = argparse.ArgumentParser(
        description="Shuffle CSV file columns randomly in specified folders."
    )
    parser.add_argument(
        "folders",
        nargs="*",
        help=("Folder names inside the 'data' directory to process (e.g., 'tusLarge'). "
              "If omitted, all folders under 'data' will be processed.")
    )
    args = parser.parse_args()
    
    base_dir = "data"
    
    # Determine which folders to process
    if args.folders:
        folder_list = args.folders
    else:
        folder_list = [folder for folder in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, folder))]
    
    for folder in folder_list:
        parent_path = os.path.join(base_dir, folder)
        if os.path.isdir(parent_path):
            print(f"Processing folder: {parent_path}")
            process_parent_folder(parent_path)
        else:
            print(f"Warning: {parent_path} is not a directory")

if __name__ == "__main__":
    main()