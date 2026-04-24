import json
import pandas as pd

# Define input Parquet files
input_files = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet"
}

dfs = []
for split_name, fname in input_files.items():
    try:
        df = pd.read_parquet(fname)
        df["split"] = split_name
        dfs.append(df)
    except FileNotFoundError:
        print(f"Skipped missing file: {fname}")

# Concatenate all and save as one file
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_parquet("combined.parquet")
