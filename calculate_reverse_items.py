import pandas as pd
import io
import os


def convert_reverse_scored_items(file_path, file_name="specific_columns"):
    # Read the dataset into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Mapping for conversion
    conversion_map = {
        1: 5,
        2: 4,
        4: 2,
        5: 1
    }

    # --- The columns containing the inverted items ---
    columns_to_invert = ['C3', 'C4', 'C6', 'A1', 'A2', 'A3', 'A4', 'HHR3', 'HHR4']
    # -------------------------------------------------

    # Apply the conversion only to the specified columns
    # Using .loc for label-based indexing to select specific columns by name
    df[columns_to_invert] = df[columns_to_invert].applymap(lambda x: conversion_map.get(x, x))
    output_filename = f"DATASETS/converted_dataset_{file_name}.csv"


    print("Converted DataFrame:")
    print(df.to_string())

    # -- Save the converted DataFrame to a CSV file --
    # Create output directory if it doesn't exist
    os.makedirs('DATASETS', exist_ok=True)

    # Save results to CSV
    df.to_csv(output_filename, index=False)

    print(f"\nConverted dataset saved to 'DATASETS/{output_filename}'")



# --- Main execution ---
if __name__ == "__main__":
    # Run the conversion
    # file_name = 'DATASETS/DATASET.csv'
    # convert_reverse_scored_items(file_name)

    file_name = 'DATASETS/DATASET-cement-clean.csv'
    convert_reverse_scored_items(file_name, "cement_clean")