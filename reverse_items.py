import pandas as pd
import io


def convert_reversed_items(file_path):
    # Read the dataset into a pandas DataFrame
    df = pd.read_csv(file_path)

    # The mapping for conversion
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
    output_filename = "converted_dataset_specific_columns.csv"


    print("Converted DataFrame:")
    print(df.to_string())

    # --- Save the converted DataFrame to a CSV file ---
    df.to_csv(output_filename, index=False)

    print(f"\nConverted dataset saved to '{output_filename}'")



# --- Main execution ---
if __name__ == "__main__":
    # Run the conversion
    file_name = 'DATASET.csv'
    convert_reversed_items(file_name)