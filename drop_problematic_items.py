import os
import pandas as pd

# — Load dataset —
file_name = 'reversed_DATASET'
df = pd.read_csv(f"DATASETS/{file_name}.csv")


# — Remove problematic items —
# Specify items to remove — modify list as needed
items_to_remove = ['A4', 'A7']

# Remove items
df_cleaned = df.drop(columns=items_to_remove)


# — Save new dataset —
# Create output directory if it doesn't exist
os.makedirs('DATASETS', exist_ok=True)

# Save results to CSV
df_cleaned.to_csv(f"DATASETS/{file_name}_problematic_items_clean.csv", index=False)


# — Print summary —
print(f"Removed {len(items_to_remove)} items: {', '.join(items_to_remove)}")
print(f"Original: {len(df.columns)} columns")
print(f"Cleaned: {len(df_cleaned.columns)} columns")
print(f"Saved to: {file_name}_problematic_items_clean.csv")