import pandas as pd
import semopy

# Terminal command to run the script: pixi run python cfa_analysis.py

def perform_cfa(file_path):
    """
    Performs a Confirmatory Factor Analysis (CFA) to test the hypothesized
    4-factor structure of the questionnaire. Includes data cleaning for non-variance rows.
    """
    try:
        # --- 1. Load Data ---
        df = pd.read_csv(file_path)
        
        print("="*60)
        print("      CONFIRMATORY FACTOR ANALYSIS (CFA)")
        print("="*60)


        # --- 2. Data Cleaning: Remove Inattentive Responders (Straight-liners) ---
        print("\n--- 1. Data Cleaning ---\n")
        item_cols = df.columns.drop('Respondent_ID')
        
        # Calculate the standard deviation for each respondent's answers
        row_std = df[item_cols].std(axis=1)
        
        # Identify rows with zero variance (straight-liners)
        zero_variance_rows = row_std == 0
        num_removed = zero_variance_rows.sum()

        if num_removed > 0:
            print(f"Found and removed {num_removed} 'straight-liner' respondents (rows with zero variance).")
            # Keep only the rows with some variance
            df_cleaned = df[~zero_variance_rows].copy()
            print(f"Original sample size: {len(df)}, Cleaned sample size: {len(df_cleaned)}")
        else:
            print("No straight-liner respondents found. Proceeding with the full dataset.")
            df_cleaned = df.copy()

        # Use the cleaned data for the model
        df_model = df_cleaned.drop('Respondent_ID', axis=1)
        

        # --- 3. Define the Model Specification ---
        cfa_model_spec = """
            # Measurement Model
            C   =~ C1 + C2 + C3 + C4 + C5 + C6 + C7
            A   =~ A1 + A2 + A3 + A4 + A5 + A6 + A7
            HHR =~ HHR1 + HHR2 + HHR3 + HHR4
            HRR =~ HRR1 + HRR2 + HRR3 + HRR4

            # Structural Model (allowing latent factors to correlate)
            C ~~ A
            C ~~ HHR
            C ~~ HRR
            A ~~ HHR
            A ~~ HRR
            HHR ~~ HRR
        """
        
        print("\n--- 2. CFA Model Specification ---\n")
        print(cfa_model_spec)
        

        # --- 4. Build and Fit the Model ---
        print("\n--- 3. Fitting the Model to the Cleaned Data ---\n")
        model = semopy.Model(cfa_model_spec)
        
        # Fit the model to the CLEANED data
        results = model.fit(data=df_model, obj="MLW") # Using MLW objective is robust
        

        # --- 5. Get Goodness-of-Fit Statistics ---
        print("\n--- 4. Goodness-of-Fit Indices ---\n")
        stats = semopy.calc_stats(model)
        print(stats.T)

        print("\n--- Interpretation Guide for Fit Indices ---")
        print("  - CFI/TLI: > 0.95 is excellent, > 0.90 is acceptable.")
        print("  - RMSEA:   < 0.06 is excellent, < 0.08 is acceptable.")
        print("  - SRMR:    < 0.08 is a good fit.")
        

        # --- 6. Inspect Parameter Estimates ---
        print("\n--- 5. Model Parameter Estimates (Loadings & Correlations) ---\n")
        estimates = model.inspect()
        print(estimates)
        print("\nCheck that factor loadings (op: =~ ) are statistically significant (p < 0.05).")

        print("\n" + "="*60)
        print("                CFA Analysis Complete")
        print("="*60)


    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Main execution ---
if __name__ == "__main__":
    # Run the analysis
    file_name = 'converted-dataset-reversed.csv'
    perform_cfa(file_name)