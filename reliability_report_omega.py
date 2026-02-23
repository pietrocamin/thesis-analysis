import warnings
# Ignore the specific scikit-learn deprecation warning — might have to update pixi.toml in the future
warnings.filterwarnings("ignore", message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")

import pandas as pd
import numpy as np
import os
from factor_analyzer import FactorAnalyzer
from sklearn.utils import resample

# -- Helper function for the core calculation --
def _compute_omega(data):
    """
    Calculates a single Omega value for a given dataset.
    Omega = (sum of loadings)^2 / [(sum of loadings)^2 + sum(1 - communalities)]
    """
    try:
        # 1-factor analysis to get loadings for Omega
        fa = FactorAnalyzer(n_factors=1, rotation=None)
        fa.fit(data)

        # Loadings and Communalities
        loadings = fa.loadings_
        communalities = fa.get_communalities()
        
        # McDonald's Omega Calculation
        sum_loadings = np.sum(loadings)**2
        sum_uniqueness = np.sum(1 - communalities)
        omega = sum_loadings / (sum_loadings + sum_uniqueness)
        return omega
    except:
        return np.nan

# —- Calculate McDonald's Omega for each factor with Bootstrapping —-
def calculate_advanced_reliability(df, items_dict, n_iterations=1000):
    """
    Calculates McDonald's Omega for each factor in items_dict with bootstrapped confidence intervals.
    
    param df: dataframe containing the survey data
    param items_dict: dictionary with factor names as keys and list of item columns as values
    param n_iterations: number of bootstrap iterations for confidence intervals
    return: DataFrame with Omega values and confidence intervals
    """
    results = []
    
    for name, items in items_dict.items():
        print(f"Processing {name}...")
        data = df[items]
        
        # 1. Calculate the actual (Point Estimate) Omega
        actual_omega = _compute_omega(data)
        
        # 2. Bootstrapping for Confidence Intervals
        boot_scores = []
        for i in range(n_iterations):
            # Resample with replacement
            sample = resample(data)
            score = _compute_omega(sample)
            if not np.isnan(score):
                boot_scores.append(score)
        
        # 3. Calculate 95% CI (2.5th and 97.5th percentiles)
        ci_lower = np.percentile(boot_scores, 2.5)
        ci_upper = np.percentile(boot_scores, 97.5)
        
        results.append({
            'Factor': name,
            'McDonalds_Omega': round(actual_omega, 3),
            '95% CI Lower': round(ci_lower, 3),
            '95% CI Upper': round(ci_upper, 3),
            'Items': len(items),
            'Boot_Iterations': len(boot_scores) # shows how many samples converged
        })
    
    return pd.DataFrame(results)


# --- Main execution ---
if __name__ == "__main__":
    file_name = 'reversed_DATASET_problematic_items_clean.csv'
    path = f"DATASETS/{file_name}"
    
    if os.path.exists(path):
        df = pd.read_csv(path)

        items_dict = {
            'Competence': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
            'Autonomy': ['A1', 'A2', 'A3', 'A5', 'A6', 'A7'],
            'HH-Rel': ['HHR1', 'HHR2', 'HHR3', 'HHR4'],
            'HR-Rel': ['HRR1', 'HRR2', 'HRR3', 'HRR4']
        }

        # 1. Calculations for individual factors
        results_df = calculate_advanced_reliability(df, items_dict)

        # 2. Calculations for Global Well-being (All items)
        all_items = [item for sublist in items_dict.values() for item in sublist]
        well_being_results = calculate_advanced_reliability(df, {'Well-Being (All Items)': all_items})

        # 3. Calculations for SDT Well-being (C, A, HHR)
        SDT_items = items_dict['Competence'] + items_dict['Autonomy'] + items_dict['HH-Rel']
        well_being_results_SDT = calculate_advanced_reliability(df, {'Well-Being (C, A, HHR)': SDT_items})

        # - Save results -
        output_dir = 'output/reliability'
        os.makedirs(output_dir, exist_ok=True)

        results_df.to_csv(f"{output_dir}/reliability_report_omega_{file_name}", index=False)
        well_being_results.to_csv(f"{output_dir}/reliability_report_omega_WB_{file_name}", index=False)
        well_being_results_SDT.to_csv(f"{output_dir}/reliability_report_omega_WB_SDT_{file_name}", index=False)

        print("\nAll reliability reports with 95% CIs generated successfully.")
        print(results_df) # visual confirmation in console
    else:
        print(f"Error: File not found at {path}")