import warnings
# Ignore the specific scikit-learn deprecation warning — might have to update pixi.toml in the future
warnings.filterwarnings("ignore", message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")

import pandas as pd
import numpy as np
import os
from factor_analyzer import FactorAnalyzer
from sklearn.utils import resample
from factor_analyzer import FactorAnalyzer


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
            'Scale': name,
            'McDonalds_Omega': round(actual_omega, 3),
            '95% CI Lower': round(ci_lower, 3),
            '95% CI Upper': round(ci_upper, 3),
            'Item count': len(items),
            'Boot_Iterations': len(boot_scores) # shows how many samples converged
        })
    
    return pd.DataFrame(results)


def calculate_omega_hierarchical(df, items_dict, n_iterations=1000):
    """
    Calculates omega-hierarchical (ωh) using Schmid-Leiman orthogonalization.
    ωh = variance due to the general factor / total variance
    
    This requires a bifactor structure: each item loads on a general factor
    AND its specific group factor simultaneously.

    Interpretation:
    • ωh > 0.65 : General factor is strong — supports hierarchical model
    • 0.50 < ωh < 0.65 : Moderate — general factor exists but subscales carry significant unique variance
    • ωh < 0.50 : Weak general factor — the total score may not be meaningful
    """
    all_items = [item for sublist in items_dict.values() for item in sublist]
    n_group_factors = len(items_dict)
    data = df[all_items]
    results = []
    
    def _compute_omega_h(data):
        try:
            # Step 1: Extract higher-order factor solution (oblique, n = number of subscales)
            fa = FactorAnalyzer(n_factors=n_group_factors, rotation='oblimin')
            fa.fit(data)
            group_loadings = fa.loadings_        # item x group-factor loadings
            phi = fa.phi_                         # factor intercorrelation matrix

            # Step 2: Schmid-Leiman orthogonalization
            # Eigendecompose phi to get the general factor from the inter-factor correlations
            eigenvalues, eigenvectors = np.linalg.eigh(phi)
            # General factor = first principal component of phi
            g_loadings_on_factors = eigenvectors[:, -1] * np.sqrt(eigenvalues[-1])
            
            # General factor loadings on items = group_loadings @ g_loadings_on_factors
            g = group_loadings @ g_loadings_on_factors  # shape: (n_items,)
            
            # Step 3: Residualize group loadings
            # S = group_loadings - g * g_loadings_on_factors (remove g from group factors)
            s = group_loadings - np.outer(g, g_loadings_on_factors)
            
            # Step 4: Compute ωh
            # Numerator: variance due to general factor
            var_g = np.sum(g)**2
            # Denominator: total variance = var_g + group factor variance + uniqueness
            var_groups = np.sum([np.sum(s[:, j])**2 for j in range(s.shape[1])])
            communalities = fa.get_communalities()
            uniqueness = np.sum(1 - communalities)
            
            omega_h = var_g / (var_g + var_groups + uniqueness)
            return omega_h
        except:
            return np.nan

    # Point estimate
    omega_h = _compute_omega_h(data)
    
    # Bootstrap CI
    boot_scores = []
    for _ in range(n_iterations):
        sample = resample(data)
        score = _compute_omega_h(sample)
        if not np.isnan(score):
            boot_scores.append(score)
    
    ci_lower = np.percentile(boot_scores, 2.5)
    ci_upper = np.percentile(boot_scores, 97.5)
    
    results.append({
        'Scale': 'Well-Being (omega-hierarchical)',
        'Omega_h': round(omega_h, 3),
        '95% CI Lower': round(ci_lower, 3),
        '95% CI Upper': round(ci_upper, 3),
        'Item count': len(all_items),
        'Boot_Iterations': len(boot_scores)
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

        # 4. Calculations for C + A
        C_A_items = items_dict['Competence'] + items_dict['Autonomy']
        well_being_results_C_A = calculate_advanced_reliability(df, {'Well-Being (C, A)': C_A_items})

        # 5. Calculations for C + A + HRR
        C_A_HRR_items = items_dict['Competence'] + items_dict['Autonomy'] + items_dict['HR-Rel']
        well_being_results_C_A_HRR = calculate_advanced_reliability(df, {'Well-Being (C, A, HRR)': C_A_HRR_items})

        # 6. Omega-hierarchical for the general WB factor
        omega_h_result = calculate_omega_hierarchical(df, items_dict)
        print("\nOmega-Hierarchical Result:")

        # - Save results -
        output_dir = 'output/reliability'
        os.makedirs(output_dir, exist_ok=True)

        # Combine all results into one DataFrame and save as CSV
        combined_results = pd.concat([results_df, well_being_results, well_being_results_SDT, well_being_results_C_A, well_being_results_C_A_HRR], ignore_index=True)
        combined_results.to_csv(f"{output_dir}/reliability_report_omega_{file_name}", index=False)

        # Save omega-hierarchical result separately
        omega_h_result.to_csv(f"{output_dir}/reliability_report_omega_hierarchical_{file_name}", index=False)

        # Visual confirmation in console
        print("\nAll reliability reports with 95% CIs generated successfully.")
        print(combined_results)
        print(omega_h_result)

        # - Compute the ratio ωh / ω-total for the full scale -
        omega_total = well_being_results.loc[0, 'McDonalds_Omega']
        omega_h = omega_h_result.loc[0, 'Omega_h']
        ratio = omega_h / omega_total if omega_total > 0 else np.nan
        print(f"\nRatio of ωh ({omega_h}) to ω-total ({omega_total}) for the full scale: {ratio}")
    else:
        print(f"Error: File not found at {path}")