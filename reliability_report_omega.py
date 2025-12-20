from factor_analyzer import FactorAnalyzer
import pandas as pd
import numpy as np
import os

# —- Calculate McDonald's Omega for each factor —-
def calculate_advanced_reliability(df, items_dict):
    results = []
    
    for name, items in items_dict.items():
        # Subset data
        data = df[items]
        
        # 1-factor analysis to get loadings for Omega
        fa = FactorAnalyzer(n_factors=1, rotation=None)
        fa.fit(data)
        
        # Loadings and Communalities
        loadings = fa.loadings_
        communalities = fa.get_communalities()
        
        # - McDonald's Omega Calculation (Total) -
        # Formula: (sum of loadings)^2 / [(sum of loadings)^2 + sum(1 - communalities)]
        sum_loadings = np.sum(loadings)**2
        sum_uniqueness = np.sum(1 - communalities)
        omega_total = sum_loadings / (sum_loadings + sum_uniqueness)
        
        results.append({
            'Factor': name,
            'McDonalds_Omega': round(omega_total, 3),
            'Items': len(items)
        })
    
    return pd.DataFrame(results)


# --- Main execution ---
if __name__ == "__main__":
    file_name = 'reversed_DATASET_problematic_items_clean.csv'
    df = pd.read_csv(f"DATASETS/{file_name}")

    items_dict = {
        'Competence': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
        'Autonomy': ['A1', 'A2', 'A3', 'A5', 'A6'],
        'HH-Rel': ['HHR1', 'HHR2', 'HHR3', 'HHR4'],
        'HR-Rel': ['HRR1', 'HRR2', 'HRR3', 'HRR4']
    }

    # Calculations for each factor
    results_df = calculate_advanced_reliability(df, items_dict)

    # Calculations for higher factor (well-being) with all items combined
    all_items = [item for sublist in items_dict.values() for item in sublist]
    well_being_results = calculate_advanced_reliability(df, {'Well-Being (All Items)': all_items})

    # Calculations for higher factor (well-being) with SDT items (C, A, HHR) combined only
    SDT_dict = {
        'Competence': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
        'Autonomy': ['A1', 'A2', 'A3', 'A5', 'A6'],
        'HH-Rel': ['HHR1', 'HHR2', 'HHR3', 'HHR4']
    }
    all_items_SDT = [item for sublist in SDT_dict.values() for item in sublist]
    well_being_results_SDT = calculate_advanced_reliability(df, {'Well-Being (C, A, HHR)': all_items_SDT})

    # Save results in CSV files
    os.makedirs('output/reliability', exist_ok=True)

    results_df.to_csv(f"output/reliability/reliability_report_omega_{file_name.replace('.csv', '')}.csv", index=False)
    print(f"Reliability report saved to: output/reliability/reliability_report_omega_{file_name.replace('.csv', '')}.csv")

    well_being_results.to_csv(f"output/reliability/reliability_report_omega_WB_{file_name.replace('.csv', '')}.csv", index=False)
    print(f"Well-being reliability report saved to: output/reliability/reliability_report_omega_WB_{file_name.replace('.csv', '')}.csv")

    well_being_results_SDT.to_csv(f"output/reliability/reliability_report_omega_WB_SDT_factors_{file_name.replace('.csv', '')}.csv", index=False)
    print(f"Well-being (SDT) reliability report saved to: output/reliability/reliability_report_omega_WB_SDT_factors_{file_name.replace('.csv', '')}.csv")

    print("Reliability reports generated successfully.")