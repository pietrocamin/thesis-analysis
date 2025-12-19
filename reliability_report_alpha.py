import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import os


# — Function to calculate Cronbach's Alpha for each factor —
def get_reliability_report(df, factors):
    report = []

    for factor_name, items in factors.items():
        # Calculate Alpha
        alpha_val, ci = pg.cronbach_alpha(data=df[items]) # returns (alpha, (ci_lower, ci_upper))
        
        # Extract confidence intervals
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])

        report.append({
            'Factor': factor_name,
            'Alpha': round(float(alpha_val), 3),
            '95% CI': f"[{ci_lower:.3f}, {ci_upper:.3f}]",
            'N_Items': len(items)
        })

    return pd.DataFrame(report)


# — Function to plot item-total correlations for a given factor —
def plot_item_total_correlation(df, items, factor_name):
    # Calculate correlation of each item with the sum of all items in that factor
    factor_total = df[items].sum(axis=1)
    correlations = df[items].corrwith(factor_total)
    
    plt.figure(figsize=(len(items), 4))
    ax = sns.barplot(x=correlations.index, y=correlations.values,
                     hue=list(correlations.index), palette='viridis', dodge=False)
    ax.set_xlabel('')
    
    plt.axhline(0.3, color='red', linestyle='--', label='Threshold (0.3)')
    plt.title(f"Item-Total Correlations: {factor_name}")
    plt.ylabel("Correlation Coefficient")

    plt.savefig(f"images/reliability-report/{factor_name}-alpha.png", dpi=300, bbox_inches='tight', pad_inches=0.05)


# — Function to plot grouped heatmap of inter-item correlations —
def plot_grouped_heatmap(df, all_items):
    plt.figure(figsize=(12, 10))
    corr_matrix = df[all_items].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, square=True)
    plt.title("Inter-Item Correlation Matrix (Ordered by Factor)")
    
    plt.savefig("images/reliability-report/grouped-heatmap-alpha.png", dpi=300, bbox_inches='tight', pad_inches=0.05)


# --- Main execution ---
if __name__ == "__main__":
    file_name = "reversed_DATASET_problematic_items_clean.csv"
    df = pd.read_csv(f"DATASETS/{file_name}")

    # 1. Define item lists
    items_c = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    items_a = ['A1', 'A2', 'A3', 'A5', 'A6']
    items_hhr = ['HHR1', 'HHR2', 'HHR3', 'HHR4']
    items_hrr = ['HRR1', 'HRR2', 'HRR3', 'HRR4']

    # 2. Define the factor dictionary (Subfactors + 2 Global Options)
    factor_definitions = {
        'Competence': items_c,
        'Autonomy': items_a,
        'HH-Relatedness': items_hhr,
        'HR-Relatedness': items_hrr,
        # Higher Factor Option 1: Classical SDT items only
        'Global (C+A+HHR)': items_c + items_a + items_hhr,
        # Higher Factor Option 2: Full model including Robot Relatedness
        'Global (4 factors)': items_c + items_a + items_hhr + items_hrr
    }

    results_df = get_reliability_report(df, factor_definitions)
    print(results_df)

    # Save results to CSV
    os.makedirs('output', exist_ok=True)
    results_df.to_csv(f"output/reliability_report_alpha_{file_name.replace('.csv', '')}.csv", index=False)
    print(f"Reliability report saved to: output/reliability_report_alpha_{file_name.replace('.csv', '')}.csv")

    # 3. Generate and save visualizations
    os.makedirs('images/reliability-report', exist_ok=True)

    plot_item_total_correlation(df, items_c, "Competence")
    plot_item_total_correlation(df, items_a, "Autonomy")
    plot_item_total_correlation(df, items_hhr, "HH-Relatedness")
    plot_item_total_correlation(df, items_hrr, "HR-Relatedness")
    plot_item_total_correlation(df, items_c + items_a + items_hhr, "Global (C+A+HHR)")
    plot_item_total_correlation(df, items_c + items_a + items_hhr + items_hrr, "Global (4 factors)")
    
    all_ordered_items = items_c + items_a + items_hhr + items_hrr
    plot_grouped_heatmap(df, all_ordered_items)

    print("Reliability visualizations saved in: images/reliability-report/")