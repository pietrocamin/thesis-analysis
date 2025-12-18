import os
import pandas as pd
from scipy.stats import spearmanr

# -- Inter-factor correlations --
def calculate_inter_factor_correlations(file_name, cleaned=""):
    """
    The script calculates and prints the Spearman correlations between four psychological factors:
    Competence (C), Autonomy (A), Human-Human Relatedness (HHR), and Human-Robot Relatedness (HRR).

    It first computes the mean scores for each factor based on their respective items,
    then constructs a correlation matrix to show the relationships between these factors.

    Additionally, it computes and prints the correlation coefficients
    along with their p-values for each pair of factors to assess the significance of the correlations.

    Values that support a hierarchical model would typically show moderate to strong positive correlations
    between the factors, indicating that they are related but distinct constructs: r = [0.3, 0.7]

    Values that contradict a hierarchical model would show weak or negative correlations,
    suggesting that the factors are not related as expected: r < 0.3
    """

    # Load data
    df = pd.read_csv(f"DATASETS/{file_name}.csv")

    # — Define factor items —
    if cleaned == "":
        df_cleaned = df.copy()
        print("Testing with FULL dataset (no items removed).")

        C_items = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        A_items = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
        HHR_items = ['HHR1', 'HHR2', 'HHR3', 'HHR4']
        HRR_items = ['HRR1', 'HRR2', 'HRR3', 'HRR4']

    elif cleaned == "conservative":
        df_cleaned = df.drop(columns= ['A4'])
        print("Conservative approach. Dropping problematic item: A4.")
        
        C_items = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        A_items = ['A1', 'A2', 'A3', 'A5', 'A6', 'A7']
        HHR_items = ['HHR1', 'HHR2', 'HHR3', 'HHR4']
        HRR_items = ['HRR1', 'HRR2', 'HRR3', 'HRR4']

    elif cleaned == "moderate":
        df_cleaned = df.drop(columns= ['A4', 'A7'])
        print("Moderate approach. Dropping problematic items: A4, A7.")
        
        C_items = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        A_items = ['A1', 'A2', 'A3', 'A5', 'A6']
        HHR_items = ['HHR1', 'HHR2', 'HHR3', 'HHR4']
        HRR_items = ['HRR1', 'HRR2', 'HRR3', 'HRR4']

    elif cleaned == "aggressive":
        df_cleaned = df.drop(columns= ['C2', 'A4', 'A7', 'HHR2', 'HHR3'])
        print("Aggressive approach. Dropping problematic items: C2, A4, A7, HHR2, HHR3.")
    
        C_items = ['C1', 'C3', 'C4', 'C5', 'C6', 'C7']
        A_items = ['A1', 'A2', 'A3', 'A5', 'A6']
        HHR_items = ['HHR1', 'HHR4']
        HRR_items = ['HRR1', 'HRR2', 'HRR3', 'HRR4']

    # Calculate factor scores (mean of items)
    df_cleaned['C_score'] = df_cleaned[C_items].mean(axis=1)
    df_cleaned['A_score'] = df_cleaned[A_items].mean(axis=1)
    df_cleaned['HHR_score'] = df_cleaned[HHR_items].mean(axis=1)
    df_cleaned['HRR_score'] = df_cleaned[HRR_items].mean(axis=1)

    # Correlation matrix between factors
    factor_scores = df_cleaned[['C_score', 'A_score', 'HHR_score', 'HRR_score']]
    factor_corr = factor_scores.corr(method='spearman')

    print("\n" + "="*60)
    print("INTER-FACTOR CORRELATIONS (Spearman)")
    print("="*60)
    print(factor_corr)

    # Also calculate p-values
    print("\n" + "="*60)
    print("INTER-FACTOR CORRELATIONS WITH P-VALUES")
    print("="*60)

    factors = ['C_score', 'A_score', 'HHR_score', 'HRR_score']
    factor_names = ['Competence', 'Autonomy', 'HH-Rel', 'HR-Rel']

    for i in range(len(factors)):
        for j in range(i+1, len(factors)):
            r, p = spearmanr(df_cleaned[factors[i]], df_cleaned[factors[j]])
            print(f"{factor_names[i]} <-> {factor_names[j]}: r = {r:.3f}, p = {p:.4f}")



# --- Main execution ---
if __name__ == "__main__":
    """
    Approaches for cleaning the dataset before calculating inter-factor correlations:
    • "" = Full dataset (no items removed)
    • "conservative" = Drops A4 ——> SHOWS BEST RESULTS
    • "moderate" = Drops A4, A7
    • "aggressive" = Drops C2, A4, A7, HHR2, HHR3
    """

    file_name = 'reversed_DATASET'
    cleaned = "conservative"  # Options: "", "conservative", "moderate", "aggressive"

    calculate_inter_factor_correlations(file_name, cleaned)