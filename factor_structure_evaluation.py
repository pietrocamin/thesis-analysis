import warnings
warnings.filterwarnings("ignore", message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")

import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.stats import spearmanr


# -- Helper function: compute subscale scores --
def compute_subscale_scores(df, items_dict):
    """
    Computes mean subscale scores for each factor, which are then used as the unit of analysis for inter-factor correlations.

    param df: DataFrame containing item-level data
    param items_dict: dictionary with factor names as keys and item lists as values
    return: DataFrame of subscale mean scores
    """
    scores = {}
    for factor, items in items_dict.items():
        scores[factor] = df[items].mean(axis=1)
    return pd.DataFrame(scores)


# -- Inter-factor Spearman correlation matrix with p-values --
def compute_interfactor_correlations(subscale_scores):
    """
    Computes Spearman inter-factor correlation matrix and corresponding p-values.

    param subscale_scores: DataFrame of subscale mean scores (one column per factor)
    return: tuple of (correlation DataFrame, p-value DataFrame)
    """
    factors = subscale_scores.columns.tolist()
    n = len(factors)

    corr_matrix = pd.DataFrame(np.ones((n, n)), index=factors, columns=factors)
    pval_matrix = pd.DataFrame(np.zeros((n, n)), index=factors, columns=factors)

    for i in range(n):
        for j in range(i + 1, n):
            r, p = spearmanr(subscale_scores.iloc[:, i], subscale_scores.iloc[:, j])
            corr_matrix.iloc[i, j] = round(r, 4)
            corr_matrix.iloc[j, i] = round(r, 4)
            pval_matrix.iloc[i, j] = round(p, 6)
            pval_matrix.iloc[j, i] = round(p, 6)

    return corr_matrix, pval_matrix


# -- Interpret correlation strength --
def interpret_correlation(r):
    """
    Returns a qualitative label for a Spearman correlation coefficient.
    """
    abs_r = abs(r)
    if abs_r < 0.10:
        return "Negligible"
    elif abs_r < 0.30:
        return "Weak"
    elif abs_r < 0.50:
        return "Moderate"
    elif abs_r < 0.70:
        return "Strong"
    else:
        return "Very Strong"


# -- Summarise inter-factor correlations into a long-form table --
def summarise_correlations(corr_matrix, pval_matrix):
    """
    Converts the correlation and p-value matrices into a readable long-format DataFrame.

    param corr_matrix: square DataFrame of Spearman r values
    param pval_matrix: square DataFrame of p-values
    return: long-format DataFrame with factor pair, r, p, significance, and interpretation
    """
    factors = corr_matrix.columns.tolist()
    rows = []

    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            r = corr_matrix.iloc[i, j]
            p = pval_matrix.iloc[i, j]
            rows.append({
                'Factor A': factors[i],
                'Factor B': factors[j],
                'Spearman_r': r,
                'p_value': p,
                'Significant (p < .05)': 'Yes' if p < 0.05 else 'No',
                'Interpretation': interpret_correlation(r)
            })

    return pd.DataFrame(rows)


# -- Assess rotation justification --
def assess_rotation_justification(summary_df, threshold=0.20):
    """
    Evaluates whether oblique or orthogonal rotation is more appropriate
    based on the magnitude of inter-factor correlations.

    Oblique rotation (e.g., Oblimin/Promax) is recommended when factors
    correlate at r >= threshold, as correlations suggest a common higher-order factor.
    Orthogonal rotation (e.g., Varimax) assumes factor independence.

    param summary_df: long-format correlation summary DataFrame
    param threshold: minimum r to flag as non-negligible (default 0.20)
    return: string recommendation
    """
    sig_pairs = summary_df[summary_df['Significant (p < .05)'] == 'Yes']
    above_threshold = sig_pairs[sig_pairs['Spearman_r'].abs() >= threshold]

    if len(above_threshold) == 0:
        return ("All inter-factor correlations are negligible or non-significant. "
                "Orthogonal rotation (Varimax) is appropriate for EFA.")
    else:
        prop = len(above_threshold) / len(summary_df)
        return (f"{len(above_threshold)} of {len(summary_df)} factor pairs show significant correlations "
                f"at r >= {threshold} ({prop:.0%} of pairs). "
                "Oblique rotation (Oblimin or Promax) is recommended for EFA, "
                "as inter-factor correlations are consistent with a higher-order WB construct.")


# -- Assess hierarchical model support --
def assess_hierarchical_support(summary_df):
    """
    Provides a preliminary assessment of whether inter-factor correlations
    support the hypothesised hierarchical WB model.

    A hierarchical model requires that first-order factors are meaningfully
    correlated (supporting a common higher-order factor), but not so highly
    correlated as to suggest redundancy (r > 0.85 would raise multicollinearity concerns).

    param summary_df: long-format correlation summary DataFrame
    return: string assessment
    """
    rs = summary_df['Spearman_r'].abs()
    mean_r = rs.mean()
    max_r = rs.max()
    min_r = rs.min()
    n_sig = (summary_df['Significant (p < .05)'] == 'Yes').sum()
    n_total = len(summary_df)

    lines = [
        f"Mean inter-factor |r| = {mean_r:.3f} (range: {min_r:.3f}-{max_r:.3f})",
        f"Significant pairs: {n_sig}/{n_total}",
    ]

    if max_r > 0.85:
        lines.append("WARNING: At least one factor pair shows very high correlation (r > 0.85), "
                     "suggesting possible redundancy. Consider merging those factors.")
    elif mean_r >= 0.20 and n_sig >= (n_total // 2):
        lines.append("Inter-factor correlations are moderate and mostly significant, "
                     "providing preliminary support for a higher-order WB factor. "
                     "This is consistent with the hypothesised hierarchical model.")
    else:
        lines.append("Inter-factor correlations are weak or largely non-significant. "
                     "Support for a higher-order WB factor is limited at this stage.")

    return "\n".join(lines)


# --- Main execution ---
if __name__ == "__main__":
    file_name = 'reversed_DATASET_problematic_items_clean.csv'
    path = f"DATASETS/{file_name}"

    if os.path.exists(path):
        df = pd.read_csv(path)

        items_dict = {
            'Competence':       ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
            'Autonomy':         ['A1', 'A2', 'A3', 'A5', 'A6', 'A7'],
            'HH-Relatedness':   ['HHR1', 'HHR2', 'HHR3', 'HHR4'],
            'HR-Relatedness':   ['HRR1', 'HRR2', 'HRR3', 'HRR4'],
        }

        output_dir = 'output/factor_structure'
        os.makedirs(output_dir, exist_ok=True)

        # 1. Compute subscale scores
        print("Computing subscale scores...")
        subscale_scores = compute_subscale_scores(df, items_dict)

        # 2. Compute inter-factor correlations
        print("Computing inter-factor Spearman correlations...")
        corr_matrix, pval_matrix = compute_interfactor_correlations(subscale_scores)

        # 3. Summarise into long-format table
        summary_df = summarise_correlations(corr_matrix, pval_matrix)

        # 4. Rotation justification
        print("\n--- Rotation Justification ---")
        rotation_note = assess_rotation_justification(summary_df)
        print(rotation_note)

        # 5. Hierarchical model support
        print("\n--- Hierarchical Model Assessment ---")
        hierarchical_note = assess_hierarchical_support(summary_df)
        print(hierarchical_note)

        # 6. Save outputs
        corr_matrix.to_csv(f"{output_dir}/interfactor_correlation_matrix.csv")
        pval_matrix.to_csv(f"{output_dir}/interfactor_pvalue_matrix.csv")
        summary_df.to_csv(f"{output_dir}/interfactor_correlation_summary.csv", index=False)

        # Save narrative notes
        with open(f"{output_dir}/factor_structure_notes.txt", "w") as f:
            f.write("=== ROTATION JUSTIFICATION ===\n")
            f.write(rotation_note + "\n\n")
            f.write("=== HIERARCHICAL MODEL ASSESSMENT ===\n")
            f.write(hierarchical_note + "\n")

        print(f"\nAll factor structure outputs saved to '{output_dir}/'.")
        print("\nCorrelation matrix:")
        print(corr_matrix)
        print("\nSummary table:")
        print(summary_df.to_string(index=False))

    else:
        print(f"Error: File not found at {path}")