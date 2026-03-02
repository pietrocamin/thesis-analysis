import warnings
warnings.filterwarnings("ignore", message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import chi2
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity


# =============================================================================
# 1. FACTORABILITY ASSESSMENT
# =============================================================================

def compute_kmo(df, items):
    """
    Computes the Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy.

    KMO assesses whether the partial correlations among items are small
    relative to the ordinary correlations, indicating that factor analysis
    is likely to yield distinct and reliable factors.

    Interpretation (Kaiser, 1974):
        >= 0.90 : Marvellous
        >= 0.80 : Meritorious
        >= 0.70 : Middling
        >= 0.60 : Mediocre
        >= 0.50 : Miserable
        <  0.50 : Unacceptable

    param df: DataFrame containing item-level data
    param items: list of item column names
    return: tuple of (kmo_per_item Series, kmo_overall float)
    """
    data = df[items].values
    kmo_per_item, kmo_overall = calculate_kmo(data)
    return pd.Series(kmo_per_item, index=items), kmo_overall


def interpret_kmo(kmo_value):
    """Returns a qualitative label for a KMO value (Kaiser, 1974)."""
    if kmo_value >= 0.90:
        return "Marvellous"
    elif kmo_value >= 0.80:
        return "Meritorious"
    elif kmo_value >= 0.70:
        return "Middling"
    elif kmo_value >= 0.60:
        return "Mediocre"
    elif kmo_value >= 0.50:
        return "Miserable"
    else:
        return "Unacceptable"


def compute_bartlett(df, items):
    """
    Computes Bartlett's Test of Sphericity.

    Tests the null hypothesis that the correlation matrix is an identity matrix
    (i.e., items are uncorrelated). A significant result (p < .05) indicates
    that the correlation matrix is factorable.

    param df: DataFrame containing item-level data
    param items: list of item column names
    return: dict with chi-square statistic, degrees of freedom, and p-value
    """
    data = df[items].values
    chi2_val, p_val = calculate_bartlett_sphericity(data)
    n_items = len(items)
    df_val = n_items * (n_items - 1) / 2
    return {
        'chi2': round(chi2_val, 3),
        'df': int(df_val),
        'p_value': p_val,
        'Significant (p < .05)': 'Yes' if p_val < 0.05 else 'No'
    }


# =============================================================================
# 2. NUMBER OF FACTORS DETERMINATION
# =============================================================================

def compute_eigenvalues(df, items):
    """
    Computes eigenvalues from Principal Axis Factoring for scree plot analysis.

    The Kaiser criterion (eigenvalue > 1) is noted but not recommended as the
    sole decision rule, as it tends to overestimate the number of factors.

    param df: DataFrame containing item-level data
    param items: list of item column names
    return: numpy array of eigenvalues
    """
    fa = FactorAnalyzer(n_factors=len(items), rotation=None, method='principal')
    fa.fit(df[items])
    ev, _ = fa.get_eigenvalues()
    return ev


def parallel_analysis(df, items, n_iterations=1000, percentile=95, random_state=42):
    """
    Performs Parallel Analysis to determine the number of factors.

    Compares observed eigenvalues from PAF against eigenvalues generated from
    random data of the same shape. Factors whose observed eigenvalues exceed
    the specified percentile of the random distribution are retained.

    This is widely considered the most accurate method for factor number determination.

    param df: DataFrame containing item-level data
    param items: list of item column names
    param n_iterations: number of random datasets to generate
    param percentile: percentile threshold for the random eigenvalue distribution
    param random_state: seed for reproducibility
    return: dict with observed eigenvalues, random eigenvalue means and
            percentiles, and the suggested number of factors
    """
    rng = np.random.default_rng(random_state)
    data = df[items].values
    n_obs, n_vars = data.shape

    # Observed eigenvalues
    fa_obs = FactorAnalyzer(n_factors=n_vars, rotation=None, method='principal')
    fa_obs.fit(data)
    observed_ev, _ = fa_obs.get_eigenvalues()

    # Random eigenvalues
    random_evs = []
    for _ in range(n_iterations):
        random_data = rng.standard_normal((n_obs, n_vars))
        fa_rand = FactorAnalyzer(n_factors=n_vars, rotation=None, method='principal')
        fa_rand.fit(random_data)
        rand_ev, _ = fa_rand.get_eigenvalues()
        random_evs.append(rand_ev)

    random_evs = np.array(random_evs)
    random_mean = np.mean(random_evs, axis=0)
    random_pct = np.percentile(random_evs, percentile, axis=0)

    # Number of factors: observed EV > random percentile
    n_factors_suggested = int(np.sum(observed_ev > random_pct))

    return {
        'observed_eigenvalues': observed_ev,
        'random_mean': random_mean,
        'random_percentile': random_pct,
        'percentile_used': percentile,
        'n_factors_suggested': n_factors_suggested
    }


def map_test(df, items):
    """
    Performs Velicer's Minimum Average Partial (MAP) test.

    Extracts increasing numbers of components and computes the average squared
    partial correlation after partialling out each component. The number of
    factors is the point at which the average squared partial correlation is
    minimised, indicating that the most systematic variance has been extracted.

    param df: DataFrame containing item-level data
    param items: list of item column names
    return: dict with MAP values per number of components and suggested n_factors
    """
    data = df[items].values
    n_obs, n_vars = data.shape

    # Standardise
    data_std = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
    corr_matrix = np.corrcoef(data_std, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    map_values = []

    # Step 0: average squared correlation (no components partialled)
    corr_sq = corr_matrix ** 2
    np.fill_diagonal(corr_sq, 0)
    n_offdiag = n_vars * (n_vars - 1)
    map0 = np.sum(corr_sq) / n_offdiag
    map_values.append(map0)

    for k in range(1, n_vars):
        # Partial correlation matrix after removing k components
        V = eigenvectors[:, :k]
        partial_corr = corr_matrix - V @ np.diag(eigenvalues[:k]) @ V.T
        # Normalise to partial correlation scale
        d = np.sqrt(np.diag(partial_corr))
        d[d == 0] = 1e-10
        partial_corr_norm = partial_corr / np.outer(d, d)

        sq = partial_corr_norm ** 2
        np.fill_diagonal(sq, 0)
        map_k = np.sum(sq) / n_offdiag
        map_values.append(map_k)

    map_values = np.array(map_values)
    n_factors_suggested = int(np.argmin(map_values))  # index = number of factors retained

    return {
        'map_values': map_values,
        'n_factors_suggested': n_factors_suggested
    }


# =============================================================================
# 3. FACTOR EXTRACTION AND ROTATION
# =============================================================================

def run_efa(df, items, n_factors, rotation='oblimin', method='principal'):
    """
    Runs Exploratory Factor Analysis with the specified number of factors,
    rotation method, and extraction method.

    Principal Axis Factoring (PAF) is used as the extraction method, as it
    is robust to violations of multivariate normality and appropriate for
    ordinal Likert data (Fabrigar et al., 1999).

    Oblimin rotation is applied as inter-factor correlations (established in
    the factor structure evaluation) indicate that factors are correlated,
    making orthogonal rotation inappropriate.

    param df: DataFrame containing item-level data
    param items: list of item column names
    param n_factors: number of factors to extract
    param rotation: rotation method (default: 'oblimin')
    param method: extraction method (default: 'principal' = PAF)
    return: fitted FactorAnalyzer object
    """
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method)
    fa.fit(df[items])
    return fa


def get_loadings_table(fa, items, n_factors, threshold=0.30):
    """
    Constructs a factor loadings table with communalities and flags.

    Loadings below the specified threshold are suppressed for readability,
    following the convention that loadings >= 0.30 are considered meaningful
    for interpretation (Hair et al., 2019).

    Cross-loadings are flagged when an item loads >= threshold on more than
    one factor.

    param fa: fitted FactorAnalyzer object
    param items: list of item column names
    param n_factors: number of factors extracted
    param threshold: minimum absolute loading to display (default: 0.30)
    return: DataFrame with loadings, communalities, and cross-loading flags
    """
    loadings = fa.loadings_
    communalities = fa.get_communalities()

    factor_labels = [f'Factor {i+1}' for i in range(n_factors)]
    df_loadings = pd.DataFrame(loadings, index=items, columns=factor_labels)

    # Communalities
    df_loadings['Communality'] = communalities

    # Primary factor (highest absolute loading)
    df_loadings['Primary Factor'] = df_loadings[factor_labels].abs().idxmax(axis=1)

    # Cross-loading flag
    above_threshold = df_loadings[factor_labels].abs() >= threshold
    df_loadings['Cross-loading'] = above_threshold.sum(axis=1) > 1

    # Suppress loadings below threshold for display
    display_df = df_loadings.copy()
    for col in factor_labels:
        display_df[col] = display_df[col].apply(
            lambda x: round(x, 3) if abs(x) >= threshold else ''
        )
    display_df['Communality'] = display_df['Communality'].round(3)

    return df_loadings, display_df


def get_variance_explained(fa, n_factors):
    """
    Returns variance explained statistics for each factor.

    param fa: fitted FactorAnalyzer object
    param n_factors: number of factors extracted
    return: DataFrame with SS loadings, proportion variance, cumulative variance
    """
    variance = fa.get_factor_variance()
    factor_labels = [f'Factor {i+1}' for i in range(n_factors)]
    df_var = pd.DataFrame(
        variance,
        index=['SS Loadings', 'Proportion Variance', 'Cumulative Variance'],
        columns=factor_labels
    ).T
    return df_var.round(4)


# =============================================================================
# 4. VISUALISATION
# =============================================================================

def plot_scree_parallel(eigenvalues, pa_results, output_path):
    """
    Plots a combined scree plot with parallel analysis results.

    Displays observed eigenvalues alongside the random data mean and
    95th percentile from parallel analysis, with a Kaiser criterion
    reference line at eigenvalue = 1.

    param eigenvalues: array of observed eigenvalues
    param pa_results: dict returned by parallel_analysis()
    param output_path: file path to save the figure
    """
    n = len(eigenvalues)
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, eigenvalues, 'o-', color='#2c4a7c', linewidth=2,
            markersize=6, label='Observed eigenvalues (PAF)')
    ax.plot(x, pa_results['random_mean'], 's--', color='#a0a0a0', linewidth=1.5,
            markersize=5, label='Random data mean (Parallel Analysis)')
    ax.plot(x, pa_results['random_percentile'], '^--', color='#e07b39', linewidth=1.5,
            markersize=5, label=f"Random data {pa_results['percentile_used']}th percentile")
    ax.axhline(y=1, color='#888888', linestyle=':', linewidth=1.2, label='Kaiser criterion (λ = 1)')

    # Shade suggested retention zone
    n_sug = pa_results['n_factors_suggested']
    ax.axvspan(0.5, n_sug + 0.5, alpha=0.07, color='#2c4a7c', label=f'Suggested factors (n={n_sug})')

    ax.set_xlabel('Factor Number', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title('Scree Plot with Parallel Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scree plot saved to {output_path}")


def plot_map_test(map_results, output_path):
    """
    Plots the MAP test values across number of components.

    The minimum point indicates the suggested number of factors.

    param map_results: dict returned by map_test()
    param output_path: file path to save the figure
    """
    map_values = map_results['map_values']
    n_sug = map_results['n_factors_suggested']
    x = np.arange(0, len(map_values))

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, map_values, 'o-', color='#2c4a7c', linewidth=2, markersize=6)
    ax.axvline(x=n_sug, color='#e07b39', linestyle='--', linewidth=1.5,
               label=f'Suggested n factors = {n_sug}')
    ax.scatter([n_sug], [map_values[n_sug]], color='#e07b39', s=80, zorder=5)

    ax.set_xlabel('Number of Components Partialled Out', fontsize=12)
    ax.set_ylabel('Average Squared Partial Correlation', fontsize=12)
    ax.set_title("Velicer's MAP Test", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MAP test plot saved to {output_path}")


def plot_loadings_heatmap(loadings_df, factor_labels, output_path, threshold=0.30):
    """
    Plots a heatmap of factor loadings, masking values below threshold.

    param loadings_df: raw loadings DataFrame (from get_loadings_table)
    param factor_labels: list of factor column names
    param output_path: file path to save the figure
    param threshold: loadings below this value are greyed out
    """
    import matplotlib.colors as mcolors

    data = loadings_df[factor_labels].values
    items = loadings_df.index.tolist()
    n_items, n_factors = data.shape

    fig, ax = plt.subplots(figsize=(n_factors * 1.6 + 2, n_items * 0.45 + 1.5))

    cmap = plt.cm.RdYlBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')

    for i in range(n_items):
        for j in range(n_factors):
            val = data[i, j]
            text = f'{val:.2f}' if abs(val) >= threshold else ''
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                    color=color, fontweight='bold' if abs(val) >= 0.50 else 'normal')

    ax.set_xticks(range(n_factors))
    ax.set_xticklabels(factor_labels, fontsize=11)
    ax.set_yticks(range(n_items))
    ax.set_yticklabels(items, fontsize=10)
    ax.set_title('Factor Loadings Heatmap (PAF, Oblimin rotation)',
                 fontsize=13, fontweight='bold', pad=12)

    plt.colorbar(im, ax=ax, shrink=0.6, label='Loading')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loadings heatmap saved to {output_path}")


# =============================================================================
# 5. SUMMARY REPORTING
# =============================================================================

def summarise_factor_determination(eigenvalues, pa_results, map_results, kaiser_threshold=1.0):
    """
    Produces a consolidated summary of all factor number determination methods.

    param eigenvalues: array of observed eigenvalues
    param pa_results: dict from parallel_analysis()
    param map_results: dict from map_test()
    param kaiser_threshold: eigenvalue cutoff for Kaiser criterion (default: 1.0)
    return: dict summarising each method's suggestion and a convergence note
    """
    n_kaiser = int(np.sum(eigenvalues > kaiser_threshold))
    n_parallel = pa_results['n_factors_suggested']
    n_map = map_results['n_factors_suggested']

    suggestions = [n_kaiser, n_parallel, n_map]
    converge = len(set(suggestions)) == 1

    summary = {
        'Kaiser criterion (λ > 1)': n_kaiser,
        'Parallel Analysis (95th pct)': n_parallel,
        'MAP test (Velicer)': n_map,
        'Convergence': 'Full agreement' if converge else f'Divergent — values: {suggestions}',
        'Recommended n_factors': n_parallel  # PA is most reliable; used as primary
    }
    return summary


# =============================================================================
# 6. MULTI-SOLUTION COMPARISON
# =============================================================================

def compare_factor_solutions(df, items, n_factors_list, theoretical_mapping,
                              rotation='oblimin', method='principal', threshold=0.30):
    """
    Extracts and compares multiple EFA solutions across a range of factor numbers.

    For each solution, computes:
      - Factor loadings with primary factor assignment and cross-loading flags
      - Communality statistics (mean, % below threshold)
      - Congruence with the theoretical item-factor mapping
      - Variance explained
      - Factor intercorrelations (phi matrix)

    Congruence is defined as the proportion of items whose primary factor
    matches their theoretically expected factor. This requires that the
    analyst manually inspects the factor labels and assigns them after
    running (see 'theoretical_mapping' parameter).

    param df: DataFrame containing item-level data
    param items: list of item column names
    param n_factors_list: list of integers specifying solutions to extract
    param theoretical_mapping: dict mapping item -> expected factor label
                               (used only when n_factors matches theory)
    param rotation: oblique rotation method (default: 'oblimin')
    param method: extraction method (default: 'principal' = PAF)
    param threshold: loading threshold for primary factor assignment and flags
    return: dict keyed by n_factors, each containing solution details
    """
    solutions = {}

    for n in n_factors_list:
        print(f"\n  Extracting {n}-factor solution...")
        try:
            fa = FactorAnalyzer(n_factors=n, rotation=rotation, method=method)
            fa.fit(df[items])

            loadings = fa.loadings_
            communalities = fa.get_communalities()
            factor_labels = [f'F{i+1}' for i in range(n)]

            df_load = pd.DataFrame(loadings, index=items, columns=factor_labels)
            df_load['Communality'] = communalities
            df_load['Primary_Factor'] = df_load[factor_labels].abs().idxmax(axis=1)
            df_load['Max_Loading'] = df_load[factor_labels].abs().max(axis=1)

            above = df_load[factor_labels].abs() >= threshold
            df_load['Cross_loading'] = above.sum(axis=1) > 1
            df_load['Below_threshold'] = df_load['Max_Loading'] < threshold

            variance = fa.get_factor_variance()
            var_df = pd.DataFrame(
                variance,
                index=['SS_Loadings', 'Proportion_Variance', 'Cumulative_Variance'],
                columns=factor_labels
            ).T.round(4)

            phi = None
            if hasattr(fa, 'phi_') and fa.phi_ is not None:
                phi = pd.DataFrame(fa.phi_, index=factor_labels,
                                   columns=factor_labels).round(4)

            # Summary statistics
            n_cross = df_load['Cross_loading'].sum()
            n_low_comm = (df_load['Communality'] < threshold).sum()
            n_below_thresh = df_load['Below_threshold'].sum()
            mean_comm = df_load['Communality'].mean()
            cumulative_var = var_df['Cumulative_Variance'].iloc[-1]

            solutions[n] = {
                'fa': fa,
                'loadings': df_load,
                'variance': var_df,
                'phi': phi,
                'n_cross_loading': int(n_cross),
                'n_low_communality': int(n_low_comm),
                'n_below_threshold': int(n_below_thresh),
                'mean_communality': round(float(mean_comm), 4),
                'cumulative_variance': float(cumulative_var),
            }

        except Exception as e:
            print(f"    WARNING: {n}-factor solution failed — {e}")
            solutions[n] = None

    return solutions


def print_solution_summary(solutions, items):
    """
    Prints a concise comparison table across all extracted solutions.

    param solutions: dict returned by compare_factor_solutions()
    param items: list of item column names
    """
    print("\n" + "-" * 70)
    print("MULTI-SOLUTION COMPARISON SUMMARY")
    print("-" * 70)

    rows = []
    for n, sol in solutions.items():
        if sol is None:
            continue
        rows.append({
            'N Factors': n,
            'Cumul. Variance (%)': round(sol['cumulative_variance'] * 100, 1),
            'Mean Communality': sol['mean_communality'],
            'Cross-loading items': sol['n_cross_loading'],
            'Low communality (<.30)': sol['n_low_communality'],
            'Items below threshold': sol['n_below_threshold'],
        })

    summary_df = pd.DataFrame(rows).set_index('N Factors')
    print(summary_df.to_string())
    return summary_df


def print_solution_loadings(solutions, n):
    """
    Prints the full loading table for a specific solution.

    param solutions: dict returned by compare_factor_solutions()
    param n: number of factors to display
    """
    if solutions.get(n) is None:
        print(f"No {n}-factor solution available.")
        return

    sol = solutions[n]
    factor_labels = [f'F{i+1}' for i in range(n)]
    display = sol['loadings'].copy()

    for col in factor_labels:
        display[col] = display[col].apply(
            lambda x: f'{x:.3f}' if abs(x) >= 0.30 else '     '
        )

    print(f"\n{'-' * 70}")
    print(f"{n}-FACTOR SOLUTION — Loadings (suppressed < 0.30)")
    print(f"{'-' * 70}")
    cols = factor_labels + ['Communality', 'Primary_Factor', 'Cross_loading']
    print(display[cols].to_string())
    print(f"\nVariance Explained:")
    print(sol['variance'].to_string())
    if sol['phi'] is not None:
        print(f"\nFactor Intercorrelations (Phi):")
        print(sol['phi'].to_string())


def plot_communality_comparison(solutions, items, output_path):
    """
    Plots item communalities across all extracted solutions as a grouped bar chart.

    Higher communalities indicate that the solution explains more variance
    per item. This plot helps identify items that are consistently poorly
    explained regardless of the number of factors extracted.

    param solutions: dict returned by compare_factor_solutions()
    param items: list of item column names
    param output_path: file path to save the figure
    """
    valid = {n: s for n, s in solutions.items() if s is not None}
    n_items = len(items)
    n_solutions = len(valid)

    x = np.arange(n_items)
    width = 0.8 / n_solutions
    colors = ['#2c4a7c', '#e07b39', '#3a8a5c', '#9b4a8c']

    fig, ax = plt.subplots(figsize=(max(14, n_items * 0.7), 5))

    for i, (n, sol) in enumerate(valid.items()):
        comm = sol['loadings']['Communality'].values
        offset = (i - n_solutions / 2 + 0.5) * width
        bars = ax.bar(x + offset, comm, width, label=f'{n}-factor solution',
                      color=colors[i % len(colors)], alpha=0.85, edgecolor='white')

    ax.axhline(y=0.30, color='#cc3333', linestyle='--', linewidth=1.2,
               label='Communality threshold (0.30)')
    ax.set_xlabel('Item', fontsize=12)
    ax.set_ylabel('Communality', fontsize=12)
    ax.set_title('Item Communalities Across Factor Solutions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(items, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Communality comparison plot saved to {output_path}")


def plot_loading_heatmaps_grid(solutions, n_factors_list, output_path, threshold=0.30):
    """
    Saves individual heatmaps for each solution into a single multi-panel figure.

    param solutions: dict returned by compare_factor_solutions()
    param n_factors_list: list of factor numbers to plot
    param output_path: file path to save the figure
    param threshold: loadings below this are not labelled
    """
    import matplotlib.colors as mcolors

    valid = [(n, solutions[n]) for n in n_factors_list if solutions.get(n) is not None]
    n_panels = len(valid)

    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 5, 10))
    if n_panels == 1:
        axes = [axes]

    cmap = plt.cm.RdYlBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    for ax, (n, sol) in zip(axes, valid):
        factor_labels = [f'F{i+1}' for i in range(n)]
        items_list = sol['loadings'].index.tolist()
        data = sol['loadings'][factor_labels].values

        im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')

        for i in range(len(items_list)):
            for j in range(n):
                val = data[i, j]
                text = f'{val:.2f}' if abs(val) >= threshold else ''
                color = 'white' if abs(val) > 0.60 else 'black'
                ax.text(j, i, text, ha='center', va='center', fontsize=8,
                        color=color,
                        fontweight='bold' if abs(val) >= 0.50 else 'normal')

        ax.set_xticks(range(n))
        ax.set_xticklabels(factor_labels, fontsize=10)
        ax.set_yticks(range(len(items_list)))
        ax.set_yticklabels(items_list, fontsize=9)
        ax.set_title(f'{n}-Factor Solution', fontsize=12, fontweight='bold', pad=10)
        plt.colorbar(im, ax=ax, shrink=0.5, label='Loading')

    fig.suptitle('Factor Loadings Comparison (PAF, Oblimin)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loading heatmaps grid saved to {output_path}")


def save_all_solutions(solutions, output_dir):
    """
    Saves loadings, variance, phi matrix, and flags for every solution
    into individual subdirectories under output_dir/solutions/.

    param solutions: dict returned by compare_factor_solutions()
    param output_dir: base output directory
    """
    for n, sol in solutions.items():
        if sol is None:
            continue
        sol_dir = f"{output_dir}/solutions/{n}_factors"
        os.makedirs(sol_dir, exist_ok=True)

        factor_labels = [f'F{i+1}' for i in range(n)]

        # Raw loadings
        sol['loadings'].round(4).to_csv(f"{sol_dir}/loadings_raw.csv")

        # Display loadings (suppressed)
        display = sol['loadings'].copy()
        for col in factor_labels:
            display[col] = display[col].apply(
                lambda x: round(x, 3) if abs(x) >= 0.30 else ''
            )
        display.to_csv(f"{sol_dir}/loadings_display.csv")

        # Variance
        sol['variance'].to_csv(f"{sol_dir}/variance_explained.csv")

        # Phi matrix
        if sol['phi'] is not None:
            sol['phi'].to_csv(f"{sol_dir}/factor_correlations_phi.csv")

        # Flags
        flags = pd.DataFrame({
            'Item': sol['loadings'].index,
            'Communality': sol['loadings']['Communality'].round(4),
            'Primary_Factor': sol['loadings']['Primary_Factor'],
            'Cross_loading': sol['loadings']['Cross_loading'],
            'Low_Communality': sol['loadings']['Communality'] < 0.30,
            'Below_threshold': sol['loadings']['Below_threshold'],
        })
        flags.to_csv(f"{sol_dir}/item_flags.csv", index=False)

    print(f"All solution files saved under '{output_dir}/solutions/'")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    file_name = 'reversed_DATASET_problematic_items_clean.csv'
    path = f"DATASETS/{file_name}"

    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        exit()

    df = pd.read_csv(path)


    # -- Data preparation: item selection and theoretical mapping --
    # items = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
    #          'A1', 'A2', 'A3', 'A5', 'A6', 'A7',
    #          'HHR1', 'HHR2', 'HHR3', 'HHR4',
    #          'HRR1', 'HRR2', 'HRR3', 'HRR4']

    # # Theoretical item-factor mapping (for congruence reference)
    # theoretical_mapping = {
    #     'C1': 'Competence', 'C2': 'Competence', 'C3': 'Competence', 'C4': 'Competence',
    #     'C5': 'Competence', 'C6': 'Competence', 'C7': 'Competence',
    #     'A1': 'Autonomy',   'A2': 'Autonomy',   'A3': 'Autonomy',
    #     'A5': 'Autonomy',   'A6': 'Autonomy',   'A7': 'Autonomy',
    #     'HHR1': 'HH-Rel',  'HHR2': 'HH-Rel',  'HHR3': 'HH-Rel',  'HHR4': 'HH-Rel',
    #     'HRR1': 'HR-Rel',  'HRR2': 'HR-Rel',  'HRR3': 'HR-Rel',  'HRR4': 'HR-Rel',
    # }

    # - Without item HRR3 (problematic item identified) -
    if 'HRR3' in df.columns:
        df = df.drop(columns=['HRR3'])
        print("Note: Item 'HRR3' has been removed from the dataset for analysis.")

    items = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
             'A1', 'A2', 'A3', 'A5', 'A6', 'A7',
             'HHR1', 'HHR2', 'HHR3', 'HHR4',
             'HRR1', 'HRR2', 'HRR4']

    # Theoretical item-factor mapping (for congruence reference)
    theoretical_mapping = {
        'C1': 'Competence', 'C2': 'Competence', 'C3': 'Competence', 'C4': 'Competence',
        'C5': 'Competence', 'C6': 'Competence', 'C7': 'Competence',
        'A1': 'Autonomy',   'A2': 'Autonomy',   'A3': 'Autonomy',
        'A5': 'Autonomy',   'A6': 'Autonomy',   'A7': 'Autonomy',
        'HHR1': 'HH-Rel',  'HHR2': 'HH-Rel',  'HHR3': 'HH-Rel',  'HHR4': 'HH-Rel',
        'HRR1': 'HR-Rel',  'HRR2': 'HR-Rel',  'HRR4': 'HR-Rel',
    }

    # # - Without HHR items (problematic factor) -
    # if 'HHR1' in df.columns:
    #     df = df.drop(columns=['HHR1', 'HHR2', 'HHR3', 'HHR4'])
    #     print("Note: Items 'HHR1', 'HHR2', 'HHR3', and 'HHR4' have been removed from the dataset for analysis.")
    # if 'HRR3' in df.columns:
    #     df = df.drop(columns=['HRR3'])
    #     print("Note: Item 'HRR3' has been removed from the dataset for analysis.")

    # items = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
    #          'A1', 'A2', 'A3', 'A5', 'A6', 'A7',
    #          'HRR1', 'HRR2', 'HRR4']

    # # Theoretical item-factor mapping (for congruence reference)
    # theoretical_mapping = {
    #     'C1': 'Competence', 'C2': 'Competence', 'C3': 'Competence', 'C4': 'Competence',
    #     'C5': 'Competence', 'C6': 'Competence', 'C7': 'Competence',
    #     'A1': 'Autonomy', 'A2': 'Autonomy', 'A3': 'Autonomy',
    #     'A5': 'Autonomy', 'A6': 'Autonomy', 'A7': 'Autonomy',
    #     'HRR1': 'HR-Rel', 'HRR2': 'HR-Rel', 'HRR4': 'HR-Rel',
    # }
    # ------------------------------------------------------------------


    output_dir = 'output/efa'
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: Factorability Assessment
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: FACTORABILITY ASSESSMENT")
    print("=" * 60)

    kmo_per_item, kmo_overall = compute_kmo(df, items)
    bartlett = compute_bartlett(df, items)

    print(f"\nKMO Overall: {kmo_overall:.4f} — {interpret_kmo(kmo_overall)}")
    print(f"KMO per item:\n{kmo_per_item.round(4).to_string()}")
    print(f"\nBartlett's Test: χ²({bartlett['df']}) = {bartlett['chi2']}, "
          f"p = {bartlett['p_value']:.6f} — Significant: {bartlett['Significant (p < .05)']}")

    kmo_df = kmo_per_item.reset_index()
    kmo_df.columns = ['Item', 'KMO']
    kmo_df['Interpretation'] = kmo_df['KMO'].apply(interpret_kmo)
    kmo_df.to_csv(f"{output_dir}/kmo_per_item.csv", index=False)

    pd.DataFrame([{
        'KMO_Overall': round(kmo_overall, 4),
        'KMO_Interpretation': interpret_kmo(kmo_overall),
        'Bartlett_chi2': bartlett['chi2'],
        'Bartlett_df': bartlett['df'],
        'Bartlett_p': bartlett['p_value'],
        'Bartlett_Significant': bartlett['Significant (p < .05)']
    }]).to_csv(f"{output_dir}/factorability_summary.csv", index=False)

    # ------------------------------------------------------------------
    # STEP 2: Number of Factors Determination
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: NUMBER OF FACTORS DETERMINATION")
    print("=" * 60)

    eigenvalues = compute_eigenvalues(df, items)
    print(f"\nEigenvalues (first 8): {eigenvalues[:8].round(3)}")
    print(f"Kaiser criterion (λ > 1): {int(np.sum(eigenvalues > 1))} factors")

    print("\nRunning Parallel Analysis (1000 iterations)...")
    pa_results = parallel_analysis(df, items, n_iterations=1000)
    print(f"Parallel Analysis suggests: {pa_results['n_factors_suggested']} factors")

    print("\nRunning MAP test...")
    map_results = map_test(df, items)
    print(f"MAP test suggests: {map_results['n_factors_suggested']} factors")

    factor_summary = summarise_factor_determination(eigenvalues, pa_results, map_results)
    print(f"\nFactor determination summary:\n{pd.Series(factor_summary).to_string()}")

    pd.DataFrame({
        'Factor': range(1, len(eigenvalues) + 1),
        'Observed_EV': eigenvalues.round(4),
        'Random_Mean_EV': pa_results['random_mean'].round(4),
        'Random_95pct_EV': pa_results['random_percentile'].round(4),
        'Retained_PA': eigenvalues > pa_results['random_percentile']
    }).to_csv(f"{output_dir}/eigenvalues_parallel_analysis.csv", index=False)

    pd.DataFrame({
        'n_components_partialled': range(len(map_results['map_values'])),
        'MAP_value': map_results['map_values'].round(6)
    }).to_csv(f"{output_dir}/map_test_values.csv", index=False)

    pd.DataFrame([factor_summary]).to_csv(
        f"{output_dir}/factor_determination_summary.csv", index=False)

    plot_scree_parallel(eigenvalues, pa_results,
                        f"{output_dir}/scree_parallel_analysis.png")
    plot_map_test(map_results, f"{output_dir}/map_test.png")

    # ------------------------------------------------------------------
    # STEP 3: Multi-Solution Comparison (2, 3, 4 factors)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: MULTI-SOLUTION EXTRACTION (2, 3, 4, 5 FACTORS)")
    print("=" * 60)

    # Solutions to compare:
    #   2 = empirically suggested (PA + MAP)
    #   3 = intermediate — tests whether a 3-factor structure emerges
    #   4 = theoretically hypothesised (Competence, Autonomy, HHR, HRR)
    #   5 = exploratory — tests whether a more complex structure emerges that may better fit the data
    n_factors_to_compare = [2, 3, 4, 5]

    solutions = compare_factor_solutions(
        df, items,
        n_factors_list=n_factors_to_compare,
        theoretical_mapping=theoretical_mapping,
        rotation='oblimin',
        method='principal',
        threshold=0.30
    )

    # Print comparison summary table
    summary_df = print_solution_summary(solutions, items)
    summary_df.to_csv(f"{output_dir}/multi_solution_comparison.csv")

    # Print full loadings for each solution
    for n in n_factors_to_compare:
        print_solution_loadings(solutions, n)

    # ------------------------------------------------------------------
    # STEP 4: Visualisations
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: VISUALISATIONS")
    print("=" * 60)

    plot_communality_comparison(
        solutions, items,
        f"{output_dir}/communality_comparison.png"
    )

    plot_loading_heatmaps_grid(
        solutions, n_factors_to_compare,
        f"{output_dir}/loading_heatmaps_grid.png",
        threshold=0.30
    )

    # Individual heatmaps per solution
    for n in n_factors_to_compare:
        if solutions.get(n):
            factor_labels = [f'F{i+1}' for i in range(n)]
            plot_loadings_heatmap(
                solutions[n]['loadings'], factor_labels,
                f"{output_dir}/heatmap_{n}_factors.png",
                threshold=0.30
            )

    # ------------------------------------------------------------------
    # STEP 5: Save All Solution Files
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: SAVING ALL SOLUTION FILES")
    print("=" * 60)

    save_all_solutions(solutions, output_dir)

    # ------------------------------------------------------------------
    # STEP 6: Decision Guidance
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: DECISION GUIDANCE")
    print("=" * 60)

    print("""
Review the outputs and consider the following criteria when choosing
the final solution for CFA:

  1. THEORETICAL ALIGNMENT
     Does the solution reproduce the expected Competence / Autonomy /
     HH-Relatedness / HR-Relatedness structure?
     Check: do items load on their theoretically assigned factor?

  2. SIMPLE STRUCTURE
     Minimise cross-loading items (loading >= 0.30 on 2+ factors).
     Prefer the solution with fewest cross-loaders.

  3. COMMUNALITIES
     Mean communality should ideally exceed 0.30.
     Items with communality < 0.20 may need to be reviewed or removed.

  4. VARIANCE EXPLAINED
     More factors explain more variance; weigh the gain per added factor.
     A meaningful factor should add at least 5% explained variance.

  5. INTERPRETABILITY
     Each factor should contain at least 3 items with primary loadings
     >= 0.40, and the factor should be substantively interpretable.

  6. EMPIRICAL vs THEORETICAL TENSION
     PA and MAP suggest 2 factors; theory predicts 4.
     Examine whether the 4-factor solution achieves acceptable simple
     structure. If yes, theory-driven retention is defensible and
     standard in scale development (Fabrigar et al., 1999).
     Report all three solutions and justify the chosen one explicitly.
""")

    print(f"All EFA outputs saved to '{output_dir}/'")
    print("EFA multi-solution analysis complete.")
