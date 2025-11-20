import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced visualizations
from scipy import stats  # For correlation calculations
from scipy.stats import skew, kurtosis  # For calculations
import os  # For directory operations
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings for cleaner output


# --- Likert EDA and Item Analysis Class ---
class LikertEDAItemAnalysis:
    """
    Comprehensive EDA and Item Analysis for Likert scale dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with respondents as rows and Likert items as columns
    item_cols : list
        List of column names containing Likert items
    factor_structure : dict, optional
        Dictionary mapping factor names to lists of item names
        e.g., {'Factor_C': ['C1', ...], 'Factor_A': ['A1', ...]}
    scale_range : tuple
        Min and max values of Likert scale (default: (1, 5))
    """
    

    # -- Initialization --
    def __init__(self, data, item_cols, factor_structure=None, scale_range=(1, 5)):
        self.data = data.copy() # Work on a copy to avoid modifying original data
        self.item_cols = item_cols
        self.factor_structure = factor_structure
        self.scale_range = scale_range
        self.n_items = len(item_cols)
        self.n_respondents = len(data)
        
        # Results storage
        self.item_stats = None
        self.item_total_corr = None
        self.corrected_item_total_corr = None


    # -- Univariate Statistics --    
    def univariate_statistics(self):
        """
        Calculate comprehensive univariate statistics for each item.
        
        Returns:
        --------
        pd.DataFrame : Item-level statistics
            Columns: Item, Countinng number of items, Mean, Standard deviation, Median,
                     Min, Max, Skewness, Kurtosis, Q1, Q3, IQR for each scale value.
        """
        stats_list = []
        
        for item in self.item_cols:
            item_data = self.data[item]
            
            stats_dict = {
                'Item': item,
                'N': item_data.count(),
                'Mean': item_data.mean(),
                'SD': item_data.std(),
                'Median': item_data.median(),
                'Min': item_data.min(),
                'Max': item_data.max(),
                'Skewness': skew(item_data),
                'Kurtosis': kurtosis(item_data),
                'Q1': item_data.quantile(0.25),
                'Q3': item_data.quantile(0.75),
                'IQR': item_data.quantile(0.75) - item_data.quantile(0.25)
            }
            
            # Response distribution (frequencies)
            for value in range(self.scale_range[0], self.scale_range[1] + 1):
                freq = (item_data == value).sum()
                pct = 100 * freq / len(item_data)
                stats_dict[f'Freq_{value}'] = freq
                stats_dict[f'Pct_{value}'] = pct
            
            stats_list.append(stats_dict)
        
        self.item_stats = pd.DataFrame(stats_list)
        return self.item_stats # Return the DataFrame of item statistics


    # -- Floor and Ceiling Effects --
    def floor_ceiling_effects(self, threshold=0.15):
        """
        Identify items with floor or ceiling effects.
        Floor effect occurs when many data points cluster at the minimum possible value, 
            and is defined as >threshold% responses at minimum value.
        Ceiling effect occurs when many data points cluster at the maximum possible value, 
            and is defined as >threshold% responses at maximum value.
        These effects matter because they can limit the ability to detect differences or changes.

        Why they matter:
        • Reduce variability in your data (i.e., lower standard deviation)
        • Distort correlations and statistical tests
        • Hide true differences between groups
        • Limit ability to detect effects

        Possible solutions:
        • Use wider/more sensitive measurement scales (e.g., 7-point Likert)
        • Adjust difficulty of questions
        • Use statistical methods designed for bounded data
        • Collect more data
        
        Parameters:
        -----------
        threshold : float
            Proportion threshold for flagging (default: 0.15 = 15%)
        
        Returns:
        --------
        pd.DataFrame : Items with floor/ceiling effects
        """
        if self.item_stats is None:
            self.univariate_statistics()
        
        floor_ceiling = []
        
        for _, row in self.item_stats.iterrows():
            item = row['Item']
            
            # Floor effect: >threshold% at minimum value
            floor_pct = row[f'Pct_{self.scale_range[0]}'] / 100
            # Ceiling effect: >threshold% at maximum value
            ceiling_pct = row[f'Pct_{self.scale_range[1]}'] / 100
            
            has_floor = floor_pct > threshold
            has_ceiling = ceiling_pct > threshold
            
            if has_floor or has_ceiling:
                floor_ceiling.append({
                    'Item': item,
                    'Floor_Effect': has_floor,
                    'Floor_Pct': floor_pct * 100,
                    'Ceiling_Effect': has_ceiling,
                    'Ceiling_Pct': ceiling_pct * 100,
                    'Issue': 'Floor' if has_floor else 'Ceiling'
                })
        
        floor_ceiling_df = pd.DataFrame(floor_ceiling)
        return floor_ceiling_df


    # -- Item-Total Correlations --
    def item_total_correlations(self, method='spearman'):
        """
        Calculate item-total correlations (uncorrected and corrected).
        • Uncorrected item-total correlation: correlation between item and total score.
        • Corrected item-total correlation: correlation between item and total score excluding that item.

        Correlation methods supported: 'pearson', 'spearman' (default), or 'both'.
        • Pearson correlation measures linear relationships, assumes continuous data.
        • Spearman correlation is rank-based, more appropriate for ordinal Likert data.
        • 'both' calculates both methods for comparison and sensitivity analysis.
        
        METHODOLOGICAL NOTE:
        For ordinal Likert scale data, Spearman is theoretically more appropriate as it:
        (1) respects the ordinal nature of the data
        (2) makes no assumptions about interval equality between scale points
        (3) is more robust to non-normality and outliers
        
        However, with 5+ categories and reasonable distributions, Pearson and Spearman
        typically yield similar results. Using 'both' allows comparison to verify robustness.
        
        Parameters:
        -----------
        method : str
            Correlation method: 'pearson', 'spearman' (default), or 'both'
        
        Returns:
        --------
        pd.DataFrame : Item-total correlation statistics
        """
        # Calculate total score
        total_score = self.data[self.item_cols].sum(axis=1)
        
        correlations = []
        
        for item in self.item_cols:
            # Corrected item-total correlation (remove item from total)
            total_without_item = total_score - self.data[item]
            
            item_dict = {'Item': item}
            
            if method in ['pearson', 'both']:
                # Pearson correlations
                r_uncorr_p, p_uncorr_p = stats.pearsonr(self.data[item], total_score)
                r_corr_p, p_corr_p = stats.pearsonr(self.data[item], total_without_item)
            
                if method == 'pearson':
                    item_dict.update({
                        'Item_Total_r': r_uncorr_p,
                        'Item_Total_p': p_uncorr_p,
                        'Corrected_Item_Total_r': r_corr_p,
                        'Corrected_Item_Total_p': p_corr_p,
                        'Method': 'Pearson'
                    })
                else:  # both
                    item_dict.update({
                        'Pearson_Item_Total_r': r_uncorr_p,
                        'Pearson_Corrected_r': r_corr_p,
                        'Pearson_p': p_corr_p
                    })
            
            if method in ['spearman', 'both']:
                # Spearman correlations
                r_uncorr_s, p_uncorr_s = stats.spearmanr(self.data[item], total_score)
                r_corr_s, p_corr_s = stats.spearmanr(self.data[item], total_without_item)
                
                if method == 'spearman':
                    item_dict.update({
                        'Item_Total_r': r_uncorr_s,
                        'Item_Total_p': p_uncorr_s,
                        'Corrected_Item_Total_r': r_corr_s,
                        'Corrected_Item_Total_p': p_corr_s,
                        'Method': 'Spearman'
                    })
                else:  # both
                    item_dict.update({
                        'Spearman_Item_Total_r': r_uncorr_s,
                        'Spearman_Corrected_r': r_corr_s,
                        'Spearman_p': p_corr_s
                    })
            
            # Add comparison if both methods used
            if method == 'both':
                item_dict['Difference'] = abs(r_corr_p - r_corr_s)
                item_dict['Agreement'] = 'High' if abs(r_corr_p - r_corr_s) < 0.05 else 'Moderate' if abs(r_corr_p - r_corr_s) < 0.10 else 'Low'
                # Use Spearman for discrimination judgment (more conservative for ordinal data)
                item_dict['Discrimination'] = 'Good' if r_corr_s >= 0.30 else 'Poor'
            else:
                item_dict['Discrimination'] = 'Good' if item_dict['Corrected_Item_Total_r'] >= 0.30 else 'Poor'
            
            correlations.append(item_dict)
        
        self.item_total_corr = pd.DataFrame(correlations)
        
        # Print summary if both methods used
        if method == 'both':
            print("\n" + "=" * 60)
            print("CORRELATION METHOD COMPARISON (Pearson vs Spearman)")
            print("=" * 60)
            mean_diff = self.item_total_corr['Difference'].mean()
            max_diff = self.item_total_corr['Difference'].max()
            high_agree = (self.item_total_corr['Agreement'] == 'High').sum()
            
            print(f"\nMean absolute difference: {mean_diff:.4f}")
            print(f"Maximum difference: {max_diff:.4f}")
            print(f"Items with high agreement (|diff| < 0.05): {high_agree}/{len(correlations)}")
            
            if mean_diff < 0.05:
                print("\n✓ Methods show high agreement. Either method is appropriate.")
                print("  Treating Likert data as quasi-continuous appears justified.")
            elif mean_diff < 0.10:
                print("\n⚠ Methods show moderate agreement. Consider using Spearman.")
                print("  Some ordinal/non-linear patterns may be present.")
            else:
                print("\n✗ Methods show substantial disagreement. Use Spearman.")
                print("  Data violates assumptions for Pearson correlation.")
            
            # Show items with largest disagreement
            largest_diff = self.item_total_corr.nlargest(3, 'Difference')
            if largest_diff['Difference'].iloc[0] > 0.10:
                print(f"\nItems with largest method disagreement:")
                print(largest_diff[['Item', 'Pearson_Corrected_r', 'Spearman_Corrected_r', 'Difference']].to_string(index=False))
        
        return self.item_total_corr


    # -- Factor-level Item-Total Correlations --
    def item_total_by_factor(self, method='spearman'):
        """
        Calculate item-total correlations within each factor.
        
        Parameters:
        -----------
        method : str
            Correlation method: 'pearson', 'spearman' (default), or 'both'
        
        Returns:
        --------
        pd.DataFrame : Factor-specific item-total correlations
        """
        if self.factor_structure is None:
            print("No factor structure provided. Skipping factor-level analysis.")
            return None
        
        factor_correlations = []
        
        for factor_name, items in self.factor_structure.items():
            # Calculate factor total score
            factor_total = self.data[items].sum(axis=1)
            
            for item in items:
                # Corrected item-total (within factor)
                total_without_item = factor_total - self.data[item]
                
                item_dict = {'Factor': factor_name, 'Item': item}
                
                if method in ['pearson', 'both']:
                    r_p, p_p = stats.pearsonr(self.data[item], total_without_item)
                    if method == 'pearson':
                        item_dict.update({
                            'Factor_Item_Total_r': r_p,
                            'Factor_Item_Total_p': p_p,
                            'Method': 'Pearson'
                        })
                    else:
                        item_dict.update({
                            'Pearson_r': r_p,
                            'Pearson_p': p_p
                        })
                
                if method in ['spearman', 'both']:
                    r_s, p_s = stats.spearmanr(self.data[item], total_without_item)
                    if method == 'spearman':
                        item_dict.update({
                            'Factor_Item_Total_r': r_s,
                            'Factor_Item_Total_p': p_s,
                            'Method': 'Spearman'
                        })
                    else:
                        item_dict.update({
                            'Spearman_r': r_s,
                            'Spearman_p': p_s
                        })
                
                if method == 'both':
                    item_dict['Difference'] = abs(r_p - r_s)
                    item_dict['Discrimination'] = 'Good' if r_s >= 0.30 else 'Poor'
                else:
                    item_dict['Discrimination'] = 'Good' if item_dict['Factor_Item_Total_r'] >= 0.30 else 'Poor'
                
                factor_correlations.append(item_dict)
        
        return pd.DataFrame(factor_correlations)


    # -- Inter-Item Correlation Matrix --
    def inter_item_correlation_matrix(self):
        """
        Calculate inter-item correlation matrix.
        Inter-item correlations help identify redundancy or poor items.
        If many items correlate very highly (e.g., r > 0.85), they may be redundant.
        If items show very low correlations (e.g., r < 0.20), they may not measure the same construct.
        The expected range for well-designed Likert items is typically 0.30 to 0.70.
        
        Returns:
        --------
        pd.DataFrame : Correlation matrix
        """
        corr_matrix = self.data[self.item_cols].corr()
        return corr_matrix


    # -- Identify Problematic Items --
    def identify_problematic_items(self, min_item_total_r=0.30, max_floor_ceiling=0.15):
        """
        Identify problematic items based on discrimination, floor/ceiling effects, and variance.
        Troublesome items may:
        • Show low discrimination (corrected item-total r < min_item_total_r)
        • Exhibit floor or ceiling effects (> max_floor_ceiling proportion)
        • Have very low variance (SD < 0.5)
        
        Parameters:
        -----------
        min_item_total_r : float
            Minimum acceptable corrected item-total correlation (default: 0.30)
        max_floor_ceiling : float
            Maximum acceptable floor/ceiling proportion (default: 0.15)
        
        Returns:
        --------
        pd.DataFrame : Summary of problematic items
        """
        if self.item_stats is None:
            self.univariate_statistics()
        if self.item_total_corr is None:
            self.item_total_correlations()
        
        problematic = []
        
        for item in self.item_cols:
            issues = []
            
            # Check discrimination (item-total correlation)
            item_corr = self.item_total_corr[self.item_total_corr['Item'] == item]
            corrected_r = item_corr['Corrected_Item_Total_r'].values[0]
            
            if corrected_r < min_item_total_r:
                issues.append(f'Low discrimination (r={corrected_r:.3f})')
            
            # Check floor/ceiling effects
            item_stat = self.item_stats[self.item_stats['Item'] == item]
            floor_pct = item_stat[f'Pct_{self.scale_range[0]}'].values[0] / 100
            ceiling_pct = item_stat[f'Pct_{self.scale_range[1]}'].values[0] / 100
            
            if floor_pct > max_floor_ceiling:
                issues.append(f'Floor effect ({floor_pct*100:.1f}%)')
            if ceiling_pct > max_floor_ceiling:
                issues.append(f'Ceiling effect ({ceiling_pct*100:.1f}%)')
            
            # Check variance
            sd = item_stat['SD'].values[0]
            if sd < 0.5:
                issues.append(f'Low variance (SD={sd:.3f})')
            
            if issues:
                problematic.append({
                    'Item': item,
                    'Issues': '; '.join(issues),
                    'Corrected_r': corrected_r,
                    'Mean': item_stat['Mean'].values[0],
                    'SD': sd
                })
        
        return pd.DataFrame(problematic)


    # -- Plotting Functions --
    def plot_univariate_distributions(self, ncols=4, figsize=(16, 12)):
        """
        Plot distribution for each item.

        Parameters:
        -----------
        ncols : int
            Number of columns in the subplot grid (default: 4)
        figsize : tuple
            Figure size (default: (16, 12))
        """
        nrows = int(np.ceil(self.n_items / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if self.n_items > 1 else [axes]
        
        for idx, item in enumerate(self.item_cols):
            ax = axes[idx]
            
            # Count frequencies
            counts = self.data[item].value_counts().sort_index()
            
            # Bar plot
            ax.bar(counts.index, counts.values, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Likert answer')
            ax.set_ylabel('Frequency')
            ax.set_title(f'$\\mathbf{{{item}}}$\nM={self.data[item].mean():.2f}, SD={self.data[item].std():.2f}')
            ax.set_xticks(range(self.scale_range[0], self.scale_range[1] + 1))
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(self.n_items, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig


    # -- Plot Item Statistics --
    def plot_item_statistics(self, figsize=(18, 16)):
        """
        Create comprehensive visualization of item statistics.

        Parameters:
        -----------
        figsize : tuple
            Figure size (default: (16, 10))
        """
        if self.item_stats is None:
            self.univariate_statistics()
        if self.item_total_corr is None:
            self.item_total_correlations()
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Mean and SD by item
        ax1 = fig.add_subplot(gs[0, :])
        x = range(len(self.item_cols))
        ax1.errorbar(x, self.item_stats['Mean'], yerr=self.item_stats['SD'], 
                     fmt='o', capsize=5, capthick=2)
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.item_stats['Item'], rotation=45, ha='right')
        ax1.set_ylabel('Mean ± SD')
        ax1.set_title('Item Means and Standard Deviations')
        ax1.axhline(y=np.mean(self.item_stats['Mean']), color='r', 
                    linestyle='--', label='Overall Mean')
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend()
        
        # 2. Item-total correlations
        ax2 = fig.add_subplot(gs[1, 0])
        colors = ['green' if r >= 0.30 else 'red' 
                  for r in self.item_total_corr['Corrected_Item_Total_r']]
        ax2.barh(range(len(self.item_cols)), 
                 self.item_total_corr['Corrected_Item_Total_r'],
                 color=colors, alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(self.item_cols)))
        ax2.set_yticklabels(self.item_total_corr['Item'])
        ax2.set_xlabel('Corrected Item-Total Correlation')
        ax2.set_title('Item Discrimination')
        ax2.axvline(x=0.30, color='black', linestyle='--', label='Threshold (0.30)')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Skewness
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.barh(range(len(self.item_cols)), self.item_stats['Skewness'],
                 alpha=0.7, edgecolor='black')
        ax3.set_yticks(range(len(self.item_cols)))
        ax3.set_yticklabels(self.item_stats['Item'])
        ax3.set_xlabel('Skewness')
        ax3.set_title('Item Skewness')
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax3.axvline(x=-1, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(x=1, color='red', linestyle='--', alpha=0.5)
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Floor/Ceiling effects
        ax4 = fig.add_subplot(gs[2, 0])
        floor_pcts = self.item_stats[f'Pct_{self.scale_range[0]}'].values
        ceiling_pcts = self.item_stats[f'Pct_{self.scale_range[1]}'].values
        
        x_pos = np.arange(len(self.item_cols))
        width = 0.35
        
        ax4.bar(x_pos - width/2, floor_pcts, width, label='Floor (%)', 
                alpha=0.7, edgecolor='black')
        ax4.bar(x_pos + width/2, ceiling_pcts, width, label='Ceiling (%)', 
                alpha=0.7, edgecolor='black')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(self.item_stats['Item'], rotation=45, ha='right')
        ax4.set_ylabel('Percentage')
        ax4.set_title('Floor and Ceiling Effects')
        ax4.axhline(y=15, color='red', linestyle='--', label='15% Threshold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Distribution of means
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(self.item_stats['Mean'], bins=20, edgecolor='black', alpha=0.7)
        ax5.set_xlabel('Item Mean')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Distribution of Item Means')
        ax5.axvline(x=np.mean(self.item_stats['Mean']), color='red', 
                   linestyle='--', label=f'Mean={np.mean(self.item_stats["Mean"]):.2f}')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        return fig


    # -- Plot Correlation Heatmap --
    def plot_correlation_heatmap(self, figsize=(12, 10), annot=False):
        """
        Plot inter-item correlation heatmap.

        Parameters:
        -----------
        figsize : tuple
            Figure size (default: (12, 10))
        annot : bool
            Whether to annotate cells with correlation values (default: False)
        """
        corr_matrix = self.inter_item_correlation_matrix()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Inter-Item Correlation Matrix', fontsize=14, pad=20)
        plt.tight_layout()
        
        return fig


    # -- Generate Comprehensive Report --
    def generate_report(self):
        """
        Generate comprehensive text report.
        """
        print("=" * 60)
        print("LIKERT SCALE EDA & ITEM ANALYSIS REPORT")
        print("=" * 60)
        print(f"\nSample Size: {self.n_respondents}")
        print(f"Number of Items: {self.n_items}")
        print(f"Scale Range: {self.scale_range[0]} to {self.scale_range[1]}")
        
        # Univariate statistics
        if self.item_stats is None:
            self.univariate_statistics()
        
        print("\n" + "=" * 60)
        print("UNIVARIATE STATISTICS SUMMARY")
        print("=" * 60)
        print(f"\nOverall Mean: {self.item_stats['Mean'].mean():.3f} (SD={self.item_stats['Mean'].std():.3f})")
        print(f"Mean Range: {self.item_stats['Mean'].min():.3f} to {self.item_stats['Mean'].max():.3f}")
        print(f"Average SD: {self.item_stats['SD'].mean():.3f}")
        print(f"SD Range: {self.item_stats['SD'].min():.3f} to {self.item_stats['SD'].max():.3f}")
        
        # Floor/ceiling effects
        floor_ceiling_df = self.floor_ceiling_effects()
        print("\n" + "=" * 60)
        print("FLOOR AND CEILING EFFECTS")
        print("=" * 60)
        if len(floor_ceiling_df) > 0:
            print(f"\n{len(floor_ceiling_df)} items with floor/ceiling effects (>15%):")
            print(floor_ceiling_df.to_string(index=False))
        else:
            print("\nNo significant floor or ceiling effects detected.")
        
        # Item-total correlations
        if self.item_total_corr is None:
            self.item_total_correlations()
        
        print("\n" + "=" * 60)
        print("ITEM DISCRIMINATION (CORRECTED ITEM-TOTAL CORRELATIONS)")
        print("=" * 60)
        print(f"\nMean corrected r: {self.item_total_corr['Corrected_Item_Total_r'].mean():.3f}")
        print(f"Range: {self.item_total_corr['Corrected_Item_Total_r'].min():.3f} to {self.item_total_corr['Corrected_Item_Total_r'].max():.3f}")
        
        poor_disc = self.item_total_corr[self.item_total_corr['Corrected_Item_Total_r'] < 0.30]
        if len(poor_disc) > 0:
            print(f"\n{len(poor_disc)} items with poor discrimination (r < 0.30):")
            print(poor_disc[['Item', 'Corrected_Item_Total_r']].to_string(index=False))
        else:
            print("\nAll items show adequate discrimination.")
        
        # Problematic items
        problematic = self.identify_problematic_items()
        print("\n" + "=" * 60)
        print("PROBLEMATIC ITEMS SUMMARY")
        print("=" * 60)
        if len(problematic) > 0:
            print(f"\n{len(problematic)} items flagged with issues:")
            print(problematic.to_string(index=False))
        else:
            print("\nNo problematic items identified.")
        
        # Inter-item correlations
        corr_matrix = self.inter_item_correlation_matrix()
        # Get lower triangle (excluding diagonal)
        lower_tri = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool))
        correlations = lower_tri.stack()
        
        print("\n" + "=" * 60)
        print("INTER-ITEM CORRELATIONS")
        print("=" * 60)
        print(f"\nMean inter-item correlation: {correlations.mean():.3f}")
        print(f"Range: {correlations.min():.3f} to {correlations.max():.3f}")
        
        # Flag very high correlations (potential redundancy)
        high_corr = correlations[correlations > 0.85]
        if len(high_corr) > 0:
            print(f"\n{len(high_corr)} item pairs with very high correlations (r > 0.85):")
            for (item1, item2), r in high_corr.items():
                print(f"  {item1} - {item2}: r = {r:.3f}")
        
        print("\n" + "=" * 60)
        print("END OF REPORT")
        print("=" * 60)



# --- Main execution ---
if __name__ == "__main__":
    # - Initialize class in an object -
    # Load dataset
    file_name = 'DATASETS/reversed_DATASET.csv'
    df = pd.read_csv(file_name)
    
    # Define factor structure
    factor_structure = {
        'Competence': [f'C{i}' for i in range(1, 8)],
        'Autonomy': [f'A{i}' for i in range(1, 8)],
        'HH-Relatedness': [f'HHR{i}' for i in range(1, 5)],
        'HR-Relatedness': [f'HRR{i}' for i in range(1, 5)]
    }
    
    # Initialize analyzer
    item_cols = [f'C{i}' for i in range(1, 8)] + \
                [f'A{i}' for i in range(1, 8)] + \
                [f'HHR{i}' for i in range(1, 5)] + \
                [f'HRR{i}' for i in range(1, 5)]
    
    analyzer = LikertEDAItemAnalysis(df, item_cols, factor_structure, scale_range=(1, 5))
    

    # - Generate comprehensive report -
    analyzer.generate_report()
    

    # - Create visualizations -
    print("\nGenerating plots...")
    
    # Univariate distributions
    fig_univariate = analyzer.plot_univariate_distributions()
    plt.savefig('images/preliminary-EDA/item_distributions.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    # Item statistics
    fig_statistics = analyzer.plot_item_statistics()
    plt.savefig('images/preliminary-EDA/item_statistics.png', dpi=300, bbox_inches='tight', pad_inches=0.05)

    # Correlation heatmap
    fig_correlation = analyzer.plot_correlation_heatmap()
    plt.savefig('images/preliminary-EDA/correlation_heatmap.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    plt.show()
    

    # - Export detailed statistics -
    # Chech and create output directory if it doesn't exist
    os.makedirs('output/preliminary-EDA', exist_ok=True)

    statistics = analyzer.univariate_statistics()
    statistics.to_csv('output/preliminary-EDA/item_statistics.csv', index=False)
    
    item_total = analyzer.item_total_correlations()
    item_total.to_csv('output/preliminary-EDA/item_total_correlations.csv', index=False)
    
    factor_item_total = analyzer.item_total_by_factor()
    if factor_item_total is not None:
        factor_item_total.to_csv('output/preliminary-EDA/factor_item_total_correlations.csv', index=False)
    
    problematic = analyzer.identify_problematic_items()
    if len(problematic) > 0:
        problematic.to_csv('output/preliminary-EDA/problematic_items.csv', index=False)
    
    print("\nAnalysis complete! Check output files and plots.")