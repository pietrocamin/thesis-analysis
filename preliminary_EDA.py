import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


# --- Likert EDA and Item Analysis Class ---
class LikertEDAItemAnalysis:
    """
    Comprehensive EDA and Item Analysis for Likert scale data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with respondents as rows and Likert items as columns
    item_cols : list
        List of column names containing Likert items
    factor_structure : dict, optional
        Dictionary mapping factor names to lists of item names
        e.g., {'Factor_C': ['C1', 'C2', ...], 'Factor_A': ['A1', ...]}
    scale_range : tuple
        Min and max values of Likert scale (default: (1, 5))
    """
    

    # -- Initialization --
    def __init__(self, data, item_cols, factor_structure=None, scale_range=(1, 5)):
        self.data = data.copy()
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
        return self.item_stats


    # -- Floor and Ceiling Effects --
    def floor_ceiling_effects(self, threshold=0.15):
        """
        Identify items with floor or ceiling effects.
        
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
    def item_total_correlations(self, method='pearson'):
        """
        Calculate item-total correlations (uncorrected and corrected).
        
        Parameters:
        -----------
        method : str
            Correlation method: 'pearson' or 'spearman' (default: 'pearson')
        
        Returns:
        --------
        pd.DataFrame : Item-total correlation statistics
        """
        # Calculate total score
        total_score = self.data[self.item_cols].sum(axis=1)
        
        correlations = []
        
        for item in self.item_cols:
            # Uncorrected item-total correlation
            if method == 'pearson':
                r_uncorrected, p_uncorrected = stats.pearsonr(self.data[item], total_score)
            else:
                r_uncorrected, p_uncorrected = stats.spearmanr(self.data[item], total_score)
            
            # Corrected item-total correlation (remove item from total)
            total_without_item = total_score - self.data[item]
            if method == 'pearson':
                r_corrected, p_corrected = stats.pearsonr(self.data[item], total_without_item)
            else:
                r_corrected, p_corrected = stats.spearmanr(self.data[item], total_without_item)
            
            correlations.append({
                'Item': item,
                'Item_Total_r': r_uncorrected,
                'Item_Total_p': p_uncorrected,
                'Corrected_Item_Total_r': r_corrected,
                'Corrected_Item_Total_p': p_corrected,
                'Discrimination': 'Good' if r_corrected >= 0.30 else 'Poor'
            })
        
        self.item_total_corr = pd.DataFrame(correlations)
        return self.item_total_corr


    # -- Factor-level Item-Total Correlations --
    def item_total_by_factor(self, method='pearson'):
        """
        Calculate item-total correlations within each factor.
        
        Parameters:
        -----------
        method : str
            Correlation method: 'pearson' or 'spearman'
        
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
                
                if method == 'pearson':
                    r, p = stats.pearsonr(self.data[item], total_without_item)
                else:
                    r, p = stats.spearmanr(self.data[item], total_without_item)
                
                factor_correlations.append({
                    'Factor': factor_name,
                    'Item': item,
                    'Factor_Item_Total_r': r,
                    'Factor_Item_Total_p': p,
                    'Discrimination': 'Good' if r >= 0.30 else 'Poor'
                })
        
        return pd.DataFrame(factor_correlations)


    # -- Inter-Item Correlation Matrix --
    def inter_item_correlation_matrix(self):
        """
        Calculate inter-item correlation matrix.
        
        Returns:
        --------
        pd.DataFrame : Correlation matrix
        """
        corr_matrix = self.data[self.item_cols].corr()
        return corr_matrix


    # -- Identify Problematic Items --
    def identify_problematic_items(self, min_item_total_r=0.30, max_floor_ceiling=0.15):
        """
        Identify problematic items based on multiple criteria.
        
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
            ax.set_xlabel('Response')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{item}\nM={self.data[item].mean():.2f}, SD={self.data[item].std():.2f}')
            ax.set_xticks(range(self.scale_range[0], self.scale_range[1] + 1))
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(self.n_items, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig


    # -- Plot Item Statistics --
    def plot_item_statistics(self, figsize=(16, 10)):
        """
        Create comprehensive visualization of item statistics.
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
        print("=" * 80)
        print("LIKERT SCALE EDA & ITEM ANALYSIS REPORT")
        print("=" * 80)
        print(f"\nSample Size: {self.n_respondents}")
        print(f"Number of Items: {self.n_items}")
        print(f"Scale Range: {self.scale_range[0]} to {self.scale_range[1]}")
        
        # Univariate statistics
        if self.item_stats is None:
            self.univariate_statistics()
        
        print("\n" + "=" * 80)
        print("UNIVARIATE STATISTICS SUMMARY")
        print("=" * 80)
        print(f"\nOverall Mean: {self.item_stats['Mean'].mean():.3f} (SD={self.item_stats['Mean'].std():.3f})")
        print(f"Mean Range: {self.item_stats['Mean'].min():.3f} to {self.item_stats['Mean'].max():.3f}")
        print(f"Average SD: {self.item_stats['SD'].mean():.3f}")
        print(f"SD Range: {self.item_stats['SD'].min():.3f} to {self.item_stats['SD'].max():.3f}")
        
        # Floor/ceiling effects
        floor_ceiling_df = self.floor_ceiling_effects()
        print("\n" + "=" * 80)
        print("FLOOR AND CEILING EFFECTS")
        print("=" * 80)
        if len(floor_ceiling_df) > 0:
            print(f"\n{len(floor_ceiling_df)} items with floor/ceiling effects (>15%):")
            print(floor_ceiling_df.to_string(index=False))
        else:
            print("\nNo significant floor or ceiling effects detected.")
        
        # Item-total correlations
        if self.item_total_corr is None:
            self.item_total_correlations()
        
        print("\n" + "=" * 80)
        print("ITEM DISCRIMINATION (CORRECTED ITEM-TOTAL CORRELATIONS)")
        print("=" * 80)
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
        print("\n" + "=" * 80)
        print("PROBLEMATIC ITEMS SUMMARY")
        print("=" * 80)
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
        
        print("\n" + "=" * 80)
        print("INTER-ITEM CORRELATIONS")
        print("=" * 80)
        print(f"\nMean inter-item correlation: {correlations.mean():.3f}")
        print(f"Range: {correlations.min():.3f} to {correlations.max():.3f}")
        
        # Flag very high correlations (potential redundancy)
        high_corr = correlations[correlations > 0.85]
        if len(high_corr) > 0:
            print(f"\n{len(high_corr)} item pairs with very high correlations (r > 0.85):")
            for (item1, item2), r in high_corr.items():
                print(f"  {item1} - {item2}: r = {r:.3f}")
        
        print("\n" + "=" * 80)
        print("END OF REPORT")
        print("=" * 80)



# --- Main execution ---
if __name__ == "__main__":
    # - Load dataset -
    file_name = 'DATASETS/reversed_DATASET.csv'
    df = pd.read_csv(file_name)
    
    # - Define factor structure -
    factor_structure = {
        'C': [f'C{i}' for i in range(1, 8)],
        'A': [f'A{i}' for i in range(1, 8)],
        'HHR': [f'HHR{i}' for i in range(1, 5)],
        'HRR': [f'HRR{i}' for i in range(1, 5)]
    }
    
    # - Initialize analyzer -
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
    fig1 = analyzer.plot_univariate_distributions()
    plt.savefig('item_distributions.png', dpi=300, bbox_inches='tight')
    
    # Item statistics
    fig2 = analyzer.plot_item_statistics()
    plt.savefig('item_statistics.png', dpi=300, bbox_inches='tight')
    
    # Correlation heatmap
    fig3 = analyzer.plot_correlation_heatmap()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    

    # - Export detailed statistics -
    stats = analyzer.univariate_statistics()
    stats.to_csv('item_statistics.csv', index=False)
    
    item_total = analyzer.item_total_correlations()
    item_total.to_csv('item_total_correlations.csv', index=False)
    
    factor_item_total = analyzer.item_total_by_factor()
    if factor_item_total is not None:
        factor_item_total.to_csv('factor_item_total_correlations.csv', index=False)
    
    problematic = analyzer.identify_problematic_items()
    if len(problematic) > 0:
        problematic.to_csv('problematic_items.csv', index=False)
    
    print("\nAnalysis complete! Check output files and plots.")
