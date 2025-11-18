import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

def validate_questionnaire(file_path):
    """
    Performs a full validation analysis on the Likert scale data,
    assessing dimensionality, reliability, and construct validity.
    """
    try:
        # --- 1. Load and Prepare Data ---
        df = pd.read_csv(file_path)
        # Select only the 22 item columns for factor and reliability analysis
        item_cols = df.columns[1:] 
        df_items = df[item_cols]

        print("="*60)
        print("          QUESTIONNAIRE VALIDATION ANALYSIS")
        print("="*60)


        # --- 2. Dimensionality Analysis (Exploratory Factor Analysis) ---
        print("\n--- Section 1: Dimensionality (Exploratory Factor Analysis) ---\n")

        # 2a. Pre-check: KMO and Bartlett's Test
        kmo_all, kmo_model = calculate_kmo(df_items)
        bartlett_stat, bartlett_p_value = calculate_bartlett_sphericity(df_items)

        print(f"Kaiser-Meyer-Olkin (KMO) Test: {kmo_model:.3f}")
        if kmo_model < 0.6:
            print("WARNING: KMO value is below the acceptable threshold of 0.6. Factor analysis may not be suitable.")
        else:
            print("KMO value is adequate (> 0.6).")

        print(f"\nBartlett's Test of Sphericity: p-value = {bartlett_p_value:.3e}")
        if bartlett_p_value > 0.05:
            print("WARNING: Bartlett's test is not significant (p > 0.05). Correlations may be too low for EFA.")
        else:
            print("Bartlett's test is significant (p < 0.05), indicating that EFA is appropriate.")
        
        # 2b. Perform EFA
        print("\nPerforming EFA with 4 factors...")
        fa = FactorAnalyzer(n_factors=4, rotation='varimax', method='minres')
        fa.fit(df_items)

        # 2c. Get and display the factor loadings
        loadings = pd.DataFrame(fa.loadings_, index=df_items.columns, columns=[f'Factor {i+1}' for i in range(4)])
        print("\nFactor Loadings (Varimax Rotation):")
        print("An item's highest loading should be on its intended factor (ideally > 0.4)")
        print(loadings.style.apply(lambda x: ['background: lightgreen' if v > 0.4 else '' for v in x], axis=1))
        
        # 2d. Scree Plot to visualize factor importance
        ev, v = fa.get_eigenvalues()
        plt.figure(figsize=(8, 6))
        plt.scatter(range(1, df_items.shape[1] + 1), ev)
        plt.plot(range(1, df_items.shape[1] + 1), ev)
        plt.title('Scree Plot')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalue')
        plt.grid()
        plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (Eigenvalue=1)')
        plt.legend()
        print("\nDisplaying Scree Plot. Close the plot to continue...")
        plt.show()


        # --- 3. Reliability Analysis (Internal Consistency) ---
        print("\n--- Section 2: Reliability (Cronbach's Alpha) ---\n")
        print("Calculating Cronbach's Alpha for each intended factor (should be > 0.7).")

        # Define the columns for each factor based on the original structure
        factor_items = {
            'C': [f'C{i}' for i in range(1, 8)],
            'A': [f'A{i}' for i in range(1, 8)],
            'HHR': [f'HHR{i}' for i in range(1, 5)],
            'HRR': [f'HRR{i}' for i in range(1, 5)]
        }
        
        # correlation matrix for every item to see if correlate by factor
        
        for factor, cols in factor_items.items():
            alpha = pg.cronbach_alpha(data=df[cols])
            print(f"  - Factor '{factor}': Cronbach's Alpha = {alpha[0]:.3f}")


        # --- 4. Construct Validity Analysis (Correlations) ---
        print("\n--- Section 3: Construct Validity (Factor Correlation Matrix) ---\n")

        # Calculate mean scores for each factor
        df['C_Mean'] = df[factor_items['C']].mean(axis=1)
        df['A_Mean'] = df[factor_items['A']].mean(axis=1)
        df['HHR_Mean'] = df[factor_items['HHR']].mean(axis=1)
        df['HRR_Mean'] = df[factor_items['HRR']].mean(axis=1)
        
        factor_means_df = df[['C_Mean', 'A_Mean', 'HHR_Mean', 'HRR_Mean']]
        
        # Calculate and visualize the correlation matrix
        corr_matrix = factor_means_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Factor Mean Scores')
        plt.tight_layout()
        print("Displaying Factor Correlation Heatmap. Close the plot to end the script.")
        print("Check for expected patterns of convergence (related factors) and divergence (unrelated factors).")
        plt.show()

        print("\n" + "="*70)
        print("                  Validation Analysis Complete")
        print("="*70)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Main execution ---
if __name__ == "__main__":
    file_name = 'converted-dataset-reversed.csv'
    validate_questionnaire(file_name)