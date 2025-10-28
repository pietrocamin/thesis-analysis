import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# –-- LEGENDA ---
# C = Competence
# A = Autonomy
# HHR = Human-Human Relatendess
# HRR = Human-Robot Relatendess

def analyze_likert_data(file_path):
    """
    Loads, analyzes, and visualizes the Likert scale dataset.
    """
    try:
        # --- 1. Load the Dataset ---
        df = pd.read_csv(file_path)
        print("\n" + "--- 1. Data loaded successfully ---")
        print("First 5 entries of the dataset:")
        print(df.head())
        print("\n" + "="*50 + "\n")


        # --- 2. Get Items Descriptive Statistics ---
        # This provides a quick overview of the data (mean, std, min, max, etc.) for each item.
        print("--- 2. Descriptive statistics for all items ---")
        print(df.describe().T) # Using .T to transpose the output for better readability
        print("\n" + "="*50 + "\n")


        # --- 3. Calculate Mean Scores for Each Factor ---
        # This is a common step in survey analysis: combining items to get a score for a construct.
        print("--- 3. Calculating mean scores for each factor ---")

        # Define the columns for each factor
        c_cols = [f'C{i}' for i in range(1, 8)]      # C1 to C7
        a_cols = [f'A{i}' for i in range(1, 8)]      # A1 to A7
        hhr_cols = [f'HHR{i}' for i in range(1, 5)]  # HHR1 to HHR4
        hrr_cols = [f'HRR{i}' for i in range(1, 5)]  # HRR1 to HRR4

        # Calculate the mean for each factor for each respondent (row-wise mean)
        df['C_Mean'] = df[c_cols].mean(axis=1)
        df['A_Mean'] = df[a_cols].mean(axis=1)
        df['HHR_Mean'] = df[hhr_cols].mean(axis=1)
        df['HRR_Mean'] = df[hrr_cols].mean(axis=1)

        print("Dataset with new factor mean scores (first 5 entries):")
        print(df[['Respondent_ID', 'C_Mean', 'A_Mean', 'HHR_Mean', 'HRR_Mean']].head())
        print("\n" + "="*50 + "\n")


        # --- 4. Get statistics for factor scores ---
        print("--- 4. Descriptive statistics for factor mean scores ---")
        print(df[['C_Mean', 'A_Mean', 'HHR_Mean', 'HRR_Mean']].describe().T)
        print("\n" + "="*50 + "\n")
        

        # --- 5. Visualize the Data ---
        print("--- 5. Generating visualizations ---")

        # Set the plot style using seaborn
        sns.set_theme(style="whitegrid")

        # Create a figure with subplots for each factor's distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Distribution of Mean Scores for Each Factor', fontsize=16)

        # Plot for Competence (Factor C)
        sns.histplot(df['C_Mean'], kde=True, ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Competence')
        axes[0, 0].set_xlim(1, 6)

        # Plot for Autonomy (Factor A)
        sns.histplot(df['A_Mean'], kde=True, ax=axes[0, 1], color='olive')
        axes[0, 1].set_title('Autonomy')
        axes[0, 1].set_xlim(1, 6)

        # Plot for Human-Human Relatendess (Factor HHR)
        sns.histplot(df['HHR_Mean'], kde=True, ax=axes[1, 0], color='gold')
        axes[1, 0].set_title('HH Relatendess')
        axes[1, 0].set_xlim(1, 6)

        # Plot for Human-Robot Relatendess (Factor HRR)
        sns.histplot(df['HRR_Mean'], kde=True, ax=axes[1, 1], color='teal')
        axes[1, 1].set_title('HR Relatendess')
        axes[1, 1].set_xlim(1, 6)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        print("Displaying plot. Close the plot window to end the script.")
        plt.show()


    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure 'test-dataset.csv' is in the same directory as the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Main execution ---
if __name__ == "__main__":
    file_name = 'converted-dataset-reversed.csv'
    analyze_likert_data(file_name)