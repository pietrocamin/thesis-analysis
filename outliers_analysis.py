import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis # for Mahalanobis distance calculation
from scipy.linalg import inv # for matrix inversion
from scipy.stats import chi2 # for chi-squared distribution
import io
import os

# -- Save IQR outliers boxplot --
def outliers_IQR_boxplots(file_path):
    # Read the dataset into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Create output directory for box plots if it doesn't exist
    os.makedirs('images/boxplot-outliers', exist_ok=True)

    for column in df.columns:
        if df[column].dtype in ['int64']: # Check for integer columns, just to be sure
            # Interquartile Range (IQR) method to detect outliers
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
            
            # Print outliers
            print(f"Outliers in {column} (IQR Method):")
            if not outliers.empty:
                print(outliers)
            else:
                print("No outliers found.")
            print("" + "-"*30)

            # Visualise outliers with box plots
            plt.figure(figsize=(4, 6))
            sns.boxplot(y=df[column])
            plt.title(f'IQR Outlier Box Plot for {column}')
            plt.ylabel('') # blank y-axis label
            plt.yticks([1, 2, 3, 4, 5]) # set specific y-axis ticks
            
            # Add outlier information as text below the plot
            if not outliers.empty:
                # Get value counts of outliers
                outlier_counts = outliers.value_counts().sort_index()
                outlier_text = f"Outliers (value: count) — tot: {outliers.size}\n"

                # Special formatting for column "C6" given its wider range of outlier values
                if column == "C6":
                    outlier_text += "     ".join([f"• {value}: {count}" for value, count in outlier_counts.items()])
                else:
                    outlier_text += "\n".join([f"• {value}: {count}" for value, count in outlier_counts.items()])
            else:
                outlier_text = "No outliers found.\n"
                
            # Add text below the plot
            plt.figtext(0.15, 0, outlier_text, 
                        fontsize=10,
                        ha='left',
                        va='bottom')
            
            # Save the box plot image with extra padding at bottom
            plt.savefig(f"images/boxplot-outliers/boxplot_outliers_{column}.png", 
                       dpi=200, # image resolution value
                       bbox_inches='tight',
                       pad_inches=0.05) # small padding to avoid cutting off text
            plt.close() # close the figure to free memory

    ''' # Dispaly all box plots in a single figure
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Check if there are numeric columns to plot
    if not num_cols:
        print("No numeric columns to plot.")
        return

    # Grid size: roughly square
    n = len(num_cols)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).reshape(-1) # flatten (works if axes is single or grid)

    for i, col in enumerate(num_cols):
        ax = axes[i]
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(col)
        ax.set_ylabel('')

    # Turn off unused axes
    for j in range(i + 1, axes.size):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

    # Save the graphs figure
    fig.savefig("boxplots.png", dpi=150)
    '''


# -- Save CSV with IQR outliers --
def outliers_IQR_CSV(file_path):
    # Read the dataset into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Create a dictionary to store outliers
    outliers_dict = {}

    for column in df.columns:
        if df[column].dtype in ['int64']: # Check for integer columns
            # Interquartile Range (IQR) method to detect outliers
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers and their corresponding Respondent_IDs
            outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            outliers = df[outlier_mask][['Respondent_ID', column]]
            
            # Store both Respondent_ID and value
            if not outliers.empty:
                outliers_dict[f"{column}_ID"] = outliers['Respondent_ID'].tolist()
                outliers_dict[column] = outliers[column].tolist()
            else:
                outliers_dict[f"{column}_ID"] = []
                outliers_dict[column] = []

    # Find the maximum length among all lists in the dictionary
    max_len = 0
    for value in outliers_dict.values():
        max_len = max(max_len, len(value))

    # Pad shorter lists with None values
    for key in outliers_dict:
        current_len = len(outliers_dict[key])
        if current_len < max_len:
            outliers_dict[key].extend([None] * (max_len - current_len))

    # Convert outliers dictionary to DataFrame
    outliers_df = pd.DataFrame.from_dict(outliers_dict)

    # Convert value columns to integers
    for col in outliers_df.columns:
        if not col.endswith('_ID'): # do not convert ID columns
            outliers_df[col] = outliers_df[col].astype('Int64')

    # Reorder columns to alternate ID and value
    cols = []
    for col in df.columns:
        if f"{col}_ID" in outliers_df.columns:
            cols.extend([f"{col}_ID", col])
    outliers_df = outliers_df[cols]

    # Remove rows that contain only None values
    outliers_df = outliers_df.dropna(how='all')
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save outliers to CSV
    outliers_df.to_csv('output/outliers_IQR.csv', index=False) # save without index
    print("Outliers have been saved to 'output/outliers_IQR.csv'")


# -- Save histograms/density plots for numeric columns --
def save_histograms(file_path):
    # Read the dataset into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Create output directory for histograms if it doesn't exist
    os.makedirs('images/histograms', exist_ok=True)

    for column in df.columns:
        if df[column].dtype in ['int64']:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[column], kde=True) # create histogram with density plot
            plt.title(f'Histogram/Density Plot for {column}')
            plt.xlabel(column)
            plt.xticks([1, 2, 3, 4, 5]) # set specific x-axis ticks
            plt.ylabel('Frequency')

            # Save the histogram image
            plt.savefig(f"images/histograms/histogram_{column}.png", 
                        dpi=200, 
                        bbox_inches='tight',
                        pad_inches=0.05) # small padding
            plt.close() # close the figure to free memory

    print("Histograms have been saved in the 'images/histograms' directory.")


# -- Identify outliers using Mahalanobis Distance --
def outliers_mahalanobis(file_path):
    # - Step 1: read the dataset into a pandas DataFrame -
    try:
        df = pd.read_csv(file_path)
        print("CSV loaded successfully. First 5 rows:")
        print(df.head())
        print("\n" + "="*40 + "\n")
    except FileNotFoundError:
        print(f"Error: '{file_path}.csv' not found. Please ensure the CSV file is in the correct directory.")
        exit()

    # - Step 2: identify the observed variables for your measurement model -
    measurement_vars = [
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
        'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
        'HHR1', 'HHR2', 'HHR3', 'HHR4',
        'HRR1', 'HRR2', 'HRR3', 'HRR4'
    ]

    # Ensure all specified measurement variables exist in the DataFrame
    missing_vars = [v for v in measurement_vars if v not in df.columns]
    if missing_vars:
        print(f"Error: The following measurement variables are missing from your CSV: {missing_vars}")
        print("Please check your CSV headers and the 'measurement_vars' list.")
        exit()

    X = df[measurement_vars].values

    # - Step 3: calculate the Mean Vector -
    mean_vector = np.mean(X, axis=0)
    print("Mean Vector (first 5 elements):")
    print(mean_vector[:5])
    print("\n" + "="*40 + "\n")

    # - Step 4: calculate the Covariance Matrix -
    covariance_matrix = np.cov(X, rowvar=False) # rowvar=False because rows are observations
    print("Covariance Matrix Shape:")
    print(covariance_matrix.shape)
    print("\n" + "="*40 + "\n")

    # - Step 5: calculate the Inverse Covariance Matrix -
    try:
        inv_covariance_matrix = inv(covariance_matrix)
        print("Inverse Covariance Matrix (first 3x3 block):")
        print(inv_covariance_matrix[:3, :3])
        print("\n" + "="*50 + "\n")
    except np.linalg.LinAlgError:
        print("Error: Covariance matrix is singular. Cannot compute inverse.")
        print("This might happen if you have perfectly correlated variables, insufficient data,")
        print("or variables with zero variance. Consider checking your data for such issues.")
        exit()

    # - Step 6: compute Mahalanobis Distance for each observation -
    mahalanobis_distances = []
    for i in range(len(X)):
        dist = mahalanobis(X[i], mean_vector, inv_covariance_matrix)
        mahalanobis_distances.append(dist)

    df['Mahalanobis_Distance'] = mahalanobis_distances

    print("Data Head with Mahalanobis Distances (first 5 rows):")
    print(df[['Respondent_ID'] + measurement_vars[:4] + ['Mahalanobis_Distance']].head())
    print("\n" + "="*40 + "\n")

    # - Step 7: determine threshold for outlier detection -
    num_variables = len(measurement_vars)
    alpha = 0.001 # common choice for outlier detection
    threshold = chi2.ppf(1 - alpha, num_variables)

    print(f"Number of variables used for Mahalanobis Distance: {num_variables}")
    print(f"Chi-squared threshold for alpha={alpha} (for {num_variables} degrees of freedom): {threshold:.2f}")

    # - Step 8: identify potential outliers -
    outliers = df[df['Mahalanobis_Distance'] > threshold].sort_values(by='Mahalanobis_Distance', ascending=False)

    print(f"\nPotential Outliers (Mahalanobis Distance > {threshold:.2f}):")
    if not outliers.empty:
        print(outliers[['Respondent_ID', 'Mahalanobis_Distance']].head(10)) # Show top 10 outliers
        print(f"\nTotal number of outliers found: {len(outliers)}")
    else:
        print("No outliers found at the specified threshold.")

    # - Step 9: visualize the Distribution of Mahalanobis Distances -
    plt.figure(figsize=(12, 7))
    sns.histplot(df['Mahalanobis_Distance'], bins=50, kde=True)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Outlier Threshold (alpha={alpha})')
    plt.title('Distribution of Mahalanobis Distances')
    plt.xlabel('Mahalanobis Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # - Save the results with Mahalanobis distances to a new CSV -
    # # Create output directory if it doesn't exist
    # os.makedirs('output', exist_ok=True)

    # # Save outliers to CSV
    # df.to_csv('output/dataset_with_mahalanobis.csv', index=False)
    # print("Outliers have been saved to 'output/dataset_with_mahalanobis.csv'")


# -- Identify outliers using Mahalanobis Distance for each factor separately --
def single_factor_mahalanobis(file_path):
    # --- Step 1: load your CSV dataset ---
    try:
        # Read the dataset into a pandas DataFrame
        df = pd.read_csv(file_path)
        print("CSV loaded successfully. First 5 rows:")
        print(df.head())
        print("\n" + "="*40 + "\n")
    except FileNotFoundError:
        print(f"Error: '{file_path}.csv' not found. Please ensure the CSV file is in the correct directory.")
        exit()

    # - Step 2: define the indicators for each factor, matching the CSV column headers -
    factor_indicators = {
        'C': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
        'A': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'],
        'HHR': ['HHR1', 'HHR2', 'HHR3', 'HHR4'],
        'HRR': ['HRR1', 'HRR2', 'HRR3', 'HRR4']
    }

    # Ensure all specified indicator variables exist in the DataFrame
    all_indicators = [item for sublist in factor_indicators.values() for item in sublist]
    missing_vars = [v for v in all_indicators if v not in df.columns]
    if missing_vars:
        print(f"Error: The following indicator variables are missing from your CSV: {missing_vars}")
        print("Please check your CSV headers and the 'factor_indicators' dictionary.")
        exit()

    # - Step 3: compute Mahalanobis Distance for each factor independently -
    outlier_summary = {} # to store outlier counts for each factor

    # Iterate over each factor and its indicators
    for factor_name, indicators in factor_indicators.items():
        print(f"--- Processing Factor: {factor_name}, with indicators: {indicators} ---")

        X_factor = df[indicators].values

        # Calculate Mean Vector
        mean_vector_factor = np.mean(X_factor, axis=0)

        # Calculate Covariance Matrix
        covariance_matrix_factor = np.cov(X_factor, rowvar=False)

        # Calculate Inverse Covariance Matrix
        try:
            inv_covariance_matrix_factor = inv(covariance_matrix_factor)
        except np.linalg.LinAlgError:
            print(f"Error: Covariance matrix for factor '{factor_name}' is singular.")
            print("This might be due to perfect correlations, zero variance, or insufficient data.")
            print("Skipping Mahalanobis calculation for this factor.")
            df[f'Mahalanobis_Distance_{factor_name}'] = np.nan # Add NaN column
            outlier_summary[factor_name] = {'count': 0, 'threshold': np.nan}
            continue # move to the next factor

        # Compute Mahalanobis Distance for each observation for this factor
        mahalanobis_distances_factor = []
        for i in range(len(X_factor)):
            dist = mahalanobis(X_factor[i], mean_vector_factor, inv_covariance_matrix_factor)
            mahalanobis_distances_factor.append(dist)

        df[f'Mahalanobis_Distance_{factor_name}'] = mahalanobis_distances_factor

        # Identify Outliers for this factor
        num_variables_factor = len(indicators)
        alpha = 0.05 # standard value for apha is 0.001, but using a higher value
        threshold_factor = chi2.ppf(1 - alpha, num_variables_factor)

        print(f"Number of variables for '{factor_name}': {num_variables_factor}")
        print(f"Chi-squared threshold for '{factor_name}' (alpha={alpha}): {threshold_factor:.2f}")

        outliers_factor = df[df[f'Mahalanobis_Distance_{factor_name}'] > threshold_factor]

        # Display potential outliers for this factor
        print(f"Potential Outliers for '{factor_name}' (Mahalanobis Distance > {threshold_factor:.2f}):")
        if not outliers_factor.empty:
            print(outliers_factor[['Respondent_ID', f'Mahalanobis_Distance_{factor_name}']].head())
            print(f"Total outliers for '{factor_name}': {len(outliers_factor)}")
        else:
            print(f"No outliers found for '{factor_name}' at the specified threshold.")

        outlier_summary[factor_name] = {'count': len(outliers_factor), 'threshold': threshold_factor}

        print("\n" + "="*40 + "\n")

    # --- Step 4: display final results and summary ---
    print("All Factors Processed. DataFrame head with new Mahalanobis Distance columns:")
    print(df[['Respondent_ID'] + [f'Mahalanobis_Distance_{f}' for f in factor_indicators.keys()]].head())

    print("\n--- Outlier Summary Per Factor ---")
    for factor, info in outlier_summary.items():
        print(f"Factor '{factor}': {info['count']} outliers (Threshold: {info['threshold']:.2f})")

    # --- Step 5: visualize Distributions for each factor ---
    fig, axes = plt.subplots(nrows=len(factor_indicators), ncols=1, figsize=(10, 5 * len(factor_indicators)))
    fig.suptitle('Distribution of Mahalanobis Distances per Factor', y=1.02)

    for i, (factor_name, indicators) in enumerate(factor_indicators.items()):
        ax = axes[i] if len(factor_indicators) > 1 else axes # Handle single subplot case
        if f'Mahalanobis_Distance_{factor_name}' in df.columns:
            sns.histplot(df[f'Mahalanobis_Distance_{factor_name}'], bins=50, kde=True, ax=ax)
            threshold = outlier_summary[factor_name]['threshold']
            if not np.isnan(threshold):
                ax.axvline(x=threshold, color='r', linestyle='--', label=f'Outlier Threshold (alpha={alpha})')
            ax.set_title(f'Factor: {factor_name}')
            ax.set_xlabel(f'Mahalanobis Distance for {factor_name}')
            ax.set_ylabel('Frequency')
            ax.legend()
        else:
            ax.set_title(f'Factor: {factor_name} (Skipped due to singular covariance matrix)')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.text(0.5, 0.5, "Calculation Skipped", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()

    # Optional: Save the results with all Mahalanobis distances to a new CSV
    # df.to_csv('your_dataset_with_factor_mahalanobis.csv', index=False)


# -- Handle outliers in the dataset --
# (Implementation can be added here as needed)


# --- Main execution ---
if __name__ == "__main__":
    file_name = 'DATASETS/reversed_DATASET.csv'
    # file_name = 'DATASETS/reversed_DATASET_cement_clean.csv' # for testing with dataset cleaned of "cement" item check
    
    # - IRQ -
    # outliers_IQR_boxplots(file_name) # save IQR outliers boxplots
    # outliers_IQR_CSV(file_name) # save IQR outliers CSV
    
    # - Histograms/Density Plots -
    # save_histograms(file_name) # save pictures of histograms/density plots per item
    
    # - Mahalanobis Distance -
    # outliers_mahalanobis(file_name) # identify outliers using Mahalanobis Distance
    # single_factor_mahalanobis(file_name) # identify outliers using Mahalanobis Distance for each factor of the model