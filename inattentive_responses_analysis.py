import pandas as pd
import io
import os


# -- Visualise outliers in the dataset for manual observation --
def detect_omogeneous_values(file_path):
    """
    Print Respondent_ID and factor name (C, A, HHR, HRR) when all items
    in that factor are the same for a respondent. Returns a DataFrame of matches.
    """
    # Read the dataset into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Define factor columns
    factors = {
        'C':  [f'C{i}' for i in range(1, 8)],
        'A':  [f'A{i}' for i in range(1, 8)],
        'HHR':[f'HHR{i}' for i in range(1, 5)],
        'HRR':[f'HRR{i}' for i in range(1, 5)],
    }

    results = []
    for _, row in df.iterrows():
        rID = row.get('Respondent_ID', None)
        for factorName, cols in factors.items():
            existing = [c for c in cols if c in df.columns]
            if len(existing) < 2:
                continue # need at least two items to compare
            vals = row[existing].dropna()
            if vals.nunique() == 1 and len(vals) > 0:
                # print respondent and factor
                print(f"{rID}: {factorName}")
                # capture optional value (try to cast to int if numeric)
                v = vals.iloc[0]
                try:
                    v = int(v)
                except Exception:
                    pass
                results.append({'Respondent_ID': rID, 'Factor': factorName, 'Value': v})

    # Print the number of times respondents answered a factor with the same values
    print(f"{len(results)} homogeneous factor instances found.")

    return pd.DataFrame(results)


# -- Find longest string of identical consecutive responses --
def longest_sequence_length(file_path, threshold=0):
        """
        Calculate longest string of identical consecutive responses in the DataFrame for each respondent.
        A threshold can be set to filter results.
        
        Parameters:
            file_path[str]: path to the CSV dataset
            threshold[int]: minimum consecutive identical responses to flag (default=10)
        
        Returns:
            List[Tuple[str, int]]: list of tuples with Respondent_ID and the length of the longest sequence
        """
        # Read the dataset into a pandas DataFrame
        df = pd.read_csv(file_path)
        results = [] # empty list to store results

        for index, row in df.iterrows():
            longest_length = 1
            current_length = 1

            # Get only the numeric columns (skip Respondent_ID) — using .iloc to access values by position
            numeric_values = row.iloc[1:].values

            # Iterate through the values in the row
            for i in range(1, len(numeric_values)):
                # Check if the current value is equal to the previous value
                if numeric_values[i] == numeric_values[i - 1]:
                    current_length += 1
                else:
                    # Update the longest length if the current sequence is longer
                    longest_length = max(longest_length, current_length)
                    current_length = 1 # reset current length for the new value

            # Final check to update the longest length at the end of the row
            longest_length = max(longest_length, current_length)

            # If the longest length is >= threshold, append the Respondent_ID and the longest length to the results
            if longest_length >= threshold:
                results.append((row['Respondent_ID'], longest_length))

        # Check if no results found after processing all rows
        if len(results) == 0:
            print("No respondents found with consecutive identical responses meeting (or exceeding) the threshold.")
        
        # Convert results to DataFrame
        dfResults = pd.DataFrame(results, columns=['Respondent_ID', f'Longest_Sequence_Length (threshold = {threshold})'])
        
        # - Save results to CSV -
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)

        # Save results to CSV
        dfResults.to_csv('output/longest_sequence_length.csv', index=False) # save without index
        print(f"Results above threshold of {threshold} have been saved to 'output/longest_sequence_length.csv'")

        return dfResults


# -- Compute Intra-individual Response Variability (IRV) --
def compute_IRV_and_filter_inattentive(file_path, lower_threshold=0.5, upper_threshold=1.8):
    """
    Compute the Intra-individual Response Variability (IRV) for each respondent
    and filter out those identified as potentially inattentive based on IRV thresholds.
    IRV is calculated as the standard deviation of responses across all items for each respondent.

    Parameters:
        file_path [str]: Path to the CSV dataset. Assumes the first column is 'Respondent_ID'
                         and subsequent columns are Likert response items.
        lower_threshold [float]: IRV values below this are considered too consistent
                                 (e.g., straight-lining) and flagged as inattentive.
        upper_threshold [float]: IRV values above this are considered too inconsistent
                                 (e.g., random responding) and flagged as inattentive.

    Returns:
        attentive_df [pandas.DataFrame]: A DataFrame containing only the 'attentive' respondents,
                                         with their calculated IRV included.
        inattentive_df [pandas.DataFrame]: A DataFrame containing the 'inattentive' respondents,
                                           with their calculated IRV included.
    """
    # # Read the dataset into a pandas DataFrame
    # df = pd.read_csv(file_path)

    # # Select the response columns
    # response_columns = df.columns[1:]

    # # Calculate IRV (which is the standard deviation) for each respondent
    # df['IRV'] = df[response_columns].std(axis=1)

    # # Filter results based on the specified bounds
    # filtered_results = df[(df['IRV'] >= lower_thrashold) & (df['IRV'] <= upper_thrashold)]

    # # Display the results
    # print(filtered_results[['Respondent_ID', 'IRV']])

    # Read the dataset into a pandas DataFrame, catching file not found error
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame(), pd.DataFrame() # return empty DataFrames

    response_columns = df.columns[1:] # select all columns with responses (assuming first column is Respondent_ID)

    # Calculate IRV (which is the standard deviation) for each respondent
    df['IRV'] = df[response_columns].std(axis=1)

    # Identify attentive respondents (IRV within the acceptable range)
    attentive_df = df[(df['IRV'] >= lower_threshold) &
                      (df['IRV'] <= upper_threshold)].copy() # .copy() to avoid SettingWithCopyWarning

    # Identify inattentive respondents (IRV outside the acceptable range)
    inattentive_df = df[(df['IRV'] < lower_threshold) |
                        (df['IRV'] > upper_threshold)].copy() # .copy() to avoid SettingWithCopyWarning

    print(f"\n--- IRV Analysis Summary ---")
    print(f"IRV thresholds for inattention: Below {lower_threshold} or Above {upper_threshold}")
    print(f"Total respondents: {len(df)}")
    print(f"Respondents identified as attentive: {len(attentive_df)}")
    print(f"Respondents identified as potentially inattentive: {len(inattentive_df)}")

    if not inattentive_df.empty:
        print("\nDetails of potentially inattentive respondents (ID and IRV):")
        print(inattentive_df[['Respondent_ID', 'IRV']].to_string(index=False)) # print without index
    else:
        print("\nNo respondents identified as potentially inattentive based on these thresholds.")

    return attentive_df, inattentive_df



# --- Main execution ---
if __name__ == "__main__":
    file_name = 'DATASETS/RAW_DATASET.csv'
    
    # - Visualise homogeneous values for "hand" detection in the dataset before negative-item manipulation -
    detect_omogeneous_values(file_name) # visualise constant factors on dataset before addressing negative items
    longest_sequence_length(file_name, 5) # to filter, add [int] parameter; 0 is default (no filter)
    compute_IRV_and_filter_inattentive(file_name, 0.8, 1.7) # visualise constant factors on dataset before addressing negative items

