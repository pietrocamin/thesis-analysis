## To run files
1. Open the folder with the terminal
2. Run the scripts with the command: *pixi run python <file_name>*


## File descriptions

### Script
- **reverse_items.py**: creates a new dataset with the values of the negative items inverted.
- **inattentive_responses_analysis.py**: analysis for the identification of careless respondents.
- **outliers_analysis**: analysis for the identification of outliers.
- **analyze_data.py**: calculates descriptive statistics, such as item means and standard deviations, for each item and factor.
- **preliminary_EDA**: exploratory data analysis aiminng at identifying the 
- **model_identification.py**: calculates the Spearman inter-factor correlations to identify the structure of the model (i.e., how each factor correlates with each other), dropping problematic items.
- **drop_problematic_items**: saves a dataset without the identified problematic items.
- **cfa_analysis.py**: runs the CFA.

### Datasets
- **RAW_DATASET**: dataset cleaned of "totally agree" item check.
- **DATASET.csv**: dataset cleaned of inattentive responses.
- **reversed_DATASET.csv**: dataset cleaned of inattentive responses, with the values of negative items reverted.
- **DATASET_cement_clean.csv**: dataset cleaned of "eat cement" item check and inattentive responses.
- **reversed_DATASET_cement_clean.csv**: dataset cleaned of "eat cement" item check and inattentive responses, with the values of negative items reverted.
