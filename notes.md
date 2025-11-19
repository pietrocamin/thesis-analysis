
# Preliminary_EDA.py script breakdown

## What the script analyses
1. **Univariate Distributions**
    - Mean, SD, median, min, max for each item
    - Skewness and kurtosis
    - Response frequency distributions
    - Visual histograms for all items
2. **Item-Total Correlations**
    - Uncorrected: Correlation of each item with total scale score
    - Corrected: Item correlated with total minus that item (better estimate)
    - Factor-specific: Item-total within each factor
    - Interpretation: r ≥ 0.30 = good discrimination
3. **Floor/Ceiling Effects**
    - Identifies items where >15% respond at minimum or maximum
    - These items have restricted range and may not discriminate well
4. **Problematic Items**, flags items with:
    - Low discrimination (corrected r < 0.30)
    - Floor or ceiling effects (>15%)
    - Low variance (SD < 0.5)
    - Very high inter-item correlations (r > 0.85 = potential redundancy)

## Outputs to Review
1. **Console Report** is a comprehensive summary with:
    - Overall statistics
    - Items with floor/ceiling effects
    - Items with poor discrimination
    - Problematic items summary
2. **Visualizations**:
    - Item distribution plots (check for normality, skewness)
    - Item means with error bars (identify items that are too easy/hard)
    - Item-total correlations (discrimination quality)
    - Floor/ceiling effect comparison
    - Correlation heatmap (identify item clusters)
3. **CSV files**:
    - Detailed item statistics
    - Item-total correlations (overall and by factor)
    - Problematic items list

## What to Look For
- **Good items** should have:
    - Corrected item-total r ≥ 0.30 (ideally ≥ 0.40)
    - Reasonable variance (SD ≥ 0.8 for 5-point scale)
    - No severe floor/ceiling effects (<15% at extremes)
    - Moderate inter-item correlations (0.30-0.70)

- **Red flags**:
    - r < 0.30 = poor discrimination
    - r > 0.85 with another item = redundancy
        - 15% at floor/ceiling = restricted range
    - Very high/low means (outside 2-4 on 5-point scale)

## Next Steps After Analysis
Based on the results:
- Identify candidates for removal (poor discrimination, floor/ceiling)
- Group items by correlation patterns (may inform EFA)
- Check if problematic items align with your CFA issues
- Document decisions about keeping vs. removing items

———

# DB for testing
## bell-dataset.csv
Generation based on the following principles of realistic human response patterns:
    1. Central Tendency Bias: Most people avoid the extreme ends of a scale (1 and 7). Their responses will tend to cluster around the middle values (3, 4, 5), creating a more bell-shaped distribution for many items.
    2. Inattentive Responders ("Straight-lining"): A small percentage of respondents will get lazy or bored and click the same number all the way down (e.g., a row of all '4's). This is a common data quality issue.
    3. Acquiescence Bias ("Yea-saying"): Some individuals have a tendency to agree with statements regardless of content. These respondents will score moderately high on most items, weakening the expected negative correlation between opposing factors (e.g., between C and A).
    4. Slight Inconsistency: Even a conscientious respondent won't be perfectly consistent. While they may feel positively about a factor in general, one or two specific items might elicit a different response. This adds "error variance" to the model.
    5. Varied "True Scores": The underlying "trait level" of respondents will be drawn from a normal distribution, meaning most people will be in the middle, with fewer people at the very high or low ends of the constructs.
This new dataset will be a more rigorous and realistic test for your validation scripts.
The model poorly fits the CFA.

## good-dit-dataset.csv
Generation principles:
    1. Strong but Imperfect Internal Consistency: Items within a factor will be highly correlated (leading to good Cronbach's Alpha, e.g., > 0.80), but with slightly more noise than the "perfect" dataset. A respondent with a high "true score" on a factor will generally score high on its items, but might rate one or two items a bit lower.
    2. Minor Model Misspecification: This is the key. I will introduce very subtle relationships that the simple CFA model does not account for. For example, I might create a tiny correlation between the error terms of two specific items (e.g., C3 and HHR2). This is a realistic scenario where two items might share some unique variance outside of their parent factors (e.g., due to similar wording). The CFA model assumes these are uncorrelated, so this small violation will degrade the fit from "perfect" to "good" without breaking the model entirely.
    3. Realistic Score Distribution: The data will still exhibit central tendency bias, with fewer extreme scores (1s and 7s), making the overall distribution more plausible.
    4. No "Broken" Data: I will not include any straight-liners that would cause the model to fail to converge.
This dataset represents a successful but realistic questionnaire validation.