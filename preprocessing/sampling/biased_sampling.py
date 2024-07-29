import pandas as pd

def calculate_biased_flow(df_origin):
    """
    Calculate biased flow and B_Flow for each origin with rounding.

    Parameters:
    df_origin (pd.DataFrame): A DataFrame containing data for a specific origin with the following columns:
                              - 'Destination': Identifier for the destination.
                              - 'Origin': Identifier for the origin.
                              - 'Income': Income value for the destination.
                              - 'Population': Population of the destination.
                              - 'Flow': Original flow value from the origin to the destination.

    Returns:
    pd.DataFrame: The input DataFrame with additional columns:
                  - 'Income Rank': Rank of the income for each destination within the origin.
                  - 'Cumulative Population': Cumulative population of destinations with lower income.
                  - 'Percentile': Percentile based on the cumulative population.
                  - 'Biased Flow': Biased flow value calculated based on the given formula.
                  - 'B_Flow': Rounded biased flow value, adjusted to ensure the total outflow remains the same.
    """
    # Step 1: Create Income Rank
    df_origin['Income Rank'] = df_origin['Income'].rank(ascending=False, method='min')

    # Step 2: Create Cumulative Population
    df_origin = df_origin.sort_values('Income', ascending=True)
    df_origin['Cumulative Population'] = df_origin['Population'].cumsum() - df_origin['Population']

    # Step 3: Create Percentile
    total_population = df_origin['Population'].sum()
    df_origin['Percentile'] = (df_origin['Cumulative Population'] + 0.5 * df_origin['Population']) / total_population * 100
    df_origin['Percentile'] = df_origin['Percentile'].apply(lambda x: f'{x:.2f}%')

    # Step 4: Calculate Biased Flow
    df_origin['Percentile Value'] = df_origin['Percentile'].str.rstrip('%').astype(float) / 100
    df_origin['Numerator'] = (df_origin['Percentile Value'] + 0.5) * df_origin['Flow']
    denominator = df_origin['Numerator'].sum()
    df_origin['Biased Flow'] = df_origin['Numerator'] / denominator

    # Step 5: Calculate B_Flow
    total_outflow = df_origin['Flow'].sum()
    df_origin['B_Flow'] = df_origin['Biased Flow'] * total_outflow

    # Step 6: Round B_Flow to integers
    df_origin['B_Flow'] = df_origin['B_Flow'].round()

    # Adjust B_Flow to ensure total outflow remains the same
    diff = total_outflow - df_origin['B_Flow'].sum()
    if diff != 0:
        adjustment = df_origin['B_Flow'].idxmax() if diff > 0 else df_origin['B_Flow'].idxmin()
        df_origin.at[adjustment, 'B_Flow'] += diff

    # Drop intermediate columns
    df_origin.drop(columns=['Percentile Value', 'Numerator'], inplace=True)

    return df_origin

# Sample DataFrame creation with multiple origins
data = {
    'Destination': ['53033000100', '53033000101', '53033000102', '53033000103', '53033000104'],
    'Origin': ['53033000100', '53033000100', '53033000101', '53033000102', '53033000102'],
    'Income': [150, 30, 40, 80, 70],
    'Population': [20, 100, 50, 70, 60],
    'Flow': [10, 5, 3, 7, 5]
}

df = pd.DataFrame(data)

# Apply the function to each origin separately
df = df.groupby('Origin', group_keys=False).apply(calculate_biased_flow).reset_index(drop=True)

# Display the updated DataFrame

print(df)
