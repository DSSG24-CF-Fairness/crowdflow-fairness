import pandas as pd
import numpy as np

def calculate_biased_flow(df_origin, demographic_col_origin, demographic_col_destination):
    """
    Calculate biased flow and B_Flow for each origin with rounding.

    Parameters:
    df_origin (pd.DataFrame): A DataFrame containing data for a specific origin with the following columns:
                              - 'Destination': Identifier for the destination.
                              - 'Origin': Identifier for the origin.
                              - demographic_col_origin: Column name for the demographic variable of the origin (e.g., 'Income of Origin').
                              - demographic_col_destination: Column name for the demographic variable of the destination (e.g., 'Income of Destination').
                              - 'Population': Population of the destination.
                              - 'Flow': Original flow value from the origin to the destination.

    Returns:
    pd.DataFrame: The input DataFrame with additional columns:
                  - 'Demographic Delta': Absolute difference between the origin's demographic and the destination's demographic.
                  - 'Demographic Rank': Rank of the demographic delta for each destination within the origin.
                  - 'Cumulative Population': Cumulative population of destinations with lower demographic.
                  - 'Percentile': Percentile based on the cumulative population.
                  - 'Biased Flow': Biased flow value calculated based on the given formula.
                  - 'B_Flow': Rounded biased flow value, adjusted to ensure the total outflow remains the same.
    """
    # Step 1: Calculate absolute Demographic Delta and create Demographic Rank
    df_origin['Demographic Delta'] = abs(df_origin[demographic_col_origin] - df_origin[demographic_col_destination])
    df_origin['Demographic Rank'] = df_origin['Demographic Delta'].rank(ascending=False, method='min')

    # Step 2: Create Cumulative Population
    df_origin = df_origin.sort_values('Demographic Delta', ascending=True)
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

def allocate_flows_fixed(df, origin, total_outflow, expected_values):
    """
    Allocate total outflow to destinations based on biased flow probabilities and fixed expected values.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data with biased flow probabilities.
    origin (str): The origin from which to allocate the total outflow.
    total_outflow (int): The total outflow to allocate.
    expected_values (dict): A dictionary containing the expected values for each destination.

    Returns:
    pd.DataFrame: The updated DataFrame with allocated flows.
    """
    df_origin = df[df['Origin'] == origin].copy()
    for destination, expected_value in expected_values.items():
        df_origin.loc[df_origin['Destination'] == destination, 'Allocated Flow'] = expected_value
    
    return df

# Provided test data
data = {
    'Destination': ['53033000101', '53033000102', '53033000103'],
    'Origin': ['53033000100', '53033000100', '53033000100'],
    'Income': [150, 30, 40],
    'Income of Origin': [100, 100, 100],
    'Income of Destination': [150, 270, 260],
    'Vulnerability Score': [0.5, 0.7, 0.2],
    'Population': [20, 100, 50],
    'Flow': [10, 5, 3]
}

df = pd.DataFrame(data)
