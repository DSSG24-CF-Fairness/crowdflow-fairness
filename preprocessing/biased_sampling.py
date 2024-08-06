import pandas as pd
import numpy as np
import os

def merge_data(features, demographics, flow, demographic_column_name='svi'):
    """
    Merges feature, demographic, and flow data into a single DataFrame.

    Parameters:
    - features: DataFrame containing 'total_population' and 'geoid'
    - demographics: DataFrame with 'geoid' and other demographic columns
    - flow: DataFrame containing 'geoid_o', 'geoid_d', and 'pop_flows'
    - demographic_column_name: The demographic column to consider from the demographics DataFrame (default 'svi')

    Returns:
    - DataFrame with columns ['geoid_o', 'geoid_d', 'demographic_o', 'demographic_d', 'population_o', 'population_d', 'pop_flows']
    """

    # Merge to get demographic and population for origin
    origin_merge = pd.merge(flow, demographics[['geoid', demographic_column_name]], how='left', left_on='geoid_o', right_on='geoid')
    origin_merge.rename(columns={demographic_column_name: 'demographic_o'}, inplace=True)
    origin_merge = pd.merge(origin_merge, features[['geoid', 'total_population']], how='left', left_on='geoid_o', right_on='geoid')
    origin_merge.rename(columns={'total_population': 'population_o'}, inplace=True)

    # Merge to get demographic and population for destination
    destination_merge = pd.merge(origin_merge, demographics[['geoid', demographic_column_name]], how='left', left_on='geoid_d', right_on='geoid')
    destination_merge.rename(columns={demographic_column_name: 'demographic_d'}, inplace=True)
    destination_merge = pd.merge(destination_merge, features[['geoid', 'total_population']], how='left', left_on='geoid_d', right_on='geoid')
    destination_merge.rename(columns={'total_population': 'population_d'}, inplace=True)

    # Select and return the necessary columns
    final_df = destination_merge[['geoid_o', 'geoid_d', 'demographic_o', 'demographic_d', 'population_o', 'population_d', 'pop_flows']]
    final_df = final_df.dropna()

    return final_df


def calculate_biased_flow(features, demographics, flow, demographic_column_name='svi', method=1, order="ascending", sampling=False, experiment_id="0", bias_factor=0.5):
    """
    Adjusts flow data based on demographic biases.

    Parameters:
    - features: DataFrame containing 'total_population' and 'geoid'.
    - demographics: DataFrame with 'geoid' and other demographic columns.
    - flow: DataFrame containing 'geoid_o', 'geoid_d', and 'pop_flows'.
    - demographic_column_name: demographic column to consider (default 'svi').
    - method: 1 for delta between origin and destination, 2 for destination only (default 1).
    - order: "ascending" or "descending" for sorting demographics (default "ascending").
    - sampling: True for probabilistic sampling, False for deterministic calculation (default False).
    - experiment_id: ID for saving the output (default "0").
    - bias_factor: Factor to adjust bias (default 0.5).

    Returns:
    - None. Saves the biased flow DataFrame to a specified path.
    """
    # Call the merge_data function to merge all required data
    result_df = merge_data(features, demographics, flow, demographic_column_name)

    # Create the demographic delta if method == 1
    if method == 1:
        result_df['delta_demographic'] = result_df['demographic_d'] - result_df['demographic_o']
        bias_column = 'delta_demographic'
    elif method == 2:
        bias_column = 'demographic_d'
    else:
        raise ValueError('Invalid method. Method should be 1 (for delta) or 2 (for destination only).')

    # Calculate the adjustment factors for each geoid_o
    grouped = result_df.groupby('geoid_o')
    biased_flows = []

    for name, group in grouped:
        group = group.copy()
        total_outflow = group['pop_flows'].sum()
        
        # Sort based on the demographic value in chosen order
        group = group.sort_values(by=bias_column, ascending=(order == "ascending"))
        
        # Calculate cumulative population and percentiles
        group['cumulative'] = group['population_d'].cumsum() - 0.5 * group['population_d']
        group['percentile'] = group['cumulative'] / group['population_d'].sum()

        # Calculate adjusted flows
        group['adjustment_factor'] = group['pop_flows']*(group['percentile'] + bias_factor)
        group['adjustment_factor'] = group['adjustment_factor']/ group['adjustment_factor'].sum()

        if sampling == False:
            group['adjusted_flows'] = group['adjustment_factor'] * total_outflow
        else: 
            np.random.choice(group['geoid_d'], size=int(total_outflow), p=group['adjustment_factor'])

        biased_flows.append(group)
    
    biased_flows_df = pd.concat(biased_flows)
    final_flows_df = biased_flows_df[['geoid_o', 'geoid_d', 'adjusted_flows']].copy()
    final_flows_df.rename(columns={'adjusted_flows':'pop_flows'}, inplace= True)

    # Choose file name based on sampling
    file_suffix = 'sampled_flows' if sampling else 'biased_flows'

    # Construct the save path
    save_path = f"../processed_data/{experiment_id}/train/{demographic_column_name}/{method}_{order}_{file_suffix}.csv"

    # Create directories and save the result
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_flows_df.to_csv(save_path, index=False)
    print(f"Saved adjusted flows to {save_path}")
