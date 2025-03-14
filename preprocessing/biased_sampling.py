import random
import pandas as pd
import numpy as np
import os



def merge_data(features_df, demographics_df, flow_df, demographic_column_name='svi'):
    """
    Merges features, demographics, and flow data into a single DataFrame.

    Parameters:
    - features_df (pd.DataFrame): DataFrame containing columns 'total_population' and 'geoid'.
    - demographics_df (pd.DataFrame): DataFrame containing 'geoid' and additional demographic columns.
    - flow_df (pd.DataFrame): DataFrame containing 'origin', 'destination', and 'flow' columns.
    - demographic_column_name (str): The demographic column to consider from the demographics DataFrame. Default is 'svi'.

    Returns:
    - A DataFrame with columns ['origin', 'destination', 'demographic_o', 'demographic_d', 'population_o', 'population_d', 'flow'].

    Notes:
    - If any data is missing after merging, a diagnostic CSV file is saved to a specified directory, and missing rows are removed from the final DataFrame.
    """

    # Merge to get demographic and population for origin
    origin_merge = pd.merge(flow_df, demographics_df[['geoid', demographic_column_name]], how='left', left_on='origin', right_on='geoid')
    origin_merge.rename(columns={demographic_column_name: 'demographic_o'}, inplace=True)
    origin_merge = pd.merge(origin_merge, features_df[['geoid', 'total_population']], how='left', left_on='origin', right_on='geoid')
    origin_merge.rename(columns={'total_population': 'population_o'}, inplace=True)
    origin_merge = origin_merge[['origin', 'destination', 'demographic_o', 'population_o', 'flow']]

    # Merge to get demographic and population for destination
    destination_merge = pd.merge(origin_merge, demographics_df[['geoid', demographic_column_name]], how='left', left_on='destination', right_on='geoid')
    destination_merge.rename(columns={demographic_column_name: 'demographic_d'}, inplace=True)
    destination_merge = pd.merge(destination_merge, features_df[['geoid', 'total_population']], how='left', left_on='destination', right_on='geoid')
    destination_merge.rename(columns={'total_population': 'population_d'}, inplace=True)

    # Select and return the necessary columns
    final_df = destination_merge[['origin', 'destination', 'demographic_o', 'demographic_d', 'population_o', 'population_d', 'flow']]

    # Drop rows with flow value of NaN and 0
    final_df = final_df.dropna()
    final_df = final_df.loc[final_df['flow'] > 0]

    return final_df


def calculate_biased_flow(features_df,
                          demographics_df,
                          flow_df,
                          folder_name,
                          demographic_column_name,
                          method=1,
                          order='ascending',
                          steepness_factor=5,
                          sampling=True,
                          random_seed=1
                          ):
    """
    Adjusts flow data to account for demographic biases using a sigmoid-based adjustment factor.

    Parameters:
    - features_df (pd.DataFrame): DataFrame containing columns 'total_population' and 'geoid'.
    - demographics_df (pd.DataFrame): DataFrame containing 'geoid' and additional demographic columns.
    - flow_df (pd.DataFrame): DataFrame containing 'origin', 'destination', and 'flow' columns.
    - demographic_column_name (str): The demographic column to consider from the demographics DataFrame. Default is 'svi'.
    - method (int): Specifies how bias is calculated; 1 for delta between origin and destination, 2 for destination only. Default is 1.
    - order (str): Sort order for demographic values, either 'ascending' or 'descending'. Default is 'ascending'.
    - steepness (float): Steepness factor to adjust the curve's steepness. Default is 5.

    Returns:
    - None. Saves the adjusted flow DataFrame to a specified file path.
    """

    # Call the merge_data function to merge all required data
    result_df = merge_data(features_df, demographics_df, flow_df, demographic_column_name)

    # Create the demographic delta if method == 1
    if method == 1:
        result_df['delta_demographic'] = abs(result_df['demographic_d'] - result_df['demographic_o'])
        bias_column = 'delta_demographic'
    elif method == 2:
        bias_column = 'demographic_d'
    else:
        raise ValueError('Invalid method. Method should be 1 (for delta) or 2 (for destination only).')

    # Calculate the adjustment factors for each origin
    grouped = result_df.groupby('origin')
    biased_flows = []

    # Sigmoid function parameters
    x = np.linspace(0.01, 0.99, 500)  # x from 0 to 1
    sigmoid_values = 2 / (1 + np.exp(-steepness_factor * (x - 0.5)))

    for name, group in grouped:
        group = group.copy()
        total_outflow = group['flow'].sum()

        # Sort based on the demographic value in chosen order
        group = group.sort_values(by=bias_column, ascending=(order == 'ascending'))

        # Calculate cumulative population and percentiles
        group['cumulative'] = group['population_d'].cumsum() - 0.5 * group['population_d']
        group['percentile'] = group['cumulative'] / group['population_d'].sum()

        # Map percentiles to sigmoid values for the adjustment factor
        group['adjustment_factor'] = np.interp(group['percentile'], x, sigmoid_values)


        group['new_flows'] = group['adjustment_factor'] * group['flow']
        total_new_flows = group['new_flows'].sum()
        group['weight'] = group['new_flows'] / total_new_flows

        if sampling is True:
            np.random.seed(random_seed)
            random.seed(random_seed)
            sampled_destinations = np.random.choice(group['destination'], size=int(total_outflow), p=group['weight'], replace=True)
            group['adjusted_flows'] = pd.Series(sampled_destinations).value_counts().reindex(group['destination']).fillna(0).values
        else:
            group['adjusted_flows'] = group['weight'] * total_outflow

        biased_flows.append(group)

    print(f'Random seed: {random_seed}.')

    biased_flows_df = pd.concat(biased_flows)
    final_flows_df = biased_flows_df[['origin', 'destination', 'adjusted_flows']].copy()
    final_flows_df.rename(columns={'adjusted_flows': 'flow'}, inplace=True)

    # Drop rows with flow value of NaN and 0
    final_flows_df = final_flows_df[(final_flows_df['flow'].notna())]
    final_flows_df = final_flows_df.loc[final_flows_df['flow'] > 0]

    # Choose file name based on sampling
    if sampling is True:
        file_suffix = f'biased_flow_{random_seed}_steep_{steepness_factor}'
    else:
        file_suffix = f'biased_flow_no_sampling_steep_{steepness_factor}'

    # Construct the save path
    save_path = f'../processed_data/{folder_name}/train/{demographic_column_name}_{method}_{order}_{file_suffix}.csv'

    # Create directories and save the result
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_flows_df.to_csv(save_path, index=False)
    print(f'Saved adjusted flows to {save_path}.')
