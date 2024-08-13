import pandas as pd
import numpy as np
import os
import geopandas as gpd


def generate_random_results(test_flow_path, tessellation_path, num_samples, experiment_id='0'):
    # Load the CSV file
    data = pd.read_csv(test_flow_path)

    # Load the tessellation.geojson file and extract the GEOIDs
    tessellation = gpd.read_file(tessellation_path)
    geoids = tessellation['GEOID'].unique()
    geoids = geoids.astype(int)

    # Create a DataFrame with all possible origin-destination pairs
    all_pairs = pd.DataFrame([(o, d) for o in geoids for d in geoids], columns=['origin', 'destination'])

    # Initialize a directory to save the output files
    save_path = f'../outputs/{experiment_id}/synthetic_data_random'
    os.makedirs(save_path, exist_ok=True)

    # Generate random data samples
    for sample_number in range(1, num_samples + 1):
        # Merge and preprocess data
        sample_data = pd.merge(all_pairs, data, on=['origin', 'destination'], how='left')
        sample_data['flow'] = sample_data['flow'].fillna(0)
        total_flows = sample_data.groupby('origin')['flow'].sum().reset_index()
        sample_data = sample_data.merge(total_flows, on='origin', suffixes=('', '_total'))

        # Generate random flows
        random_data = []
        for origin, group in sample_data.groupby('origin'):
            total_flow = group['flow_total'].iloc[0]
            random_floats = np.random.random(size=len(group))
            initial_flows = np.floor((random_floats / random_floats.sum()) * total_flow)
            remaining_flow = int(total_flow - initial_flows.sum())

            if remaining_flow > 0:
                add_indices = np.random.choice(range(len(group)), size=remaining_flow, replace=False)
                initial_flows[add_indices] += 1

            group['flow'] = initial_flows
            random_data.append(group.drop(columns=['flow_total']))

        # Combine and clean data
        random_data = pd.concat(random_data, ignore_index=True)
        random_data = random_data.dropna()
        random_data = random_data[random_data['flow'] > 0]

        # Save each sample
        output_path = os.path.join(save_path, f'synthetic_data_random_{sample_number}.csv')
        random_data.to_csv(output_path, index=False)

    return f"{num_samples} random samples have been generated and saved in {output_path}."
