import os
import pandas as pd
import geopandas as gpd
import skmob
import numpy as np
from skmob.models.gravity import Gravity
import argparse



# TODO: for later
# method train()
# method evaluate()


def grav_Model(tessellation_train, tessellation_test,
               flow_data_train_location, flow_data_test_location,
               gravity_type, out_format, folder_name):
    """
    Fits a gravity model to training flow data and generates synthetic flows or probabilities for test data.

    Parameters:
    tessellation_train (gpd.GeoDataFrame): GeoDataFrame containing the training tessellation data. Should include columns 'GEOID', 'total_population', and 'geometry'.
    tessellation_test (gpd.GeoDataFrame): GeoDataFrame containing the test tessellation data. Should include columns 'GEOID', 'total_population', and 'geometry'.
    flow_data_train_location (str): Path to the training flow data .csv file. The file should include columns 'origin', 'destination', and 'flow'.
    flow_data_test_location (str): Path to the test flow data .csv file. The file should include columns 'origin', 'destination', and 'flow'.
    gravity_type (str): Type of gravity models, options are 'gravity_singly_constrained', 'gravity_globally_constrained'.
    out_format (str): Type of output, 'flows' means synthetic population flows between two locations, and 'probabilities' means the probability of a unit flow between two locations.

    Returns:
    The function saves the generated synthetic flows or probabilities to a CSV file in the specified output directory.

    Notes:
    - The function reads flow data from the specified file paths and processes the tessellation data.
    - The results are saved to a CSV file in the output directory, with the directory structure based on the input file paths.
    """

    # Access the flow data file location and read the flow data
    train_data = skmob.FlowDataFrame.from_file(flow_data_train_location, tessellation=tessellation_train, tile_id='GEOID', sep=',')
    test_data = skmob.FlowDataFrame.from_file(flow_data_test_location, tessellation=tessellation_test, tile_id='GEOID', sep=',')

    # Sum the flow values for each origin, exclude intra-location flows
    outflows_train = train_data[train_data['origin'] != train_data['destination']].groupby('origin')[['flow']].sum().fillna(1)
    outflows_test = test_data[test_data['origin'] != test_data['destination']].groupby('origin')[['flow']].sum().fillna(1)

    # Merge the outflows with the tessellation data
    tessellation_test_merge = tessellation_test.merge(outflows_test, left_on='GEOID', right_on='origin').rename(columns={'flow': 'total_outflows'})

    # Create an instance of gravity model for fitting
    if gravity_type == 'gravity_singly_constrained':
        gravity_type_use = 'singly constrained'
    elif gravity_type == 'gravity_globally_constrained':
        gravity_type_use = 'globally constrained'
    else:
        raise ValueError("Invalid gravity type. Gravity type be 'gravity_singly_constrained' or 'gravity_globally_constrained'.")

    gravity_singly_fitted = Gravity(gravity_type=gravity_type_use)

    # Fit the gravity model to the flow data using 'total_population'
    print('Model fitting starts...') # Notification
    gravity_singly_fitted.fit(train_data, relevance_column='total_population')
    print('Model fitting completed.') # Notification

    # Generate synthetic flow data using the fitted gravity model
    np.random.seed(0)
    synth_fdf_fitted = gravity_singly_fitted.generate(tessellation_test_merge,
                                                      tile_id_column='GEOID',
                                                      tot_outflows_column='total_outflows',
                                                      relevance_column='total_population',
                                                      out_format=out_format)

    # Rename the synthetic column and save to .csv file
    if out_format == 'flows':
        synth_fdf_fitted.rename(columns={'flow': 'flow'}, inplace=True)
    elif out_format == 'probabilities':
        synth_fdf_fitted.rename(columns={'flow': 'probability'}, inplace=True)
    else:
        print("Invalid outputs format. Outputs format be 'flows' or 'probabilities'.")

    # Create output path and save files
    filename = os.path.basename(flow_data_train_location)
    prefix, suffix = filename.split('_', 1)

    output_path = os.path.join('..', 'gravity_model', 'results', f'{folder_name}')
    os.makedirs(output_path, exist_ok=True)

    file_path = os.path.join(output_path, f'synthetic_{gravity_type}_{suffix}')

    synth_fdf_fitted.to_csv(file_path, index=False)