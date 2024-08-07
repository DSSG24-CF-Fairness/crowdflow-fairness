import os
import pandas as pd
import geopandas as gpd
import skmob
import numpy as np
from skmob.models.gravity import Gravity
import argparse

pd.set_option('display.max_columns', None)



# Settings
# parser = argparse.ArgumentParser(description="Run gravity model.")
# parser.add_argument('--tessellation_file_train_location', type=str,
#                     help="Path to the training tessellation .geojson file.")
# parser.add_argument('--tessellation_file_test_location', type=str,
#                     help="Path to the test tessellation .geojson file.")
# parser.add_argument('--flow_data_train_location', type=str,
#                     help="Path to the training flow data .csv file.")
# parser.add_argument('--flow_data_test_location', type=str,
#                     help="Path to the test flow data .csv file.")
# parser.add_argument('--gravity_type', type=str, default='singly_constrained', choices=['singly_constrained', 'globally_constrained'],
#                     help="Type of gravity model, 'singly_constrained' or 'globally'. Default is 'singly_constrained'.")
# parser.add_argument('--out_format', type=str, default='flows', choices=['flows', 'probabilities'],
#                     help="Type of output, 'flows' or 'probabilities'. Default is 'flows'.")

"""
- Change to dataframes not paths
- Delete arguments 
- for later - method train()
- for later - method evaluate()

"""

def grav_Model(tessellation_train, tessellation_test,
               flow_data_train_location, flow_data_test_location,
               gravity_type, out_format,experiment_id = "0"):
    '''
    Inputs:
    tessellation_file_train_location (str): Path to the training tessellation .geojson file.
    tessellation_file_test_location (str): Path to the test tessellation .geojson file.
    tessellation file (train and test) - include columns 'GEOID', 'total_population', 'geometry'.

    flow_data_train_location (str): Path to the training flow data .csv file.
    flow_data_test_location (str): Path to the test flow data .csv file.
    flow data file (train and test) - include columns 'origin', 'destination', 'flow'.

    gravity_type (str): Type of gravity models, options are 'gravity_singly_constrained', 'gravity_globally_constrained'
    out_format (str): Type of output, 'flows' means synthetic population flows between two locations,
                          and 'probabilities' means the probability of a unit flow between two locations.
    '''

    # Access the flow data file location and read the flow data
    train_data = skmob.FlowDataFrame.from_file(flow_data_train_location, tessellation=tessellation_train, tile_id='GEOID', sep=",")
    test_data = skmob.FlowDataFrame.from_file(flow_data_test_location, tessellation=tessellation_test, tile_id='GEOID', sep=",")

    # Sum the flow values for each origin, exclude intra-location flows
    outflows_train = train_data[train_data['origin'] != train_data['destination']].groupby('origin')[['flow']].sum().fillna(0)
    outflows_test = test_data[test_data['origin'] != test_data['destination']].groupby('origin')[['flow']].sum().fillna(0)

    # Merge the outflows with the tessellation data
    # tessellation_train_merge = tessellation_train.merge(outflows_train, left_on='GEOID', right_on='origin').rename(columns={'flow': 'total_outflows'})
    tessellation_test_merge = tessellation_test.merge(outflows_test, left_on='GEOID', right_on='origin').rename(columns={'flow': 'total_outflows'})

    # Create an instance of gravity model for fitting
    if gravity_type == 'gravity_singly_constrained':
        gravity_type_use = 'singly constrained'
    elif gravity_type == 'gravity_globally_constrained':
        gravity_type_use = 'globally constrained'
    else:
        raise ValueError("Invalid gravity type.")
    
    
    gravity_singly_fitted = Gravity(gravity_type=gravity_type_use)

    # Fit the gravity model to the flow data using 'total_population'
    gravity_singly_fitted.fit(train_data, relevance_column='total_population')

    # Generate synthetic flow data using the fitted gravity model
    np.random.seed(0)
    synth_fdf_fitted = gravity_singly_fitted.generate(tessellation_test_merge,
                                                      tile_id_column='GEOID',
                                                      tot_outflows_column='total_outflows',
                                                      relevance_column='total_population',
                                                      out_format=out_format)

    # Rename the synthetic column and save to .csv file
    if out_format == 'flows':
        synth_fdf_fitted.rename(columns={'flow': 'synthetic_flows'}, inplace=True)
    elif out_format == 'probabilities':
        synth_fdf_fitted.rename(columns={'flow': 'synthetic_probabilities'}, inplace=True)
    else:
        print("Invalid outputs format.")

    # Create output path and save files 
    path_parts = flow_data_train_location.split(os.sep)
    filename = path_parts[-1]

    flow_dir_index = path_parts.index('flows') + 1
    subdirectory = os.sep.join(path_parts[flow_dir_index:-1])
    if subdirectory:
        # Create output filename for structured subdirectory cases
        new_filename = filename.replace('.csv', f'_synthetic_{out_format}.csv')
        output_path = os.path.join('..', 'outputs', experiment_id, subdirectory, new_filename)
    else:
        # Create output filename for general cases without subdirectory
        output_path = os.path.join('..', 'outputs', experiment_id, f'_synthetic_{out_format}.csv')

    synth_fdf_fitted.to_csv(output_path, index=False)


# def main():
#     args = parser.parse_args()
#     grav_Model(args.tessellation_file_train_location, args.tessellation_file_test_location,
#                args.flow_data_train_location, args.flow_data_test_location,
#                args.gravity_type, args.out_format)



# if __name__ == "__main__":
#     main()


"""
Test run command line (change to your local path):
python models/gravity.py \
--tessellation_file_train_location /Users/apple/Documents/GitHub/DSSG/crowdflow-fairness/data/tessellation_use_this.geojson \
--tessellation_file_test_location /Users/apple/Documents/GitHub/DSSG/crowdflow-fairness/data/tessellation_use_this.geojson \
--flow_data_train_location /Users/apple/Documents/GitHub/DSSG/crowdflow-fairness/data/flows.csv \
--flow_data_test_location /Users/apple/Documents/GitHub/DSSG/crowdflow-fairness/data/flows.csv \
--gravity_type singly_constrained \
--out_format flows
"""


