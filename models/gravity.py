import pandas as pd
import geopandas as gpd
import skmob
import numpy as np
from skmob.models.gravity import Gravity
import argparse

pd.set_option('display.max_columns', None)



# Settings
parser = argparse.ArgumentParser(description="Run gravity model.")
parser.add_argument('tessellation_file_train_location', type=str,
                    help="Path to the training tessellation .geojson file.")
parser.add_argument('tessellation_file_test_location', type=str,
                    help="Path to the test tessellation .geojson file.")
parser.add_argument('flow_data_file_train_location', type=str,
                    help="Path to the training flow data .csv file.")
parser.add_argument('flow_data_file_test_location', type=str,
                    help="Path to the test flow data .csv file.")
parser.add_argument('gravity_type', type=str, default='singly_constrained', choices=['singly_constrained', 'globally_constrained'],
                    help="Type of gravity model, 'singly_constrained' or 'globally'. Default is 'singly_constrained'.")
parser.add_argument('out_format', type=str, default='flows', choices=['flows', 'probabilities'],
                    help="Type of output, 'flows' or 'probabilities'. Default is 'flows'.")


def grav_Model(tessellation_file_train_location, tessellation_file_test_location,
               flow_data_file_train_location, flow_data_file_test_location,
               gravity_type, out_format):
    '''
    Inputs:
    tessellation_file_train_location (str): Path to the training tessellation .geojson file.
    tessellation_file_test_location (str): Path to the test tessellation .geojson file.
    tessellation file (train and test) - include columns 'GEOID', 'total_population', 'geometry'.

    flow_data_file_train_location (str): Path to the training flow data .csv file.
    flow_data_file_test_location (str): Path to the test flow data .csv file.
    flow data file (train and test) - include columns 'destination', 'origin', 'flow'.

    gravity_type (str): Type of gravity models.
    out_format (str): Type of output, 'flows' means synthetic population flows between two locations,
                          and 'probabilities' means the probability of a unit flow between two locations.
    '''

    # Access the .geojson file location and read the tessellation file
    tessellation_train = gpd.read_file(tessellation_file_train_location)
    tessellation_test = gpd.read_file(tessellation_file_test_location)

    # Access the flow data file location and read the flow data
    flow_data_train = skmob.FlowDataFrame.from_file(flow_data_file_train_location, tessellation=tessellation_train, tile_id='GEOID', sep=",")
    flow_data_test = skmob.FlowDataFrame.from_file(flow_data_file_test_location, tessellation=tessellation_test, tile_id='GEOID', sep=",")

    # Sum the flow values for each origin, exclude intra-location flows
    outflows_train = flow_data_train[flow_data_train['origin'] != flow_data_train['destination']].groupby('origin')[['flow']].sum().fillna(0)
    outflows_test = flow_data_test[flow_data_test['origin'] != flow_data_test['destination']].groupby('origin')[['flow']].sum().fillna(0)

    # Merge the outflows with the tessellation data
    tessellation_train_merge = tessellation_train.merge(outflows_train, left_on='GEOID', right_on='origin').rename(columns={'flow': 'total_outflows'})
    tessellation_test_merge = tessellation_test.merge(outflows_test, left_on='GEOID', right_on='origin').rename(columns={'flow': 'total_outflows'})

    # Create an instance of gravity model for fitting
    if gravity_type == 'singly_constrained':
        gravity_type_use = 'singly constrained'
    elif gravity_type == 'globally_constrained':
        gravity_type_use = 'globally constrained'
    else:
        print("Invalid gravity type.")
    gravity_singly_fitted = Gravity(gravity_type=gravity_type_use)

    # Fit the gravity model to the flow data using 'total_population'
    gravity_singly_fitted.fit(flow_data_train, relevance_column='total_population')

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
        synth_fdf_fitted.to_csv('../crowdflow-fairness/outputs/gravity_synthetic_flows.csv', index=False)
    elif out_format == 'probabilities':
        synth_fdf_fitted.rename(columns={'flow': 'synthetic_probabilities'}, inplace=True)
        synth_fdf_fitted.to_csv('../crowdflow-fairness/outputs/gravity_synthetic_probabilities.csv', index=False)
    else:
        print("Invalid outputs format.")


def main():
    args = parser.parse_args()
    grav_Model(args.tessellation_file_train_location, args.tessellation_file_test_location,
               args.flow_data_file_train_location, args.flow_data_file_test_location,
               args.gravity_type, args.out_format)



if __name__ == "__main__":
    main()



"""
Test run command line:
python models/gravity.py /Users/apple/Documents/GitHub/DSSG/crowdflow-fairness/data/tessellation_use_this.geojson /Users/apple/Documents/GitHub/DSSG/crowdflow-fairness/data/tessellation_use_this.geojson /Users/apple/Documents/GitHub/DSSG/crowdflow-fairness/data/flows.csv /Users/apple/Documents/GitHub/DSSG/crowdflow-fairness/data/flows.csv singly_constrained flows
"""