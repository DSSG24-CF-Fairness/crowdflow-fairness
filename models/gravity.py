import pandas as pd
import geopandas as gpd
import skmob
import numpy as np
from skmob.models.gravity import Gravity

pd.set_option('display.max_columns', None)


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
    gravity_singly_fitted = Gravity(gravity_type=gravity_type)

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
        synth_fdf_fitted.to_csv('../outputs/gravity_synthetic_flows.csv', index=False)
    elif out_format == 'probabilities':
        synth_fdf_fitted.rename(columns={'flow': 'synthetic_probabilities'}, inplace=True)
        synth_fdf_fitted.to_csv('../outputs/gravity_synthetic_probabilities.csv', index=False)
    else:
        print("Invalid outputs format.")

if __name__ == "__main__":
    tessellation_file_train_location = input("Enter the location of the training tessellation .geojson file: ")
    tessellation_file_test_location = input("Enter the location of the test tessellation .geojson file: ")
    flow_data_file_train_location = input("Enter the location of the training flow data .csv file: ")
    flow_data_file_test_location = input("Enter the location of the test flow data .csv file: ")
    gravity_type = input("Enter 'singly constrained' or 'globally constrained': ")
    out_format = input("Enter 'flows' or 'probabilities': ")

    grav_Model(tessellation_file_train_location, tessellation_file_test_location,
               flow_data_file_train_location, flow_data_file_test_location,
               gravity_type, out_format)




'''
Test run:
change to your local path
tessellation_file_train_location = '/Users/apple/Documents/GitHub/DSSG/crowdflow-fairness/data/tessellation_use_this.geojson'
tessellation_file_test_location = '/Users/apple/Documents/GitHub/DSSG/crowdflow-fairness/data/tessellation_use_this.geojson'
flow_data_file_train_location = '/Users/apple/Documents/GitHub/DSSG/crowdflow-fairness/data/flows.csv'
flow_data_file_test_location = '/Users/apple/Documents/GitHub/DSSG/crowdflow-fairness/data/flows.csv'
'''
