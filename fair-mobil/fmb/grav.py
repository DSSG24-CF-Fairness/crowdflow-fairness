import pandas as pd
import geopandas as gpd
import skmob
import numpy as np
from skmob.models.gravity import Gravity

def grav_Model(): #*** add args
    file_location = input("Enter the location of the .geojson file: ")
    flow_data_file_location = input("Enter the location of the flow data file: ")
    csv_file_location = input("Enter the location of the csv file: ")

    # Access the .geojson file location and read the tessellation file
    url_tess = file_location
    tessellation = gpd.read_file(url_tess).rename(columns={'GEOID10': 'GEOID'})  # Rename required fields if necessary
    print(tessellation.head())

    # Select relevant columns
    tessellation = tessellation[['OBJECTID', 'GEOID', 'ACRES_TOTAL', 'Total_Population', 'geometry']]
    print(tessellation.head())

    # Access the flow data file location and read the flow data
    flow_data = skmob.FlowDataFrame.from_file(flow_data_file_location, tessellation=tessellation, tile_id='GEOID', sep=",")
    print(flow_data.head())

    # Sum the flow values for each origin, exclude intra-location flows
    outflows = flow_data[flow_data['origin'] != flow_data['destination']].groupby('origin')[['flow']].sum().fillna(0)

    # Merge the outflows with the tessellation data
    tessellation = tessellation.merge(outflows, left_on='GEOID', right_on='origin').rename(columns={'flow': 'tot_outflow'})
    print(tessellation.head())

    # Create an instance of a singly constrained gravity model
    gravity_singly = Gravity(gravity_type='singly constrained')
    print(gravity_singly)

    # Generate synthetic flow data using the established gravity model
    np.random.seed(0)
    synth_fdf = gravity_singly.generate(tessellation,
                                        tile_id_column='GEOID',
                                        tot_outflows_column='tot_outflow',
                                        relevance_column='Total_Population',
                                        out_format='flows')

    # New instance of gravity model for fitting
    gravity_singly_fitted = Gravity(gravity_type='singly constrained')
    print(gravity_singly_fitted)

    # Fit the gravity model to the flow data using 'Total_Population'
    gravity_singly_fitted.fit(flow_data, relevance_column='Total_Population')
    print(gravity_singly_fitted)

    # Generate synthetic flow data using the fitted gravity model
    np.random.seed(0)
    synth_fdf_fitted = gravity_singly_fitted.generate(tessellation,
                                                    tile_id_column='GEOID',
                                                    tot_outflows_column='tot_outflow',
                                                    relevance_column='Total_Population',
                                                    out_format='flows')

    # Read additional features from a CSV file
    features_df = pd.read_csv(csv_file_location)

    # Backup the original tessellation from flow_data
    tessellation_backup = flow_data.tessellation

    # Rename the flow column to 'mFlow' in the synthetic dataframe
    synth_fdf_fitted.rename(columns={'flow': 'mFlow'}, inplace=True)

    # Merge original flow data with the synthetic data
    flow_data = flow_data.merge(synth_fdf_fitted, on=['origin', 'destination'], how='left')

    # Convert the merged dataframe to a FlowDataFrame
    flow_data = skmob.FlowDataFrame(flow_data, tessellation=tessellation_backup, tile_id='GEOID')
    print(flow_data.tessellation)

if __name__ == "__main__":
    grav_Model() #*** add args