import pandas as pd
import geopandas as gpd
import skmob
import numpy as np
from .mobfair import FairMob
from skmob.models.radiation import Radiation
import matplotlib.pyplot as plt

def rad_Model():#*** add args
    file_location = input("Enter the location of the .geojson file: ")
    flow_data_file_location = input("Enter the location of the flow data file: ")
    csv_file_location = input("Enter the location of the csv file: ")

    # Access the .geojson file location and read the tessellation file
    url_tess = file_location
    tessellation = gpd.read_file(url_tess).rename(columns={'GEOID10': 'GEOID'})
    print(tessellation.head())

    # Select relevant columns
    tessellation = tessellation[['OBJECTID', 'GEOID', 'ACRES_TOTAL', 'Total_Population', 'geometry']]
    print(tessellation.head())
    tessellation_copy = tessellation.copy()

    # Access the flow data file location and read the flow data
    flow_data = skmob.FlowDataFrame.from_file(flow_data_file_location, tessellation=tessellation, tile_id='GEOID', sep=",")
    print(flow_data.head())

    # Sum the flow values for each origin, exclude intra-location flows
    outflows = flow_data[flow_data['origin'] != flow_data['destination']].groupby('origin')[['flow']].sum().fillna(0)

    # Merge the outflows with the tessellation data
    tessellation = tessellation.merge(outflows, left_on='GEOID', right_on='origin').rename(columns={'flow': 'tot_outflow'})
    print(tessellation.head())

    # Create an instance of the Radiation model
    np.random.seed(0)
    radiation = Radiation()

    # Generate synthetic flow data using the Radiation model
    rad_flows = radiation.generate(tessellation,
                                tile_id_column='GEOID',
                                tot_outflows_column='tot_outflow',
                                relevance_column='Total_Population',
                                out_format='flows_sample')

    # Read additional features from a CSV file
    features_df = pd.read_csv(csv_file_location)
    print("Data type of GEOID column:", features_df['GEOID'].dtype)

    # Backup the original tessellation from flow_data
    tessellation_backup = flow_data.tessellation

    # Rename the flow column to 'mFlow' in the synthetic dataframe
    rad_flows.rename(columns={'flow': 'mFlow'}, inplace=True)

    # Merge original flow data with the synthetic data
    flow_data = flow_data.merge(rad_flows, on=['origin', 'destination'], how='left')

    # Convert the merged dataframe to a FlowDataFrame
    flow_data = skmob.FlowDataFrame(flow_data, tessellation=tessellation_backup, tile_id='GEOID')
    print(flow_data.tessellation)

    # Change the data type of the 'origin' and 'destination' columns to int64
    flow_data['origin'] = flow_data['origin'].astype('int64')
    print("New data type of origin:", flow_data['origin'].dtype)
    flow_data['destination'] = flow_data['destination'].astype('int64')
    print("New data type of destination:", flow_data['destination'].dtype)

    # Find common values between flow data and features
    common_values = set(flow_data['origin']).intersection(set(features_df['GEOID']))
    print("Number of common values:", len(common_values))

    # Initialize and run fairness analysis for origin
    fm = FairMob()
    fm.runFairness(
        df=flow_data,
        measurementdf=features_df,
        percentile=10,
        realflow='flow',
        modelFlow='mFlow',
        measurementOrigin='GEOID',
        measurement='RPL_THEMES',
        onOrigin=True,
        bidirectional=False
    )

    # Initialize and run fairness analysis for destination
    fmNext = FairMob()
    fmNext.runFairness(
        df=flow_data,
        measurementdf=features_df,
        percentile=10,
        realflow='flow',
        modelFlow='mFlow',
        measurementOrigin='GEOID',
        measurement='RPL_THEMES',
        onOrigin=False,
        bidirectional=False
    )

    # Data for origin and destination in the radiation model
    data_radiation = {
        'Origin': {
            'Jensen-Shannon Divergence': 0.0,
            'KL Divergence': 0.0,
            'T-statistic': 1.0,
            'p-value': 0.0
        },
        'Destination': {
            'Jensen-Shannon Divergence': 0.0,
            'KL Divergence': 0.0,
            'T-statistic': -1.0,
            'p-value': 0.0
        }
    }

    # Plotting the data
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Create a 2x2 grid of subplots

    # Jensen-Shannon Divergence
    axs[0, 0].bar(data_radiation.keys(), [data_radiation['Origin']['Jensen-Shannon Divergence'], data_radiation['Destination']['Jensen-Shannon Divergence']])
    axs[0, 0].set_title('Jensen-Shannon Divergence')
    axs[0, 0].set_ylabel('Divergence Value')

    # KL Divergence
    axs[0, 1].bar(data_radiation.keys(), [data_radiation['Origin']['KL Divergence'], data_radiation['Destination']['KL Divergence']])
    axs[0, 1].set_title('KL Divergence')
    axs[0, 1].set_ylabel('Divergence Value')

    # T-statistic
    axs[1, 0].scatter(data_radiation.keys(), [data_radiation['Origin']['T-statistic'], data_radiation['Destination']['T-statistic']], color='red')
    axs[1, 0].set_title('T-statistic')
    axs[1, 0].set_ylabel('T-statistic Value')

    # P-value
    axs[1, 1].scatter(data_radiation.keys(), [data_radiation['Origin']['p-value'], data_radiation['Destination']['p-value']], color='blue')
    axs[1, 1].set_title('p-value')
    axs[1, 1].set_ylabel('p-value')

    plt.tight_layout()
    plt.show()

    # Load your SVI data from a CSV file
    svi_df = pd.read_csv('${file_location}')  # Update path as necessary

    # Print column names for both dataframes
    print("Tessellation columns:", tessellation.columns)
    print("SVI DataFrame columns:", svi_df.columns)

    # Merge tessellation data with SVI data on 'GEOID'
    merged_data = tessellation_copy.merge(svi_df, on='GEOID', how='inner')

    # Plotting SVI data
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    merged_data.plot(column='RPL_THEMES', cmap='Greens', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
                    legend_kwds={'label': "RPL Themes Index (Wealth Indication)",
                                'orientation': "horizontal"})
    plt.title('Wealth Distribution in ${location} Area')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    merged_data.plot(column='RPL_THEMES', cmap='Greens_r', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
                    legend_kwds={'label': "RPL Themes Index (Vulnerability Level)",
                                'orientation': "horizontal"})
    plt.title('SVI in ${location} Area')
    plt.show()

    # Calculate centroids of each polygon
    merged_data['centroids'] = merged_data.geometry.centroid

    # Create a new GeoDataFrame for centroids
    centroids_gdf = gpd.GeoDataFrame(merged_data, geometry='centroids')

    # Plotting centroids with vulnerability levels
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    centroids_gdf.plot(ax=ax, color='white', edgecolor='black')  # Plot the background tessellation
    centroids_gdf.plot(ax=ax, column='RPL_THEMES', cmap='Greens_r', markersize=100, legend=True,
                    legend_kwds={'label': "RPL Themes Index (Vulnerability Level)",
                                    'orientation': "horizontal"})
    plt.title('Wealth Distribution in ${location} Area (Higher Vulnerability Shown by Lighter Colors)')
    plt.show()

    # Calculate CPC (Common Part of Commuters)
    flow_data['CPC'] = 2.0 * np.minimum(flow_data['flow'], flow_data['mFlow']) / (flow_data['flow'] + flow_data['mFlow'])
    flow_data.loc[(flow_data['mFlow'] == 0.0) & (flow_data['flow'] == 0.0), 'CPC'] = 1

    # Calculate mean CPC values and group by origin
    cpc_data = flow_data.groupby('origin')['CPC'].mean().reset_index().rename(columns={'origin': 'GEOID', 'CPC': 'mean_CPC'})

    # Merge CPC data with tessellation
    tessellation = tessellation.merge(cpc_data, on='GEOID')

    # Plot tessellation with CPC values
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    tessellation.plot(column='mean_CPC', ax=ax, legend=True, cmap='viridis')
    plt.title('Mean CPC Values for ${location} Area')
    plt.show()

    # Adjust the aspect ratio and extent of the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    tessellation.plot(column='mean_CPC', ax=ax, legend=True, cmap='viridis')
    ax.set_aspect('equal')
    ax.set_xlim(tessellation.total_bounds[0], tessellation.total_bounds[2])
    ax.set_ylim(tessellation.total_bounds[1], tessellation.total_bounds[3])
    plt.title('Mean CPC Values for ${location} Area')
    plt.show()

    # Plot the tessellation to ensure it covers the entire location area
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    tessellation.plot(ax=ax)
    plt.title('Tessellation of ${location} Area')
    plt.show()

    # Final tessellation plot
    tessellation.plot()
    
if __name__ == "__main__":
    rad_Model() #*** add args