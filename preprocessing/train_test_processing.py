import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pyproj
from shapely.geometry import box, Point
from sklearn.model_selection import StratifiedShuffleSplit


def load_state_or_county_data(file_path):
    """
    Load a state or county PUMAs .geojson file and set its CRS to EPSG:4326.

    Parameters:
    file_path (str): The file path to the state/county's PUMAs .geojson file.

    Returns:
    geopandas.GeoDataFrame: The loaded GeoDataFrame with CRS set to EPSG:4326.
    """

    state = gpd.read_file(file_path)
    return state.to_crs(epsg=4326)


def create_grid(polygon, cell_size_km, crs='EPSG:4326'):
    """
    Create a grid of cells over a given polygon area with a specified cell size.

    Parameters:
    polygon (shapely.geometry.Polygon): The polygon area over which the grid is created.
    cell_size_km (float): The size of each grid cell in kilometers.
    crs (str): The coordinate reference system for the grid. Default is 'EPSG:4326'.

    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame containing the grid cells with their geometry and grid indices.
    """

    # Get the bounds (with buffer area) of the polygon area
    bounds = polygon.bounds
    minx, miny, maxx, maxy = bounds
    buffer = 0.1  # Adjust as needed
    minx -= buffer
    miny -= buffer
    maxx += buffer
    maxy += buffer

    cell_size = cell_size_km / 110 # Conversion from km to degrees
    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(miny, maxy, cell_size)

    grid_cells = [box(x, y, x + cell_size, y + cell_size).intersection(polygon)
                  for x in x_coords for y in y_coords
                  if box(x, y, x + cell_size, y + cell_size).intersects(polygon)
                  and not box(x, y, x + cell_size, y + cell_size).intersection(polygon).is_empty]

    grid_gdf = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)
    grid_gdf['region_index'] = range(1, len(grid_gdf) + 1)
    return grid_gdf


def flow_train_test_split(tessellation_df, features_df, grid, folder_name, crs='EPSG:4326'):
    """
    Map census tracts to grid cells based on population and split the data into train and test sets using stratified shuffle split.

    Parameters:
    flow_df (pd.DataFrame): DataFrflow_train_test_splitame containing origin and destination flow data with geoid, longitude, and latitude.
    features_df (pd.DataFrame): DataFrame containing population data for census tracts with geoid and total population.
    grid (geopandas.GeoDataFrame): GeoDataFrame containing the grid cells with geometry and area_index.
    crs (str): The coordinate reference system for the geographic data. Default is 'EPSG:4326'.

    Returns:
    tuple: Two DataFrames for train and test sets containing grid indices and census tract geoids.
    .csv: Two .csv files are saved for train and test region indices.
    """

    # Convert the X and Y coordinates to latitude and longitude
    tessellation_df['centroid'] = gpd.points_from_xy(tessellation_df['lng'], tessellation_df['lat'])
    census_tracts_gdf = gpd.GeoDataFrame(tessellation_df[['GEOID', 'lng', 'lat', 'centroid']], geometry='centroid')
    census_tracts_gdf = census_tracts_gdf.set_crs('epsg:4326', allow_override=True)


    # Match census tracts with grid cells
    matched_gdf = gpd.sjoin(census_tracts_gdf, grid[['geometry', 'region_index']], how='left')
    matched_gdf = matched_gdf.drop(columns='centroid').fillna(-1).astype({'GEOID': 'int64', 'region_index': 'int64'})
    matched_gdf = matched_gdf[['GEOID', 'region_index']].rename(columns={'GEOID': 'geoid'})
    merged_df = pd.merge(matched_gdf, features_df, on='geoid', how='left').fillna(0).astype({'total_population': 'int'})
    grouped_df = merged_df.groupby('region_index').agg(
        region_index=('region_index', 'first'),
        grid_population=('total_population', 'sum'),
        census_tracts_geoids=('geoid', lambda x: ','.join(x.astype(str)))
    ).reset_index(drop=True)
    grouped_df = grouped_df[grouped_df['region_index'] != -1]

    # Sort by population and create stratified splits
    sorted_df = grouped_df.sort_values(by='grid_population', ascending=False)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=122)
    sorted_df['stratify'] = sorted_df.index % 2
    for train_idx, test_idx in splitter.split(sorted_df, sorted_df['stratify']):
        train_set = sorted_df.iloc[train_idx]
        test_set = sorted_df.iloc[test_idx]

    # Save region indices with corresponding census tracts geoids to CSV files
    train_set.reset_index(inplace=True)
    test_set.reset_index(inplace=True)

    train_set[['region_index']].to_csv(f'../processed_data/{folder_name}/train_region_index.csv', index=False)
    test_set[['region_index']].to_csv(f'../processed_data/{folder_name}/test_region_index.csv', index=False)

    train_tile_geoids = train_set[['region_index', 'census_tracts_geoids']]
    train_tile_geoids.to_csv(f'../processed_data/{folder_name}/train_tile_geoids.csv', index=False)

    test_tile_geoids = test_set[['region_index', 'census_tracts_geoids']]
    test_tile_geoids.to_csv(f'../processed_data/{folder_name}/test_tile_geoids.csv', index=False)

    return train_set[['region_index', 'census_tracts_geoids']], test_set[['region_index', 'census_tracts_geoids']]



def filter_train_test_data(flow_df, tessellation_df, features_df, train_set, test_set, folder_name, balance_sets=False):
    """
    Filters and optionally balances the data based on geoid population, then saves the filtered data for flow, tessellation, and features.

    Parameters:
    flow_df (pd.DataFrame): DataFrame with flow data.
    tessellation_df (pd.DataFrame): DataFrame containing tessellation data with geoids.
    features_df (pd.DataFrame): DataFrame containing population information for each geoid.
    train_set (pd.DataFrame): DataFrame with training set geoids.
    test_set (pd.DataFrame): DataFrame with test set geoids.
    balance_sets (bool): Flag to determine if the train/test sets should be balanced based on population. Default is False.

    Returns:
    None: Saves CSV files for adjusted train and test datasets for flow, tessellation, and features in corresponding path.
    """

    # Standardize geoid formats and ensure they are strings
    train_set['census_tracts_geoids'] = train_set['census_tracts_geoids'].astype(str).str.strip()
    test_set['census_tracts_geoids'] = test_set['census_tracts_geoids'].astype(str).str.strip()
    features_df['geoid'] = features_df['geoid'].astype(str).str.strip()
    tessellation_df['GEOID'] = tessellation_df['GEOID'].astype(str).str.strip()
    flow_df['geoid_o'] = flow_df['geoid_o'].astype(str).str.strip()
    flow_df['geoid_d'] = flow_df['geoid_d'].astype(str).str.strip()

    # Extract geoids from train and test outputs
    train_geoids = set()
    test_geoids = set()

    for geoids in train_set['census_tracts_geoids']:
        train_geoids.update(geoids.split(','))
                                         
    for geoids in test_set['census_tracts_geoids']:
        test_geoids.update(geoids.split(','))
                                        
    # Ensure geoids are strings
    train_geoids = {str(geoid).strip() for geoid in train_geoids}
    test_geoids = {str(geoid).strip() for geoid in test_geoids}

    # Get population numbers for the geoids from features
    train_pop = features_df[features_df['geoid'].isin(train_geoids)]
    test_pop = features_df[features_df['geoid'].isin(test_geoids)]

    if balance_sets:
        # Find which set is larger and balance the geoids by removing the least populated geoids from the larger set
        if len(train_pop) > len(test_pop):
            # Balance by reducing train_pop
            train_pop = train_pop.sort_values(by='total_population', ascending=True)
            train_pop = train_pop.iloc[:len(test_pop)]
        else:
            # Balance by reducing test_pop
            test_pop = test_pop.sort_values(by='total_population', ascending=True)
            test_pop = test_pop.iloc[:len(train_pop)]

        # Update geoids sets after balancing
        train_geoids = set(train_pop['geoid'])
        test_geoids = set(test_pop['geoid'])

    # Filter datasets based on adjusted geoid sets
    train_flows = flow_df[flow_df['geoid_o'].isin(train_geoids) & flow_df['geoid_d'].isin(train_geoids)]
    test_flows = flow_df[flow_df['geoid_o'].isin(test_geoids) & flow_df['geoid_d'].isin(test_geoids)]
    train_to_test_flows = flow_df[flow_df['geoid_o'].isin(train_geoids) & flow_df['geoid_d'].isin(test_geoids)]
    test_to_train_flows = flow_df[flow_df['geoid_o'].isin(test_geoids) & flow_df['geoid_d'].isin(train_geoids)]
    train_tessellation = tessellation_df[tessellation_df['GEOID'].isin(train_geoids)]
    test_tessellation = tessellation_df[tessellation_df['GEOID'].isin(test_geoids)]
    train_features = features_df[features_df['geoid'].isin(train_geoids)]
    test_features = features_df[features_df['geoid'].isin(test_geoids)]

    def convert_to_set(df):
        # go through each row of df
        res = set()
        for index, row in df.iterrows():
            curr = (row["geoid_o"], row["geoid_d"], row["pop_flows"])
            res.add(curr)
        return res

    D_set_flow_df = convert_to_set(flow_df)
    D_set_train_flows = convert_to_set(train_flows)
    D_set_test_flows = convert_to_set(test_flows)
    D_set_train_to_test_flows = convert_to_set(train_to_test_flows)
    D_set_test_to_train_flows = convert_to_set(test_to_train_flows)

    D_set_missing_flow = D_set_flow_df - D_set_train_flows - D_set_test_flows - D_set_train_to_test_flows - D_set_test_to_train_flows

    # convert D_set_missing_flow to a dataframe
    missing_flow_df = pd.DataFrame(list(D_set_missing_flow), columns=["geoid_o", "geoid_d", "pop_flows"])

    # Paths for the directories
    train_flows_dir = f'../processed_data/{folder_name}/train/'
    test_flows_dir = f'../processed_data/{folder_name}/test/'
    train_tessellation_dir = f'../processed_data/{folder_name}/train/'
    test_tessellation_dir = f'../processed_data/{folder_name}/test/'
    train_features_dir = f'../processed_data/{folder_name}/train/'
    test_features_dir = f'../processed_data/{folder_name}/test/'

    # Create directories if they do not exist
    os.makedirs(train_flows_dir, exist_ok=True)
    os.makedirs(test_flows_dir, exist_ok=True)
    os.makedirs(train_tessellation_dir, exist_ok=True)
    os.makedirs(test_tessellation_dir, exist_ok=True)
    os.makedirs(train_features_dir, exist_ok=True)
    os.makedirs(test_features_dir, exist_ok=True)

    # Renaming geoid_o and geoid_d to origin and destination
    train_flows = train_flows.rename(columns={'geoid_o': 'origin', 'geoid_d': 'destination', 'pop_flows': 'flow'})
    test_flows = test_flows.rename(columns={'geoid_o': 'origin', 'geoid_d': 'destination', 'pop_flows': 'flow'})

    # Drop rows with flow value of NaN and 0
    train_flows = train_flows[(train_flows['flow'].notna())]
    test_flows = test_flows[(test_flows['flow'].notna())]
    train_flows = train_flows.loc[train_flows['flow'] > 0]
    test_flows = test_flows.loc[test_flows['flow'] > 0]

    # Save files to their respective directories
    train_flows[['origin', 'destination', 'flow']].to_csv(train_flows_dir + 'train_flow.csv', index=False)
    test_flows[['origin', 'destination', 'flow']].to_csv(test_flows_dir + 'test_flow.csv', index=False)
    train_tessellation[['GEOID', 'lng', 'lat', 'geometry', 'total_population']].to_file(train_tessellation_dir + 'train_tessellation.geojson', driver='GeoJSON')
    test_tessellation[['GEOID', 'lng', 'lat', 'geometry', 'total_population']].to_file(test_tessellation_dir + 'test_tessellation.geojson', driver='GeoJSON')
    train_features.to_csv(train_features_dir + 'train_features.csv', index=False)
    test_features.to_csv(test_features_dir + 'test_features.csv', index=False)

    print('Processed and saved all datasets successfully.')


    # Create directories if they do not exist
    diagnosis_path = f'../processed_data/{folder_name}'
    os.makedirs(diagnosis_path, exist_ok=True)

    output_path = os.path.join(diagnosis_path, 'missing_flow.csv')
    missing_flow_df.to_csv(output_path, index=False)
    print(f'Missing flows have been saved to {output_path}.')