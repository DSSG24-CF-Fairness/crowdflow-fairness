import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pyproj
from shapely.geometry import box, Point
from sklearn.model_selection import StratifiedShuffleSplit



pd.set_option('display.max_columns', None)


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


def create_grid(polygon, cell_size_km, crs="EPSG:4326"):
    """
    Create a grid of cells over a given polygon area with a specified cell size.

    Parameters:
    polygon (shapely.geometry.Polygon): The polygon area over which the grid is created.
    cell_size_km (float): The size of each grid cell in kilometers.
    crs (str): The coordinate reference system for the grid, EPSG:4326.

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


def flow_train_test_split(tessellation_df, features_df, grid, experiment_id = "0", crs="EPSG:4326"):
    """
    Mapping of census tracts to grid cells based on population and splitting into train and test sets using stratified shuffle split.

    Parameters:
    flow_df (pd.DataFrame): DataFrame containing origin and destination flow data with geoid, longitude, and latitude.
    features_df (pd.DataFrame): DataFrame containing population data for census tracts with geoid and total population.
    crs (str): The coordinate reference system for the geographic data.
    grid (geopandas.GeoDataFrame): GeoDataFrame containing the grid cells with geometry and area_index.

    Returns:
    tuple: Two DataFrames for train and test sets containing grid indices and census tract geoids.
    .csv: Two .csv files are saved for train and test region indices.
    """
    
    # Define the input and output coordinate systems
    input_crs = pyproj.CRS.from_epsg(2927)  # Washington State Plane North (EPSG:2927)
    output_crs = pyproj.CRS.from_epsg(4326)  # WGS84 (EPSG:4326)
    # Create a transformer object for the coordinate conversion
    transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)
    # Convert the X and Y coordinates to latitude and longitude
    tessellation_df['lng'], tessellation_df['lat'] = transformer.transform(tessellation_df['XCOORD'].values, tessellation_df['YCOORD'].values)

    tessellation_df['geometry'] = gpd.points_from_xy(tessellation_df['lng'], tessellation_df['lat'])
    census_tracts_gdf = tessellation_df[['GEOID', 'lng', 'lat']]


    # # Create DataFrame for census tracts with geoids, longitudes, and latitudes
    # geo_o = flow_df[['geoid_o', 'lng_o', 'lat_o']].rename(columns={'geoid_o': 'geoid', 'lng_o': 'lng', 'lat_o': 'lat'})
    # geo_d = flow_df[['geoid_d', 'lng_d', 'lat_d']].rename(columns={'geoid_d': 'geoid', 'lng_d': 'lng', 'lat_d': 'lat'})
    # census_tracts = pd.concat([geo_o, geo_d]).drop_duplicates().reset_index(drop=True)

    # Create GeoDataFrame for census tracts
    # geometry = [Point(xy) for xy in zip(census_tracts['lng'], census_tracts['lat'])]
    # census_tracts_gdf = gpd.GeoDataFrame(census_tracts, geometry=geometry, crs=crs)
    # census_tracts_gdf = census_tracts_gdf.to_crs(crs)

    # Match census tracts with grid cells
    matched_gdf = gpd.sjoin(census_tracts_gdf, grid[['geometry', 'region_index']], how='left', op='within')
    matched_gdf = matched_gdf.drop(columns='geometry').fillna(-1).astype({'region_index': 'Int64'})
    matched_gdf = matched_gdf[['geoid', 'region_index']]
    merged_df = pd.merge(matched_gdf, features_df, on='geoid', how='left').fillna(0).astype({'total_population': 'int'})
    grouped_df = merged_df.groupby('region_index').agg(
        region_index=('region_index', 'first'),
        grid_population=('total_population', 'sum'),
        census_tracts_geoids=('geoid', lambda x: ','.join(x.astype(str)))
    ).reset_index(drop=True)
    grouped_df = grouped_df[grouped_df['region_index'] != -1]

    # Sort by population and create stratified splits
    sorted_df = grouped_df.sort_values(by='grid_population', ascending=False)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=1202)
    sorted_df['stratify'] = sorted_df.index % 2
    for train_idx, test_idx in splitter.split(sorted_df, sorted_df['stratify']):
        train_set = sorted_df.iloc[train_idx]
        test_set = sorted_df.iloc[test_idx]

    # Save region indices with corresponding census tracts geoids to CSV files
    train_set.reset_index(inplace=True)
    test_set.reset_index(inplace=True)

    # Create directories if they do not exist
    os.makedirs(f'../processed_data/{experiment_id}/train/', exist_ok=True)
    os.makedirs(f'../processed_data/{experiment_id}/test/', exist_ok=True)
    
    train_set[['region_index']].to_csv(f'../processed_data/{experiment_id}/train/train_region_index.csv', index=False)
    train_set[['region_index']].to_csv(f'../processed_data/{experiment_id}/test/test_region_index.csv', index=False)

    return train_set[['region_index', 'census_tracts_geoids']], test_set[['region_index', 'census_tracts_geoids']]



def filter_train_test_data(flow_df, tessellation_df, features_df, train_set, test_set, experiment_id="0", balance_sets=False):
    """
    Filters and optionally balances the data based on geoid population, then saves the filtered data for flow, tessellation, and features.

    Parameters:
    flow_df (pd.DataFrame): DataFrame with flow data.
    tessellation_df (pd.DataFrame): DataFrame containing tessellation data with geoids.
    features_df (pd.DataFrame): DataFrame containing population information for each geoid.
    train_set (pd.DataFrame): DataFrame with training set geoids.
    test_set (pd.DataFrame): DataFrame with test set geoids.
    experiment_id (str): Identifier for the experiment.
    balance_sets (bool): Flag to determine if the train/test sets should be balanced based on population.

    Returns:
    None: Saves six CSV files for adjusted train and test datasets for flow, tessellation, and features.
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
    train_geoids = {str(geoid) for geoid in train_geoids}
    test_geoids = {str(geoid) for geoid in test_geoids}

    # train_geoids = set(train_set['census_tracts_geoids'].explode())
    # test_geoids = set(test_set['census_tracts_geoids'].explode())

    # Step 2: Get population numbers for the geoids from features
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

    # Step 3: Filter datasets based on adjusted geoid sets
    train_flows = flow_df[flow_df['geoid_o'].isin(train_geoids) & flow_df['geoid_d'].isin(train_geoids)]
    test_flows = flow_df[flow_df['geoid_o'].isin(test_geoids) & flow_df['geoid_d'].isin(test_geoids)]
    train_tessellation = tessellation_df[tessellation_df['GEOID'].isin(train_geoids)]
    test_tessellation = tessellation_df[tessellation_df['GEOID'].isin(test_geoids)]
    train_features = features_df[features_df['geoid'].isin(train_geoids)]
    test_features = features_df[features_df['geoid'].isin(test_geoids)]

    # Step 4: Save all processed files
    # Create directories and save the result

    # Paths for the directories
    train_flows_dir = f'../processed_data/{experiment_id}/train/flows/'
    test_flows_dir = f'../processed_data/{experiment_id}/test/flows/'
    train_tessellation_dir = f'../processed_data/{experiment_id}/train/'
    test_tessellation_dir = f'../processed_data/{experiment_id}/test/'
    train_features_dir = f'../processed_data/{experiment_id}/train/'
    test_features_dir = f'../processed_data/{experiment_id}/test/'

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

    # Save files to their respective directories
    train_flows[['origin', 'destination', 'flow']].to_csv(train_flows_dir + 'train_flow.csv', index=False)
    test_flows[['origin', 'destination', 'flow']].to_csv(test_flows_dir + 'test_flow.csv', index=False)
    train_tessellation.to_file(train_tessellation_dir + 'train_tessellation.geojson', driver='GeoJSON')
    test_tessellation.to_file(test_tessellation_dir + 'test_tessellation.geojson', driver='GeoJSON')
    train_features.to_csv(train_features_dir + 'train_features.csv', index=False)
    test_features.to_csv(test_features_dir + 'test_features.csv', index=False)

    print("Processed and saved all datasets successfully.")