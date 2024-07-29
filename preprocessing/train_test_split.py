import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
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


def create_grid(polygon, cell_size_km, crs):
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


def train_test_split(flow_df, population_df, washington_crs, grid):
    """
    Mapping of census tracts to grid cells based on population and splitting into train and test sets using stratified shuffle split.

    Parameters:
    flow_df (pd.DataFrame): DataFrame containing origin and destination flow data with geoid, longitude, and latitude.
    population_df (pd.DataFrame): DataFrame containing population data for census tracts with geoid and total population.
    washington_crs (str): The coordinate reference system for the geographic data.
    grid (geopandas.GeoDataFrame): GeoDataFrame containing the grid cells with geometry and area_index.

    Returns:
    tuple: Two DataFrames for train and test sets containing grid indices and census tract geoids.
    .csv: Two .csv files are saved for train and test region indices.
    """

    # Create DataFrame for census tracts with geoids, longitudes, and latitudes
    geo_o = flow_df[['geoid_o', 'lng_o', 'lat_o']].rename(columns={'geoid_o': 'geoid', 'lng_o': 'lng', 'lat_o': 'lat'})
    geo_d = flow_df[['geoid_d', 'lng_d', 'lat_d']].rename(columns={'geoid_d': 'geoid', 'lng_d': 'lng', 'lat_d': 'lat'})
    census_tracts = pd.concat([geo_o, geo_d]).drop_duplicates().reset_index(drop=True)

    # Create GeoDataFrame for census tracts
    geometry = [Point(xy) for xy in zip(census_tracts['lng'], census_tracts['lat'])]
    census_tracts_gdf = gpd.GeoDataFrame(census_tracts, geometry=geometry, crs=washington_crs)
    census_tracts_gdf = census_tracts_gdf.to_crs(washington_crs)

    # Match census tracts with grid cells
    matched_gdf = gpd.sjoin(census_tracts_gdf, grid[['geometry', 'region_index']], how='left', op='within')
    matched_gdf = matched_gdf.drop(columns='geometry').fillna(-1).astype({'region_index': 'Int64'})
    matched_gdf = matched_gdf[['geoid', 'region_index']]
    merged_df = pd.merge(matched_gdf, population_df, on='geoid', how='left').fillna(0).astype({'total_population': 'int'})
    grouped_df = merged_df.groupby('region_index').agg(
        region_index=('region_index', 'first'),
        grid_population=('total_population', 'sum'),
        census_tracts_geoids=('geoid', lambda x: ','.join(x.astype(str)))
    ).reset_index(drop=True)
    grouped_df = grouped_df[grouped_df['region_index'] != -1]

    # Sort by population and create stratified splits
    sorted_df = grouped_df.sort_values(by='grid_population', ascending=False)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    sorted_df['stratify'] = sorted_df.index % 2
    for train_idx, test_idx in splitter.split(sorted_df, sorted_df['stratify']):
        train_set = sorted_df.iloc[train_idx]
        test_set = sorted_df.iloc[test_idx]

    # Save region indices with corresponding census tracts geoids to CSV files
    train_set.reset_index(inplace=True)
    test_set.reset_index(inplace=True)
    train_set[['region_index']].to_csv('../data/train_region_index.csv', index=False)
    train_set[['region_index']].to_csv('../data/test_region_index.csv', index=False)

    return train_set[['region_index', 'census_tracts_geoids']], test_set[['region_index', 'census_tracts_geoids']]


def filter_flow_data(flow_df, population_df, train_set, test_set):
    """
    Filters flow data to include only geoids present in the specified training and test sets.
    Saves the filtered flow data to CSV files and returns DataFrames containing the filtered data along with the count of unique geoids.

    Parameters:
    flow_df (pd.DataFrame): DataFrame with flow data, including origin and destination geoids and population flows.
    population_df (pd.DataFrame): DataFrame containing population information for each geoid.
    train_set (pd.DataFrame): DataFrame with training set information, including grid indices and census tract geoids.
    test_set (pd.DataFrame): DataFrame with test set information, including grid indices and census tract geoids.

    Returns:
    pd.DataFrame: Two DataFrames containing the filtered and adjusted training and test flow data.
    .csv: Two .csv files are saved for filtered training and test flow data.
    """

    train_geoids = set()
    test_geoids = set()

    for geoids in train_set['census_tracts_geoids']:
        train_geoids.update(geoids.split(','))
    for geoids in test_set['census_tracts_geoids']:
        test_geoids.update(geoids.split(','))

    train_geoids = {str(geoid) for geoid in train_geoids}
    test_geoids = {str(geoid) for geoid in test_geoids}

    # Filter the flow data
    train_flows = flow_df[
        flow_df['geoid_o'].astype(str).isin(train_geoids) & flow_df['geoid_d'].astype(str).isin(train_geoids)]
    test_flows = flow_df[
        flow_df['geoid_o'].astype(str).isin(test_geoids) & flow_df['geoid_d'].astype(str).isin(test_geoids)]

    # Merge the population information into train and test sets
    train_with_pop = train_flows.merge(population_df, left_on='geoid_o', right_on='geoid', how='left')
    train_pop_merged = train_with_pop.groupby('geoid_o')['total_population'].mean()
    test_with_pop = test_flows.merge(population_df, left_on='geoid_o', right_on='geoid', how='left')
    test_pop_merged = test_with_pop.groupby('geoid_o')['total_population'].mean()

    # Determine the total number of geoid_o in train/test set
    num_geoid_count = min(len(train_pop_merged), len(test_pop_merged))

    # Adjust geoid number in train/test set sorting by lowest population
    train_geoid_sorted = train_pop_merged.sort_values().head(num_geoid_count).index
    test_geoid_sorted = test_pop_merged.sort_values().head(num_geoid_count).index
    train_set_adjusted = train_flows[train_flows['geoid_o'].isin(train_geoid_sorted)]
    test_set_adjusted = test_flows[test_flows['geoid_o'].isin(test_geoid_sorted)]

    # Save the filtered flow data to CSV files
    train_set_adjusted[['geoid_o', 'geoid_d', 'pop_flows']].to_csv('../data/train_flows.csv', index=False)
    test_set_adjusted[['geoid_o', 'geoid_d', 'pop_flows']].to_csv('../data/test_flows.csv', index=False)

    return train_set_adjusted[['geoid_o', 'geoid_d', 'pop_flows']], test_set_adjusted[['geoid_o', 'geoid_d', 'pop_flows']], num_geoid_count


def main():
    washington = load_state_or_county_data('../data/2016_wa_pumas.geojson')
    flow_df = pd.read_csv('../data/washington_flow.csv')
    population_df = pd.read_csv('../data/washington_census_tracts_population.csv')

    grid = create_grid(washington.unary_union, 25, washington.crs)

    # Plot the grid
    fig, ax = plt.subplots(figsize=(10, 10))
    grid.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1)  # Plot Seattle boundary
    grid.boundary.plot(ax=ax, edgecolor='red', linewidth=0.5)  # Plot grid cell boundaries
    plt.show()

    # Split the data into train and test sets
    train_output, test_output = train_test_split(flow_df, population_df, washington.crs, grid)
    train_set_flows, test_set_flows, num_geoid_count = filter_flow_data(flow_df, population_df, train_output, test_output)

    print(train_set_flows, test_set_flows, num_geoid_count)



if __name__ == "__main__":
    main()
