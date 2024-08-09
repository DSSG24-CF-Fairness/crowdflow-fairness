import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pyproj
from shapely.geometry import box, Point, Polygon, MultiPolygon
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.patches as patches

sys.path.append(os.path.abspath('../preprocessing'))
import train_test_processing


def plot_grid_and_census_tracts(tessellation_df, grid_df, train_set, test_set, experiment_id='0'):
    """
    The function creates a plot to visualize census tracts and grid cells.
    Census tracts are colored based on their membership in the training or test set, with centroids displayed as small dots.
    Train census tracts are highlighted in red, while test census tracts are highlighted in green.
    All other tracts are shown in grey.
    Grid cells are outlined in blue lines.

    Parameters:
    tessellation_df (geopandas.GeoDataFrame): GeoDataFrame with census tracts geometry and centroid coordinates.
    grid_df (geopandas.GeoDataFrame): GeoDataFrame with grid cells geometry.
    train_set (pd.DataFrame): DataFrame with training set geoids.
    test_set (pd.DataFrame): DataFrame with test set geoids.
    experiment_id (str): Identifier for the experiment.

    Return:
    Plot census tracts with grid, highlighting train and test areas with centroids.
    """
    train_geoids = set(train_set['census_tracts_geoids'].str.split(',').explode())
    test_geoids = set(test_set['census_tracts_geoids'].str.split(',').explode())

    def get_color(geoid):
        if geoid in train_geoids:
            return 'red'
        elif geoid in test_geoids:
            return 'green'
        else:
            return 'grey'
    tessellation_df['color'] = tessellation_df['GEOID'].astype(str).apply(get_color)

    fig, ax = plt.subplots(figsize=(24, 20))
    grid_df.plot(ax=ax, edgecolor='blue', linestyle='dashdot', alpha=1, facecolor='none', linewidth=0.1)

    sample_size = len(tessellation_df['lng'])
    drawing_details = 10
    x = np.array(tessellation_df['lng'])[:sample_size]
    y = np.array(tessellation_df['lat'])[:sample_size]
    census_color = np.array(tessellation_df['color'])[:sample_size]
    area = list(tessellation_df['geometry'])[:sample_size]
    ax.scatter(x, y, color=census_color, s=1.0, alpha=1.0, label='Centroids')

    for idx, element in enumerate(area):
        if type(element) == Polygon:
            polygon = element
            xs, ys = polygon.exterior.xy
            sampledx = []
            sampledy = []
            for i in range(0, len(xs), drawing_details):
                sampledx.append(xs[i])
                sampledy.append(ys[i])
            ax.fill(sampledx, sampledy, color=census_color[idx], alpha=0.5)
        else:
            multipolygon = list(element.geoms)
            sampledx = []
            sampledy = []
            for polygon in multipolygon:
                xs, ys = polygon.exterior.xy
                for i in range (0, len(xs), drawing_details):
                    sampledx.append(xs[i])
                    sampledy.append(ys[i])
            ax.fill(sampledx, sampledy, color = census_color[idx], alpha= 0.5)

    ax.set_title('Train-Test Split Visualization')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    handles = [
        patches.Patch(color='green', label='Train', linewidth=1),
        patches.Patch(color='red', label='Test', linewidth=1)
    ]
    ax.legend(handles=handles)

    dir_path = f'../processed_data/{experiment_id}/'
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, 'train_test_split_plot.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()