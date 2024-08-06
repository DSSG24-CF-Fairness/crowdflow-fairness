import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box, Point
from sklearn.model_selection import StratifiedShuffleSplit



import sys
import os

sys.path.append(os.path.abspath('../preprocessing'))
import train_test_processing


def plot_grid_and_census_tracts(grid, census_tracts, train_set, test_set):
    """
    Plot the grid and census tracts tessellation for a particular year, coloring them according to train/test.
    """
    column_name = 'GEOID'

    train_geoids = set()
    test_geoids = set()

    for geoids in train_set['census_tracts_geoids']:
        train_geoids.update(geoids.split(','))
    for geoids in test_set['census_tracts_geoids']:
        test_geoids.update(geoids.split(','))

    # Assign colors based on whether the geoid is in the train or test set
    def assign_color(geoid):
        if geoid in train_geoids:
            return 'green'
        elif geoid in test_geoids:
            return 'red'
        else:
            return 'grey'

    census_tracts['color'] = census_tracts[column_name].apply(assign_color)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot census tracts first (bottom layer)
    census_tracts.plot(ax=ax, color=census_tracts['color'], edgecolor='black', linewidth=0.5)

    # Plot grid cells on top with dotted lines
    grid.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, linestyle='--', label='Grid Cells')

    # Create a custom legend
    handles = [
        plt.Line2D([0], [0], color='green', lw=2, label='Train'),
        plt.Line2D([0], [0], color='red', lw=2, label='Test'),
        plt.Line2D([0], [0], color='grey', lw=2, label='None'),
        plt.Line2D([0], [0], color='black', linestyle='--', lw=1, label='Grid Cells')
    ]
    ax.legend(handles=handles)

    plt.title('Grid and Census Tracts Tessellation')
    plt.show()



