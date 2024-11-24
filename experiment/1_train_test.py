import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

sys.path.append(os.path.abspath('../preprocessing'))
from train_test_processing import *

folder_name = 'WA_NEW'

geojson_path = f"../data/{folder_name}/boundary.geojson"
flow_df = pd.read_csv(f"../data/{folder_name}/flow.csv")
tessellation_df = gpd.read_file(f"../data/{folder_name}/tessellation.geojson")
features_df = pd.read_csv(f"../data/{folder_name}/features.csv")

cell_size_km = 25
grid = create_grid(geojson_path, cell_size_km, folder_name)

train_set, test_set = flow_train_test_split(tessellation_df, features_df, grid, folder_name)
filter_train_test_data(flow_df, tessellation_df, features_df, train_set, test_set, folder_name, balance_sets=False)

plot_grid_and_census_tracts(grid, tessellation_df, train_set, test_set, folder_name)