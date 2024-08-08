import sys
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

experiment_id = '1'

# Split the data into train and test sets

sys.path.append(os.path.abspath('../preprocessing'))
from train_test_processing import *
sys.path.append(os.path.abspath('../preprocessing'))
from train_test_split_vis import *

washington = load_state_or_county_data('../data/WA/boundary.geojson')
flow_df = pd.read_csv('../data/WA/flow.csv')
features_df = pd.read_csv('../data/WA/features.csv')
tessellation_df = load_state_or_county_data('../data/WA/tessellation_wpop.geojson')
grid = create_grid(washington.unary_union, 25)

train_output, test_output = flow_train_test_split(tessellation_df, features_df, grid, experiment_id = experiment_id)

# TODO:
# plot_grid_and_census_tracts(grid, tessellation_df, train_output, test_output)

filter_train_test_data(flow_df, tessellation_df, features_df, train_output, test_output, experiment_id = experiment_id)

# Biased sampling train flow data

sys.path.append(os.path.abspath('../preprocessing'))
from biased_sampling import *

features_df = pd.read_csv('../processed_data/1/train/train_features.csv')
train_flows_df = pd.read_csv('../processed_data/1/train/flows/train_flow.csv')
demographics_df = pd.read_csv('../data/WA/demographics.csv')


calculate_biased_flow(features_df, demographics_df, train_flows_df, demographic_column_name='svi', method=1, order='ascending', sampling=False, experiment_id=experiment_id, bias_factor=0.5)
calculate_biased_flow(features_df, demographics_df, train_flows_df, demographic_column_name='svi', method=1, order='descending', sampling=False, experiment_id=experiment_id, bias_factor=0.5)
calculate_biased_flow(features_df, demographics_df, train_flows_df, demographic_column_name='svi', method=2, order='ascending', sampling=False, experiment_id=experiment_id, bias_factor=0.5)
calculate_biased_flow(features_df, demographics_df, train_flows_df, demographic_column_name='svi', method=2, order='descending', sampling=False, experiment_id=experiment_id, bias_factor=0.5)

# Fit gravity model

sys.path.append(os.path.abspath('../models'))
from gravity import *

tessellation_train = gpd.read_file('../processed_data/1/train/train_tessellation.geojson')
tessellation_test = gpd.read_file('../processed_data/1/test/test_tessellation.geojson')

# gravity_0 = grav_Model(tessellation_train, tessellation_test, '../processed_data/1/train/flows/train_flow.csv', '../processed_data/1/test/flows/test_flow.csv', 'gravity_singly_constrained', 'flows')

# gravity_1 = grav_Model(tessellation_train, tessellation_test, '../processed_data/1/train/flows/svi/1_ascending_biased_flow.csv', '../processed_data/1/test/flows/test_flow.csv', 'gravity_singly_constrained', 'flows')

gravity_2 = grav_Model(tessellation_train, tessellation_test, '../processed_data/1/train/flows/svi/1_descending_biased_flow.csv', '../processed_data/1/test/flows/test_flow.csv', 'gravity_singly_constrained', 'flows')

# gravity_3 = grav_Model(tessellation_train, tessellation_test, '../processed_data/1/train/flows/svi/2_ascending_biased_flow.csv', '../processed_data/1/test/flows/test_flow.csv', 'gravity_singly_constrained', 'flows')

# gravity_4 = grav_Model(tessellation_train, tessellation_test, '../processed_data/1/train/flows/svi/2_descending_biased_flow.csv', '../processed_data/1/test/flows/test_flow.csv', 'gravity_singly_constrained', 'flows')



