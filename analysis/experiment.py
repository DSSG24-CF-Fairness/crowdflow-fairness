import os
import random
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import concurrent.futures
from pathlib import Path
import csv


sys.path.append(os.path.abspath('../preprocessing'))
from train_test_processing import *

# NOTE: choose bias sampling or bias sampling new here
sys.path.append(os.path.abspath('../preprocessing'))
from biased_sampling_new import *

sys.path.append(os.path.abspath('../gravity_model'))
from gravity import *

sys.path.append(os.path.abspath('../evaluation'))
from eval import *

#
#
# folder_name = 'WA'
# flow_df = pd.read_csv(f'../data/{folder_name}/flow.csv')
# tessellation_df = gpd.read_file(f'../data/{folder_name}/tessellation.geojson')
# features_df =pd.read_csv(f'../data/{folder_name}/features.csv')
# demographics_df = pd.read_csv(f'../data/{folder_name}/demographics.csv')
#
#
#
# base_data_path = '../data/WA'
#
# region = load_state_or_county_data(f"{base_data_path}/boundary.geojson")
# features_df = pd.read_csv(f"{base_data_path}/features.csv")
# tessellation_df = load_state_or_county_data(f"{base_data_path}/tessellation.geojson")
# grid = create_grid(region.unary_union, 25)
# train_output, test_output = flow_train_test_split(tessellation_df, features_df, grid, folder_name)
#
#
#
# train_set = pd.read_csv(f'../data/{folder_name}/train_tile_geoids.csv')
# test_set = pd.read_csv(f'../data/{folder_name}/test_tile_geoids.csv')
#
#
#
#
# filter_train_test_data(flow_df, tessellation_df, features_df, train_set, test_set, folder_name, balance_sets=False)

# train_features_df = pd.read_csv(f'../processed_data/{folder_name}/train/train_features.csv')
# train_flows_df = pd.read_csv(f'../processed_data/{folder_name}/train/train_flow.csv')

# random_seeds = [1,2,3,4,5]
# for random_seed in random_seeds:
#     calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=1, order='ascending', steepness_factor=5, random_seed = random_seed)
#     calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=1, order='descending', steepness_factor=5, random_seed = random_seed)
#     calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=2, order='ascending', steepness_factor=5, random_seed = random_seed)
#     calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=2, order='descending', steepness_factor=5, random_seed = random_seed)
#




# # Run model for all train data
# test_flow_path = f"../processed_data/{folder_name}/test/test_flow.csv"
#
# for dirpath, dirname, filenames in os.walk(f'../processed_data/{folder_name}/train'):
#     for idx, filename in enumerate(filenames):
#         if 'flow' in filename:
#             file_path = os.path.join(dirpath, filename)
#             print("Running model:", file_path)
#
#             tessellation_train = gpd.read_file(f"../processed_data/{folder_name}/train/train_tessellation.geojson")
#             tessellation_test = gpd.read_file(f"../processed_data/{folder_name}/test/test_tessellation.geojson")
#
#             grav_Model(tessellation_train, tessellation_test, file_path, test_flow_path, "gravity_singly_constrained", 'flows', folder_name = folder_name)
#
#
#
folder_name = 'WA'
demographic_column = 'svi'
demographics_path = f'../data/{folder_name}/demographics.csv'

accuracy_metric_list = ["CPC", "overestimation", "underestimation"]
variance_metric_list = ['kl_divergence', 'standard_deviation']


# Gravity
model_type = 'G'

for accuracy_metric in accuracy_metric_list:
    for variance_metric in variance_metric_list:
        for dirpath, dirname, filenames in os.walk(f'../gravity_model/results/{folder_name}'):
            for idx, filename in enumerate(filenames):
                if 'flow' in filename:
                    generated_flows_path = os.path.join(dirpath, filename)
                    flows_path = f'../processed_data/{folder_name}/test/test_flow.csv'

                    evaluator = FlowEvaluator(flows_path, generated_flows_path, demographics_path, model_type,
                                              folder_name)
                    evaluator.init_log(accuracy_metric, variance_metric)

                    # Evaluate fairness and accuracy
                    fairness, accuracy = evaluator.evaluate_fairness(
                        accuracy_metric=accuracy_metric,
                        variance_metric=variance_metric,
                        demographic_column=demographic_column
                    )




# # DG/NLG
# model_type = 'NLG'
# dgfolder_name = 'washington'
#
# for accuracy_metric in accuracy_metric_list:
#     for variance_metric in variance_metric_list:
#         for i in range(0, 21):
#             file_suffix = f'{dgfolder_name}{i}'
#
#             flows_path = f'../processed_data/{folder_name}/test/test_flow.csv'
#             generated_flows_path = f'../deepgravity_new_bias/results/predicted_od2flow_{model_type}_{file_suffix}.csv'
#
#             # Initialize evaluator and log
#             evaluator = FlowEvaluator(flows_path, generated_flows_path, demographics_path, model_type, folder_name)
#             evaluator.init_log(accuracy_metric, variance_metric)
#
#             # Evaluate fairness and accuracy
#             fairness, accuracy = evaluator.evaluate_fairness(
#                 accuracy_metric=accuracy_metric,
#                 variance_metric=variance_metric,
#                 demographic_column=demographic_column
#             )