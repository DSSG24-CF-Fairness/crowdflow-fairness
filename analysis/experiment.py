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



folder_name = 'WA'
flow_df = pd.read_csv(f'../data/{folder_name}/flow.csv')
tessellation_df = gpd.read_file(f'../data/{folder_name}/tessellation.geojson')
features_df =pd.read_csv(f'../data/{folder_name}/features.csv')
test_set = pd.read_csv(f'../data/{folder_name}/test_tile_geoids.csv')
train_set = pd.read_csv(f'../data/{folder_name}/train_tile_geoids.csv')
demographics_df = pd.read_csv(f'../data/{folder_name}/demographics.csv')




# filter_train_test_data(flow_df, tessellation_df, features_df, train_set, test_set, folder_name, balance_sets=False)
#
# train_features_df = pd.read_csv(f'../processed_data/{folder_name}/train/train_features.csv')
# train_flows_df = pd.read_csv(f'../processed_data/{folder_name}/train/train_flow.csv')
#
# random_seeds = [1,2,3,4,5]
# for random_seed in random_seeds:
#     calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=1, order='ascending', sampling=True, bias_factor=0.5, random_seed = random_seed)
#     calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=1, order='descending', sampling=True, bias_factor=0.5, random_seed = random_seed)
#     calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=2, order='ascending', sampling=True, bias_factor=0.5, random_seed = random_seed)
#     calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=2, order='descending', sampling=True, bias_factor=0.5, random_seed = random_seed)




# Run model for all train data
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



folder_name = 'NY'
demographic_column = 'svi'
features_path = f'../data/{folder_name}/demographics.csv'


# Gravity
# model_type = 'G'
# log_path = f'../evaluation/{folder_name}_{model_type}_log.csv'
#
# with open(log_path, mode='w', newline='') as log_file:
#     log_writer = csv.writer(log_file)
#     log_writer.writerow(['file_name', 'fairness', 'accuracy'])
#
# for dirpath, dirname, filenames in os.walk(f'../gravity_model/results/{folder_name}'):
#     for idx, filename in enumerate(filenames):
#         if 'flow' in filename:
#             generated_flows_path = os.path.join(dirpath, filename)
#             flows_path = f'../processed_data/{folder_name}/test/test_flow.csv'
#
#             evaluator = FlowEvaluator(flows_path, generated_flows_path, features_path, model_type, folder_name)
#
#             # Evaluate fairness and accuracy
#             fairness, accuracy = evaluator.evaluate_fairness(
#                 accuracy_metric='CPC',
#                 variance_metric='kl_divergence',
#                 demographic_column=demographic_column
#             )
#
#             # Append fairness results to the log CSV file
#             with open(log_path, mode='a', newline='') as log_file:
#                 log_writer = csv.writer(log_file)
#                 log_writer.writerow([filename, fairness, accuracy])
#
#             print(f"Fairness results for {filename} logged to '{folder_name}_{model_type}_log.csv'")




# DG
# model_type = 'DG'
# log_path = f'../evaluation/{folder_name}_{model_type}_log.csv'

# dgfolder_name = 'new_york'
# with open(log_path, mode='w', newline='') as log_file:
#     log_writer = csv.writer(log_file)
#     log_writer.writerow(['file_name', 'fairness', 'accuracy'])
#
# # Loop over the range 0 to 21
# for i in range(0, 21):
#     file_suffix = f'{dgfolder_name}{i}'
#
#     flows_path = f'../processed_data/{folder_name}/test/test_flow.csv'
#     generated_flows_path = f'../deepgravity/results/predicted_od2flow_{model_type}_{file_suffix}.csv'
#
#     # Initialize evaluator for the current file set
#     evaluator = FlowEvaluator(flows_path, generated_flows_path, features_path, model_type, folder_name)
#
#     # Evaluate fairness and accuracy
#     fairness, accuracy = evaluator.evaluate_fairness(
#         accuracy_metric='CPC',
#         variance_metric='kl_divergence',
#         demographic_column=demographic_column
#     )
#
#     # Append fairness results to the log CSV file
#     with open(log_path, mode='a', newline='') as log_file:
#         log_writer = csv.writer(log_file)
#         log_writer.writerow([file_suffix, fairness, accuracy])
#
#     print(f"Fairness results for {file_suffix} logged to '{folder_name}_{model_type}_log.csv'")


# NLG
model_type = 'NLG'
log_path = f'../evaluation/{folder_name}_{model_type}_log.csv'

nlgfolder_name = 'new_york'
with open(log_path, mode='w', newline='') as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow(['file_name', 'fairness', 'accuracy'])

# Loop over the range 0 to 21
for i in range(0, 21):
    file_suffix = f'{nlgfolder_name}{i}'

    flows_path = f'../processed_data/{folder_name}/test/test_flow.csv'
    generated_flows_path = f'../deepgravity/results/predicted_od2flow_{model_type}_{file_suffix}.csv'

    # Initialize evaluator for the current file set
    evaluator = FlowEvaluator(flows_path, generated_flows_path, features_path, model_type, folder_name)

    # Evaluate fairness and accuracy
    fairness, accuracy = evaluator.evaluate_fairness(
        accuracy_metric='CPC',
        variance_metric='kl_divergence',
        demographic_column=demographic_column
    )

    # Append fairness results to the log CSV file
    with open(log_path, mode='a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([file_suffix, fairness, accuracy])

    print(f"Fairness results for {file_suffix} logged to '{folder_name}_{model_type}_log.csv'")