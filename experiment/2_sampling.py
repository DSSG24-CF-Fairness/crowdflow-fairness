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
from biased_sampling import *

folder_name = 'WA'

demographics_df = pd.read_csv(f'../data/{folder_name}/demographics.csv')
train_features_df = pd.read_csv(f'../processed_data/{folder_name}/train/train_features.csv')
train_flows_df = pd.read_csv(f'../processed_data/{folder_name}/train/train_flow.csv')

calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=2, order='descending', steepness_factor=5, sampling=True, random_seed=6)


'''
calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=1, order='ascending', steepness_factor=20, sampling=False)
calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=1, order='descending', steepness_factor=20, sampling=False)
calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=2, order='ascending', steepness_factor=20, sampling=False)
calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=2, order='descending', steepness_factor=20, sampling=False)

random_seeds = [1,2,3,4,5]
for random_seed in random_seeds:
    calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=1, order='ascending', steepness_factor=20, sampling = True, random_seed = random_seed)
    calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=1, order='descending', steepness_factor=20, sampling = True, random_seed = random_seed)
    calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=2, order='ascending', steepness_factor=20, sampling = True, random_seed = random_seed)
    calculate_biased_flow(train_features_df, demographics_df, train_flows_df, folder_name, demographic_column_name='svi', method=2, order='descending', steepness_factor=20, sampling = True, random_seed = random_seed)


'''