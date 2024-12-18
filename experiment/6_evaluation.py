import argparse
import os
import random
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import concurrent.futures
from pathlib import Path
import csv


sys.path.append(os.path.abspath('../evaluation'))
from eval import *


parser = argparse.ArgumentParser(description="Set folder and model type options.")
parser.add_argument(
    '--steepness_factor',
    type=str,
    default='steep5',
    help='Steepness Factor (default: steep5)'
)
parser.add_argument(
    '--folder_name',
    type=str,
    default='NY',
    help='Name of the folder (default: NY)'
)
parser.add_argument(
    '--model_type',
    type=str,
    default='DG',
    help='Type of the model (default: DG)'
)
parser.add_argument(
    '--dgfolder_name',
    type=str,
    default='new_york',
    help='Name of the DG folder (default: new_york)'
)

args = parser.parse_args()

steepness_factor = args.steepness_factor
folder_name = args.folder_name
folder_name_path = f'{folder_name}_{steepness_factor}'
model_type = args.model_type
dgfolder_name = args.dgfolder_name

print(f"Steepness Factor: {steepness_factor}")
print(f"Folder Name: {folder_name_path}")
print(f"Model Type: {model_type}")
print(f"DG Folder Name: {dgfolder_name}")



demographic_column = 'svi'
performance_metric_list = ['CPC', 'overestimation', 'underestimation']
variance_metric_list = ['kl_divergence']


demographics_path = f'../data/{folder_name}/demographics.csv'


if model_type == 'G':
    for performance_metric in performance_metric_list:
        for variance_metric in variance_metric_list:
            for dirpath, dirname, filenames in os.walk(os.path.join('..', f'gravity_model_{steepness_factor}', 'results', f'{folder_name_path}')):
                for idx, filename in enumerate(filenames):
                    if 'flow' in filename:
                        generated_flows_path = os.path.join(dirpath, filename)
                        flows_path = f'../processed_data/{folder_name_path}/test/test_flow.csv'

                        evaluator = FlowEvaluator(flows_path, generated_flows_path, demographics_path, model_type, folder_name, steepness_factor)
                        evaluator.init_log(performance_metric, variance_metric)

                        # Evaluate unfairness and performance
                        unfairness, performance = evaluator.evaluate_unfairness(
                            performance_metric=performance_metric,
                            variance_metric=variance_metric,
                            demographic_column=demographic_column
                        )

elif model_type == 'DG' or model_type == 'NLG':
    for performance_metric in performance_metric_list:
        for variance_metric in variance_metric_list:
            for i in range(25):
                file_suffix = f'{dgfolder_name}{i}'

                flows_path = f'../processed_data/{folder_name_path}/test/test_flow.csv'
                generated_flows_path = f'../deepgravity_{steepness_factor}/results/predicted_od2flow_{model_type}_{file_suffix}.csv'

                # Initialize evaluator and log
                evaluator = FlowEvaluator(flows_path, generated_flows_path, demographics_path, model_type, folder_name, steepness_factor)
                evaluator.init_log(performance_metric, variance_metric)

                # Evaluate unfairness and performance
                unfairness, performance = evaluator.evaluate_unfairness(
                    performance_metric=performance_metric,
                    variance_metric=variance_metric,
                    demographic_column=demographic_column
                )
else:
    print("Invalid model type")


