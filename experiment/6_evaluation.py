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
    '--folder_name',
    type=str,
    default='NY',
    help='Name of the folder (default: NY_NEW)'
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
    help='Name of the DG folder (default: new_york_new)'
)

args = parser.parse_args()

folder_name = args.folder_name
model_type = args.model_type
dgfolder_name = args.dgfolder_name

print(f"Folder Name: {folder_name}")
print(f"Model Type: {model_type}")
print(f"DG Folder Name: {dgfolder_name}")



demographic_column = 'svi'
performance_metric_list = ['CPC','overestimation','underestimation']
variance_metric_list = ['kl_divergence']


demographics_path = f'../data/{folder_name}/demographics.csv'


if model_type == 'G':
    for performance_metric in performance_metric_list:
        for variance_metric in variance_metric_list:
            for dirpath, dirname, filenames in os.walk(f'../gravity_model_steep20/results/{folder_name}'):
                for idx, filename in enumerate(filenames):
                    if 'flow' in filename:
                        generated_flows_path = os.path.join(dirpath, filename)
                        flows_path = f'../processed_data/{folder_name}/test/test_flow.csv'

                        evaluator = FlowEvaluator(flows_path, generated_flows_path, demographics_path, model_type, folder_name)
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

                flows_path = f'../processed_data/{folder_name}/test/test_flow.csv'
                generated_flows_path = f'../deepgravity_steep20/results/predicted_od2flow_{model_type}_{file_suffix}.csv'

                # Initialize evaluator and log
                evaluator = FlowEvaluator(flows_path, generated_flows_path, demographics_path, model_type, folder_name)
                evaluator.init_log(performance_metric, variance_metric)

                # Evaluate unfairness and performance
                unfairness, performance = evaluator.evaluate_unfairness(
                    performance_metric=performance_metric,
                    variance_metric=variance_metric,
                    demographic_column=demographic_column
                )
else:
    print("Invalid model type")


