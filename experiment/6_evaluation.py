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

sys.path.append(os.path.abspath('../evaluation'))
from eval_plot import *


parser = argparse.ArgumentParser(description="Set folder and model type options.")
parser.add_argument(
    '--folder_name',
    type=str,
    default='NY_NEW',
    help='Name of the folder (default: NY_NEW)'
)
parser.add_argument(
    '--model_type',
    type=str,
    default='G',
    help='Type of the model (default: G)'
)
parser.add_argument(
    '--dgfolder_name',
    type=str,
    default='new_york_new',
    help='Name of the DG folder (default: new_york_new)'
)

args = parser.parse_args()

folder_name = args.folder_name
model_type = args.model_type
dgfolder_name = args.dgfolder_name

print(f"Folder Name: {folder_name}")
print(f"Model Type: {model_type}")
print(f"DG Folder Name: {dgfolder_name}")


# folder_name = 'NY_NEW'
demographic_column = 'svi'
accuracy_metric_list = ['CPC']
variance_metric_list = ['kl_divergence', 'standard_deviation']
# model_type = 'G'
# dgfolder_name = 'new_york_new'

demographics_path = f'../data/{folder_name}/demographics.csv'


if model_type == 'G':
    for accuracy_metric in accuracy_metric_list:
        for variance_metric in variance_metric_list:
            for dirpath, dirname, filenames in os.walk(f'../gravity_model/results/{folder_name}'):
                for idx, filename in enumerate(filenames):
                    if 'flow' in filename:
                        generated_flows_path = os.path.join(dirpath, filename)
                        flows_path = f'../processed_data/{folder_name}/test/test_flow.csv'

                        evaluator = FlowEvaluator(flows_path, generated_flows_path, demographics_path, model_type, folder_name)
                        evaluator.init_log(accuracy_metric, variance_metric)

                        # Evaluate fairness and accuracy
                        fairness, accuracy = evaluator.evaluate_fairness(
                            accuracy_metric=accuracy_metric,
                            variance_metric=variance_metric,
                            demographic_column=demographic_column
                        )

elif model_type == 'DG' or model_type == 'NLG':
    for accuracy_metric in accuracy_metric_list:
        for variance_metric in variance_metric_list:
            for i in range(0, 21):
                file_suffix = f'{dgfolder_name}{i}'

                flows_path = f'../processed_data/{folder_name}/test/test_flow.csv'
                generated_flows_path = f'../deepgravity_new_bias/results/predicted_od2flow_{model_type}_{file_suffix}.csv'

                # Initialize evaluator and log
                evaluator = FlowEvaluator(flows_path, generated_flows_path, demographics_path, model_type, folder_name)
                evaluator.init_log(accuracy_metric, variance_metric)

                # Evaluate fairness and accuracy
                fairness, accuracy = evaluator.evaluate_fairness(
                    accuracy_metric=accuracy_metric,
                    variance_metric=variance_metric,
                    demographic_column=demographic_column
                )
else:
    print("Invalid model type")


# location_name = 'NY_NEW'
# accuracy_type = 'CPC'
# metric_type = 'kl_divergence'
# plot_fairness_vs_accuracy(location_name, accuracy_type, metric_type)