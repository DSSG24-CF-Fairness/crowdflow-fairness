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


folder_name = 'NY'
demographic_column = 'svi'
demographics_path = f'../data/{folder_name}/demographics.csv'

accuracy_metric_list = ["CPC", "overestimation", "underestimation"]
variance_metric_list = ['kl_divergence', 'standard_deviation']


# Gravity
model_type = 'DG'

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
