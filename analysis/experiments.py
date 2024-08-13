import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import concurrent.futures
from pathlib import Path

sys.path.append(os.path.abspath('../preprocessing'))
from train_test_processing import *
from biased_sampling import *
from train_test_split_vis import *

sys.path.append(os.path.abspath('../models'))
from gravity import *
from random import *

sys.path.append(os.path.abspath('../evaluation'))
from eval import FlowEvaluator

class ExperimentRunner:
    def __init__(self, demographic_columns, base_data_path, output_dir, experiment_id, num_bias_samples, model_name):
        """
        Initialize the experiment runner with the necessary configurations.

        Parameters:
        base_data_path (str): Base directory path where the data resides.
        output_dir (str): Directory to save output files and visualizations.
        experiment_id (str): Unique identifier for the experiment.
        """
        self.demographic_columns = demographic_columns
        self.base_data_path = base_data_path
        self.output_dir = output_dir
        self.experiment_id = experiment_id
        self.results_df = pd.DataFrame()
        self.num_bias_samples = num_bias_samples
        self.model_name = model_name

    def create_train_test_split(self):

        region = load_state_or_county_data(f"{self.base_data_path}/boundary.geojson")
        flow_df = pd.read_csv(f"{self.base_data_path}/flow.csv")
        features_df = pd.read_csv(f"{self.base_data_path}/features.csv")
        tessellation_df = load_state_or_county_data(f"{self.base_data_path}/tessellation.geojson")
        grid = create_grid(region.unary_union, 25)

        train_output, test_output = flow_train_test_split(tessellation_df, features_df, grid, experiment_id = self.experiment_id)
        plot_grid_and_census_tracts(tessellation_df, grid, train_output, test_output, experiment_id = self.experiment_id)
        filter_train_test_data(flow_df, tessellation_df, features_df, train_output, test_output, experiment_id = self.experiment_id)


    def create_biased_samples(self, demographic_column, demographics_df, method, order, sampling = False, sample_id = None):
        features_df = pd.read_csv(f"../processed_data/{self.experiment_id}/train/train_features.csv")
        train_flows_df = pd.read_csv(f"../processed_data/{self.experiment_id}/train/flows/train_flow.csv")

        calculate_biased_flow(features_df, demographics_df, train_flows_df, demographic_column_name=demographic_column, method=method, order=order, sampling=sampling, sample_id=sample_id, experiment_id=self.experiment_id, bias_factor=0.5)


    def run_model(self, train_flow_path, test_flow_path):
        tessellation_train = gpd.read_file(f"../processed_data/{self.experiment_id}/train/train_tessellation.geojson")
        tessellation_test = gpd.read_file(f"../processed_data/{self.experiment_id}/test/test_tessellation.geojson")

        if self.model_name == 'gravity_singly_constrained':
            grav_Model(tessellation_train, tessellation_test, train_flow_path, test_flow_path, "gravity_singly_constrained", 'flows')
        elif self.model_name == 'random':
            generate_random_results(test_flow_path, tessellation_test, self.num_bias_samples*4+1, self.experiment_id)
        else:
            raise ValueError(f"Invalid model name: {self.model_name}.")


    def evaluate_results(self):
        real_flow_path = f'../processed_data/{self.experiment_id}/test/flows/test_flow.csv'
        generated_flow_dir = f'..outputs/{self.experiment_id}/synthetic_data_{self.model_name}/{self.demographic_columns}'
        demographics_path = '../data/WA/demographics.csv'

        generated_flow_filenames = os.listdir(generated_flow_dir)
        generated_flow_filenames = [x for x in generated_flow_filenames if x.endswith('.csv')]

        results = []
        for filename in generated_flow_filenames:
            generated_flow_path = os.path.join(generated_flow_dir, filename)
            evaluator = FlowEvaluator(real_flow_path, generated_flow_path, demographics_path)
            fairness, performance= evaluator.evaluate_fairness(accuracy_metric='CPC', variance_metric='kl_divergence', demographic_column=self.demographic_columns)
            print(f'{filename} Fairness Metric (Variance of Accuracy): fairness is {fairness}, mean accuracy is {performance}.')
            results.append({
                'filename': generated_flow_path,
                'model': self.model_name,
                'demographic': self.demographic_columns,
                'fairness': fairness,
                'accuracy': performance
            })
        results_df = pd.DataFrame(results)
        save_path = os.path.join(self.output_dir, f'{self.experiment_id}', 'results', 'results.csv')
        results_df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}.")

    
    def run_all(self):
        # Split train and test data
        self.create_train_test_split()

        # Check if demographic columns exist in the demographic.csv file at given data path
        demographics_path = f"{self.base_data_path}/demographics.csv"
        demographics_df = pd.read_csv(demographics_path)

        # Create biased samples based on given demographic_columns
        for method in [1,2]:
            for order in ['ascending', 'descending']:
                for sample_id in range(self.num_bias_samples):
                    self.create_biased_samples(self.demographic_columns, demographics_df, method, order, sampling=True, sample_id=sample_id)

        # Run model for all train data
        test_flow_path = f"../processed_data/{self.experiment_id}/test/flows/test_flow.csv"

        for dirpath, dirname, filenames in os.walk(f'../processed_data/{self.experiment_id}/train/flows'):
            for idx, filename in enumerate(filenames):
                if filename.endswith('.csv'):
                    # Construct the full file path
                    file_path = os.path.join(dirpath, filename)
                    # You can now open and process each CSV file as needed
                    print("Running model:", file_path)
                    self.run_model(file_path, test_flow_path)
                    
        # # **INCLUDE IF YOU WANT FOR MODEL**
        # # Run model for all train data (execute parallely)

        # # Construct the path to search for files
        # base_path = f'../processed_data/{self.experiment_id}/train/flows'
        # csv_files = [os.path.join(dirpath, filename)
        #              for dirpath, _, filenames in os.walk(base_path)
        #              for filename in filenames if filename.endswith('.csv')]
        #
        # # Using a process pool to execute tasks concurrently
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     # Use a lambda to include the static test_flow_path with each task
        #     executor.map(lambda file_path: self.run_model(file_path, test_flow_path), csv_files)

        # Evaluation
        self.evaluate_results()



def main():

    base_data_path = '../data/WA'
    output_dir = '../outputs'
    experiment_id = 2
    demographic_columns = 'svi'
    num_bias_samples = 5
    model_name = 'gravity_singly_constrained'

    runner = ExperimentRunner(demographic_columns, base_data_path, output_dir, experiment_id, num_bias_samples, model_name)
    runner.run_all()


if __name__ == "__main__":
    main()


