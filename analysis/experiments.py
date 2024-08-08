import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../preprocessing'))
from train_test_processing import *
from biased_sampling import *

sys.path.append(os.path.abspath('../models'))
from gravity import *

sys.path.append(os.path.abspath('../evaluation'))
from eval import FlowEvaluator

class ExperimentRunner:
    def __init__(self, demographic_columns, base_data_path, output_dir, experiment_id):
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

    def setup_directories(self):
        """ Ensure all necessary directories exist """
        os.makedirs(f"{self.output_dir}/{self.experiment_id}/train", exist_ok=True)
        os.makedirs(f"{self.output_dir}/{self.experiment_id}/test", exist_ok=True)
        os.makedirs(f"{self.output_dir}/{self.experiment_id}/results", exist_ok=True)

    def create_train_test_split(self):

        washington = load_state_or_county_data(f"{self.base_data_path}/boundary.geojson")
        flow_df = pd.read_csv(f"{self.base_data_path}/flow.csv")
        features_df = pd.read_csv(f"{self.base_data_path}/features.csv")
        tessellation_df = load_state_or_county_data(f"{self.base_data_path}/tessellation_wpop.geojson")
        grid = create_grid(washington.unary_union, 25)

        # Split the data into train and test sets
        train_output, test_output = flow_train_test_split(tessellation_df, features_df, grid, experiment_id = self.experiment_id)

        filter_train_test_data(flow_df, tessellation_df, features_df, train_output, test_output, experiment_id = self.experiment_id)
  
    def create_biased_samples(self, demographic_column, method, order, sampling = False):
        features_df = pd.read_csv(f"../processed_data/{self.experiment_id}/train/train_features.csv")
        train_flows_df = pd.read_csv(f"../processed_data/{self.experiment_id}/train/flows/train_flow.csv")
        demographics_df = pd.read_csv("{self.base_data_path}/demographics.csv")

        calculate_biased_flow(features_df, demographics_df, train_flows_df, demographic_column_name=demographic_column, method=method, order=order, sampling=sampling, experiment_id=self.experiment_id, bias_factor=0.5)

    def run_model(self,train_flow_path,test_flow_path):

        tessellation_train = gpd.read_file(f"../processed_data/{self.experiment_id}/train/train_tessellation.geojson")
        tessellation_test = gpd.read_file(f"../processed_data/{self.experiment_id}/test/test_tessellation.geojson")
        

        grav_Model(tessellation_train, tessellation_test, train_flow_path,test_flow_path, "gravity_singly_constrained", 'flows', experiment_id=experiment_id)


    def evaluate_results(self):
        
        flows_path = '../processed_data/1/train/flows/train_flow.csv'
        generated_flows_path = '..outputs/1/synthetic_flows.csv' # TODO change accordingly
        demographics_path = '../data/WA/demographics.csv'
        demographic_column = 'svi'

        evaluator = FlowEvaluator(flows_path, generated_flows_path, demographics_path)
        fairness = evaluator.evaluate_fairness(accuracy_metric='CPC', variance_metric='kl_divergence', demographic_column=demographic_column)
        print(f'Fairness Metric (Variance of Accuracy): {fairness}')
        
        self.save_results() #Do this inside this function
    
    def run_all(self):
        self.create_train_test_split()

        # Check if demographic columns exist in the demographic.csv file at given data path
        demographics_df = pd.read_csv(f"{self.base_data_path}/demographics.csv")
        # TODO Write code to check columns from self.demographic_columns here

        # Create biased samples based on given demographic_columns
        for demographic in self.demographic_columns:
            for method in [1,2]:
                for order in ['ascending','descending']:
                    self.create_biased_samples(demographic, method, order)
        
        test_flow_path = f"../processed_data/{self.experiment_id}/test/flows/test_flow.csv"
        # Run model for unbiased data
        self.run_model(f"../processed_data/{self.experiment_id}/train/flows/train_flow.csv")

        # Run model for biased data
        for demographic in self.demographic_columns:
            for method in [1,2]:
                for order in ['ascending','descending']:

        




def main():
    if len(sys.argv) < 4:
        print("Usage: python script_name.py <base_data_path> <output_dir> <experiment_id>")
        return

    base_data_path = sys.argv[1]
    output_dir = sys.argv[2]
    experiment_id = sys.argv[3]
    demographic_columns = ['svi']

    runner = ExperimentRunner(demographic_columns, base_data_path, output_dir, experiment_id)
    runner.run_all()


if __name__ == "__main__":
    main()


