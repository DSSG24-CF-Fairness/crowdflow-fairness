import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import csv

class FlowEvaluator:
    """
    A class used to evaluate the unfairness of generated flow data compared to real flow data.

    """

    def __init__(self, flows_path, generated_flows_path, demographics_path, model_type, folder_name):
        self.flows_path = flows_path
        self.generated_flows_path = generated_flows_path
        self.demographics_path = demographics_path
        self.model_type = model_type
        self.folder_name = folder_name
        # Initialize the save path based on performance and variance metrics
        self.save_path = None

    def init_log(self, performance_metric, variance_metric):
        """
        Initializes the log file path and writes headers if the log file does not exist.
        """
        self.save_path = f'../evaluation/{self.folder_name}_{self.model_type}/{performance_metric}/{variance_metric}/'
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        # Create log file with headers if it doesn't exist
        self.log_path = os.path.join(self.save_path, f'{self.folder_name}_{self.model_type}_log.csv')
        if not os.path.isfile(self.log_path):
            with open(self.log_path, mode='w', newline='') as log_file:
                log_writer = csv.writer(log_file)
                log_writer.writerow(['file_name', 'unfairness', 'performance'])

    def load_data(self):
        """
        Loads the data from the CSV files into DataFrame objects.
        """
        self.flows = pd.read_csv(self.flows_path)
        self.generated_flows = pd.read_csv(self.generated_flows_path)
        self.demographics = pd.read_csv(self.demographics_path)


    def create_buckets(self, demographic_column):
        """
        Creates demographic buckets based on the specified demographic column.

        Parameters
        ----------
        demographic_column : str
            The column name in the demographics DataFrame to be used for creating demographic buckets.
        """
        self.demographics['bucket'] = pd.qcut(self.demographics[demographic_column], q=10, labels=False)
        self.geoid_to_bucket = dict(zip(self.demographics['geoid'], self.demographics['bucket']))

    def assign_buckets(self):
        """
        Assigns each flow pair to a demographic bucket pair.
        """

        def get_bucket_pair(row):
            origin_bucket = self.geoid_to_bucket.get(row['origin'], -1)
            destination_bucket = self.geoid_to_bucket.get(row['destination'], -1)
            return tuple(sorted((origin_bucket, destination_bucket)))

        self.flows['bucket_pair'] = self.flows.apply(get_bucket_pair, axis=1)
        self.generated_flows['bucket_pair'] = self.generated_flows.apply(get_bucket_pair, axis=1)

    def merge_flows(self):
        """
        Merges the real and generated flows DataFrames on the origin and destination columns.
        """
        self.merged_flows = pd.merge(self.generated_flows, self.flows, on=['origin', 'destination'], how='left',
                                     suffixes=('_gen', '_real'))
        self.merged_flows['flow_real'].fillna(0, inplace=True)

    def calculate_performance(self, performance_metric):
        """
        Calculates the performance for each bucket pair using the specified performance metric.

        Parameters
        ----------
        performance_metric : str
            The performance metric to be used for calculating performance. Supports 'mean_squared_error' and 'CPC'.
        """
        self.performance_per_bucket = {}
        self.total_performance = 0

        for bucket_pair in self.flows['bucket_pair'].unique():
            if bucket_pair == (-1, -1):
                continue

            real_flows = self.merged_flows[self.merged_flows['bucket_pair_real'] == bucket_pair]['flow_real']
            gen_flows = self.merged_flows[self.merged_flows['bucket_pair_gen'] == bucket_pair]['flow_gen']

            if len(real_flows) > 0 and len(gen_flows) > 0:
                if performance_metric == "MSE":
                    mse = mean_squared_error(real_flows, gen_flows)
                    self.performance_per_bucket[bucket_pair] = mse
                elif performance_metric == "CPC":
                    cpc_numerator = 2 * np.sum(np.minimum(gen_flows, real_flows))
                    cpc_denominator = np.sum(gen_flows) + np.sum(real_flows)
                    cpc = cpc_numerator / cpc_denominator
                    self.performance_per_bucket[bucket_pair] = cpc
                elif performance_metric == "overestimation":
                    overestimation = np.mean(np.maximum(gen_flows - real_flows, 0))
                    self.performance_per_bucket[bucket_pair] = overestimation
                elif performance_metric == "underestimation":
                    underestimation = np.mean(np.minimum(gen_flows - real_flows, 0))
                    self.performance_per_bucket[bucket_pair] = -underestimation  # make underestimation a positive value
                # Add more performance metrics as needed

        # Calculate total performance
        total_real_flows = self.merged_flows['flow_real']
        total_gen_flows = self.merged_flows['flow_gen']

        if performance_metric == "MSE":
            self.total_performance = mean_squared_error(total_real_flows, total_gen_flows)
        elif performance_metric == "CPC":
            cpc_numerator = 2 * np.sum(np.minimum(total_gen_flows, total_real_flows))
            cpc_denominator = np.sum(total_gen_flows) + np.sum(total_real_flows)
            self.total_performance = cpc_numerator / cpc_denominator
        elif performance_metric == "overestimation":
            self.total_performance = np.mean(np.maximum(np.array(total_gen_flows) - np.array(total_real_flows), 0))
        elif performance_metric == "underestimation":
            self.total_performance = -np.mean(np.minimum(np.array(total_gen_flows) - np.array(total_real_flows), 0))

    def calculate_variance(self, variance_metric, values):
        if variance_metric == "kl_divergence":
            uniform_dist = np.full_like(values, fill_value= 1 / len(values))
            idx = 0
            # kldivs = []
            # for idx in range(len(values)):
            #     current_val = values[idx] * np.log(values[idx] / uniform_dist[idx])
            #     kldivs.append(current_val)
            # sums = np.sum(kldivs)
            #
            # kl_div = np.sum(values * np.log(values / uniform_dist))
            kldivs = []
            for idx in range(len(values)):
                current_val = uniform_dist[idx] * np.log(uniform_dist[idx]/values[idx])
                kldivs.append(current_val)
            kl_div = np.sum(kldivs)
            return kl_div
        elif variance_metric == "standard_deviation":
            return np.nanstd(values)

    def evaluate_unfairness(self, performance_metric, variance_metric, demographic_column):
        """
        Evaluates the unfairness of the generated flow data compared to the real flow data.

        Parameters
        ----------
        performance_metric : str
            The performance metric to be used for calculating performance. Currently supports 'mean_squared_error'.
        variance_metric : str
            The variance metric to be used for calculating unfairness. Currently supports 'standard_deviation','kl_divergence'
        demographic_column : str
            The column name in the features DataFrame to be used for creating demographic buckets.

        Returns
        -------
        float
            The unfairness metric calculated as the variance of performance across buckets.
        """
        self.load_data()
        self.create_buckets(demographic_column)
        self.assign_buckets()
        self.merge_flows()
        self.calculate_performance(performance_metric)

        # Create a 10x10 matrix of performance
        self.performance_matrix = np.full((10, 10), np.nan)
        for (i, j), performance in self.performance_per_bucket.items():
            self.performance_matrix[i, j] = performance
            self.performance_matrix[j, i] = performance

        # Normalize performance values for variance calculation
        normalized_performance_values = np.array(list(self.performance_per_bucket.values()))
        normalized_performance_values /= normalized_performance_values.sum()

        # if variance_metric is kl, 1-calculate_variance
        # if variance_metric is standard_deviation, 1/calculate_variance
        # Calculate variance using the specified metric
        if variance_metric == 'kl_divergence':
            # fairness = 1 - self.calculate_variance(variance_metric, normalized_performance_values)
            unfairness = self.calculate_variance(variance_metric, normalized_performance_values)
        elif variance_metric == 'standard_deviation':
            # fairness = 1 / self.calculate_variance(variance_metric, normalized_performance_values)
            unfairness = self.calculate_variance(variance_metric, normalized_performance_values)
        else:
            raise ValueError(f"Unsupported variance metric: {variance_metric}")

        # Create directory to store heatmaps
        path_parts = self.generated_flows_path.split('/')

        # Check if the directory exists, if not, create it
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(f"Directory created: {self.save_path}")
        else:
            print(f"Directory already exists: {self.save_path}")

        # Get filename from generated_flows_path and modify it
        filename = os.path.basename(
            self.generated_flows_path)  # Get the base name, e.g., 'synthetic_data_gravity_singly_constrained'
        filename = filename.replace('.csv', '')  # Remove '.csv' if present
        filename += f'_heatmap.png'

        # Full path to save the heatmap
        full_heatmap_path = os.path.join(self.save_path, filename)

        # Plot heatmap of performance
        plt.figure(figsize=(10, 8))

        performance_matrix_max = np.max(self.performance_matrix)
        self.performance_matrix_normalized = self.performance_matrix / performance_matrix_max
        # heatmap = sns.heatmap(self.performance_matrix_normalized, annot=False, cmap='Blues', cbar=True, square=True, vmin=0, vmax=1)
        heatmap = sns.heatmap(self.performance_matrix_normalized, annot=True, cmap='Blues', cbar=True, square=True, vmin=0, vmax=1)
        plt.title(f'Heatmap of {performance_metric} by Demographic Buckets', fontsize=16)
        plt.xlabel('Origin Demographic Buckets', fontsize=14)
        plt.ylabel('Destination Demographic Buckets', fontsize=14)

        # Customize the color legend
        cbar = heatmap.collections[0].colorbar
        cbar.set_label(f'{performance_metric}', fontsize=14)
        cbar.ax.tick_params(labelsize=12)  # Increase the size of the legend's text
        plt.gca().invert_yaxis()

        # Save the heatmap to file
        plt.savefig(full_heatmap_path)
        plt.close()
        print(f"Heatmap saved to {full_heatmap_path}")

        # Display the plot
        plt.show()

        # Print results
        print(f'Unfairness Metric ({variance_metric} of {performance_metric}): {unfairness}')
        print(f'Overall {performance_metric}: {self.total_performance}')


        # Append results to log file
        with open(self.log_path, mode='a', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow([filename, unfairness, self.total_performance])
        print(f"Results for {filename} logged to '{self.folder_name}_{self.model_type}_log.csv'")


        return unfairness, self.total_performance


