import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class FlowEvaluator:
    """
    A class used to evaluate the fairness of generated flow data compared to real flow data.

    Attributes
    ----------
    flows_path : str
        Path to the CSV file containing the real flow data.
    generated_flows_path : str
        Path to the CSV file containing the generated flow data.
    features_path : str
        Path to the CSV file containing the demographic features data.
    """

    def __init__(self, flows_path, generated_flows_path, features_path):
        """
        Initializes the FlowEvaluator with the paths to the data files.

        Parameters
        ----------
        flows_path : str
            Path to the CSV file containing the real flow data.
        generated_flows_path : str
            Path to the CSV file containing the generated flow data.
        features_path : str
            Path to the CSV file containing the demographic features data.
        """
        self.flows_path = flows_path
        self.generated_flows_path = generated_flows_path
        self.features_path = features_path

    def load_data(self):
        """
        Loads the data from the CSV files into DataFrame objects.
        """
        self.flows = pd.read_csv(self.flows_path)
        self.generated_flows = pd.read_csv(self.generated_flows_path)
        self.features = pd.read_csv(self.features_path)

    def create_buckets(self, demographic_column):
        """
        Creates demographic buckets based on the specified demographic column.

        Parameters
        ----------
        demographic_column : str
            The column name in the features DataFrame to be used for creating demographic buckets.
        """
        self.features['bucket'] = pd.qcut(self.features[demographic_column], q=10, labels=False)
        self.geoid_to_bucket = dict(zip(self.features['geoid'], self.features['bucket']))

    def assign_buckets(self):
        """
        Assigns each flow pair to a demographic bucket pair.
        """
        def get_bucket_pair(row):
            origin_bucket = self.geoid_to_bucket.get(row['origin'], -1)
            destination_bucket = self.geoid_to_bucket.get(row['destination'], -1)
            if origin_bucket > destination_bucket:
                origin_bucket, destination_bucket = destination_bucket, origin_bucket
            return origin_bucket, destination_bucket

        self.flows['bucket_pair'] = self.flows.apply(get_bucket_pair, axis=1)
        self.generated_flows['bucket_pair'] = self.generated_flows.apply(get_bucket_pair, axis=1)

    def merge_flows(self):
        """
        Merges the real and generated flows DataFrames on the origin and destination columns.
        """
        self.merged_flows = pd.merge(self.generated_flows, self.flows, on=['origin', 'destination'], how='left',
                                     suffixes=('_gen', '_real'))
        self.merged_flows['flow_real'].fillna(0, inplace=True)

    def calculate_accuracy(self, accuracy_metric):
        """
        Calculates the accuracy for each bucket pair using the specified accuracy metric.

        Parameters
        ----------
        accuracy_metric : str
            The accuracy metric to be used for calculating accuracy. Supports 'mean_squared_error' and 'CPC'.
        """
        self.performance_per_bucket = {}
        for bucket_pair in self.flows['bucket_pair'].unique():
            if bucket_pair == (-1, -1):
                continue

            real_flows = self.merged_flows[self.merged_flows['bucket_pair_real'] == bucket_pair]['flow_real']
            gen_flows = self.merged_flows[self.merged_flows['bucket_pair_gen'] == bucket_pair]['flow_gen']

            if len(real_flows) > 0 and len(gen_flows) > 0:
                if accuracy_metric == "mean_squared_error":
                    mse = mean_squared_error(real_flows, gen_flows)
                    self.performance_per_bucket[bucket_pair] = mse
                elif accuracy_metric == "CPC":
                    cpc_numerator = 2 * np.sum(np.minimum(gen_flows, real_flows))
                    cpc_denominator = np.sum(gen_flows) + np.sum(real_flows)
                    cpc = cpc_numerator / cpc_denominator
                    self.performance_per_bucket[bucket_pair] = cpc
                elif accuracy_metric == "overestimation":
                    overestimation = np.mean(np.maximum(gen_flows - real_flows, 0))
                    self.performance_per_bucket[bucket_pair] = overestimation
                elif accuracy_metric == "underestimation":
                    underestimation = np.mean(np.minimum(gen_flows - real_flows, 0))
                    self.performance_per_bucket[bucket_pair] = -underestimation  # make underestimation a positive value
                # Add more accuracy metrics as needed

    def calculate_variance(self, variance_metric, values):
        if variance_metric == "kl_divergence":
            uniform_dist = np.full_like(values, fill_value=1/len(values))
            kl_div = np.sum(values * np.log(values / uniform_dist))
            return kl_div
        elif variance_metric == "standard_deviation":
            return np.nanstd(values)
        

    def evaluate_fairness(self, accuracy_metric, variance_metric, demographic_column):
        """
        Evaluates the fairness of the generated flow data compared to the real flow data.

        Parameters
        ----------
        accuracy_metric : str
            The accuracy metric to be used for calculating accuracy. Currently supports 'mean_squared_error'.
        variance_metric : str
            The variance metric to be used for calculating fairness. Currently supports 'standard_deviation','kl_divergence'
        demographic_column : str
            The column name in the features DataFrame to be used for creating demographic buckets.

        Returns
        -------
        float
            The fairness metric calculated as the variance of accuracy across buckets.
        """
        self.load_data()
        self.create_buckets(demographic_column)
        self.assign_buckets()
        self.merge_flows()
        self.calculate_accuracy(accuracy_metric)

        # Create a 10x10 matrix of accuracy
        self.accuracy_matrix = np.full((10, 10), np.nan)
        for (i, j), performance in self.performance_per_bucket.items():
            self.accuracy_matrix[i, j] = performance
            self.accuracy_matrix[j, i] = performance

        # Normalize accuracy values for variance calculation
        normalized_accuracy_values = np.array(list(self.performance_per_bucket.values()))
        normalized_accuracy_values /= normalized_accuracy_values.sum()

        # Calculate variance using the specified metric
        fairness = self.calculate_variance(variance_metric, normalized_accuracy_values)

        # Plot heatmap of accuracy
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.accuracy_matrix, annot=True, cmap='viridis', cbar=True, square=True)
        plt.title(f'Heatmap of {accuracy_metric} by Demographic Buckets')
        plt.xlabel('Origin Demographic Buckets')
        plt.ylabel('Destination Demographic Buckets')
        plt.show()

        return fairness

# # Example usage
# flows_path = 'flows.csv'
# generated_flows_path = 'generated-flows.csv'
# features_path = 'features.csv'
# demographic_column = 'RPL_THEME1'

# evaluator = FlowEvaluator(flows_path, generated_flows_path, features_path)
# fairness = evaluator.evaluate_fairness(accuracy_metric='mean_squared_error', variance_metric='np.nanvar', demographic_column=demographic_column)
# print(f'Fairness Metric (Variance of Accuracy): {fairness}')
