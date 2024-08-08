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
    demographics_path : str
        Path to the CSV file containing the demographic features data.
    """

    def __init__(self, flows_path, generated_flows_path, demographics_path):
        """
        Initializes the FlowEvaluator with the paths to the data files.

        Parameters
        ----------
        flows_path : str
            Path to the CSV file containing the real flow data.
        generated_flows_path : str
            Path to the CSV file containing the generated flow data.
        demographics_path : str
            Path to the CSV file containing the demographic features data.
        """
        self.flows_path = flows_path
        self.generated_flows_path = generated_flows_path
        self.demographics_path = demographics_path

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

    # def assign_buckets(self):
    #     """
    #     Assigns each flow pair to a demographic bucket pair using vectorized operations.
    #     """
    #     # Vectorize the mapping from geoid to bucket
    #     self.flows['origin_bucket'] = self.flows['origin'].map(self.geoid_to_bucket).fillna(-1).astype(int)
    #     self.flows['destination_bucket'] = self.flows['destination'].map(self.geoid_to_bucket).fillna(-1).astype(int)

    #     # Ensure that the pairs are sorted
    #     self.flows['bucket_pair'] = self.flows.apply(
    #         lambda row: tuple(sorted((row['origin_bucket'], row['destination_bucket']))), axis=1
    #     )

    #     self.generated_flows['origin_bucket'] = self.generated_flows['origin'].map(self.geoid_to_bucket).fillna(-1).astype(int)
    #     self.generated_flows['destination_bucket'] = self.generated_flows['destination'].map(self.geoid_to_bucket).fillna(-1).astype(int)
    #     self.generated_flows['bucket_pair'] = self.generated_flows.apply(
    #         lambda row: tuple(sorted((row['origin_bucket'], row['destination_bucket']))), axis=1
    #     )

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

    def calculate_accuracy(self, accuracy_metric):
        """
        Calculates the accuracy for each bucket pair using the specified accuracy metric.

        Parameters
        ----------
        accuracy_metric : str
            The accuracy metric to be used for calculating accuracy. Supports 'mean_squared_error' and 'CPC'.
        """
        self.performance_per_bucket = {}
        self.total_performance = 0

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

        # Calculate total accuracy
        total_real_flows = self.merged_flows['flow_real']
        total_gen_flows = self.merged_flows['flow_gen']

        if accuracy_metric == "mean_squared_error":
            self.total_performance = mean_squared_error(total_real_flows, total_gen_flows)
        elif accuracy_metric == "CPC":
            cpc_numerator = 2 * np.sum(np.minimum(total_gen_flows, total_real_flows))
            cpc_denominator = np.sum(total_gen_flows) + np.sum(total_real_flows)
            self.total_performance = cpc_numerator / cpc_denominator
        elif accuracy_metric == "overestimation":
            self.total_performance = np.mean(np.maximum(np.array(total_gen_flows) - np.array(total_real_flows), 0))
        elif accuracy_metric == "underestimation":
            self.total_performance = -np.mean(np.minimum(np.array(total_gen_flows) - np.array(total_real_flows), 0))

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
        sns.heatmap(self.accuracy_matrix, annot=True, cmap='Blues', cbar=True, square=True)
        plt.title(f'Heatmap of {accuracy_metric} by Demographic Buckets')
        plt.xlabel('Origin Demographic Buckets')
        plt.ylabel('Destination Demographic Buckets')
        plt.gca().invert_yaxis()
        plt.show()

        # Print results 
        print(f'Fairness Metric ({variance_metric} of {accuracy_metric}): {fairness}')
        print(f'Overall {accuracy_metric}: {self.total_performance}')

        return fairness, self.total_performance

# # Example usage
# flows_path = 'flows.csv'
# generated_flows_path = 'generated-flows.csv'
# demographics_path = 'features.csv'
# demographic_column = 'svi'

# evaluator = FlowEvaluator(flows_path, generated_flows_path, demographics_path)
# fairness = evaluator.evaluate_fairness(accuracy_metric=CPC', variance_metric='kl-divergence', demographic_column=demographic_column)
# print(f'Fairness Metric (Variance of Accuracy): {fairness}')
