import sys
import os

from eval import FlowEvaluator

flows_path = '../processed_data/1/test/flows/test_flow.csv'
generated_flows_path = '../outputs/1/synthetic_data_gravity_singly_constrained/train_flow.csv' # TODO change accordingly
demographics_path = '../data/WA/demographics.csv'
demographic_column = 'svi'
evaluator = FlowEvaluator(flows_path, generated_flows_path, demographics_path)
fairness = evaluator.evaluate_fairness(accuracy_metric='CPC', variance_metric='kl_divergence', demographic_column=demographic_column)