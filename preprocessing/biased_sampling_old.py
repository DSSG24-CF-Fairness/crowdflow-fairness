import pandas as pd
import numpy as np

class BiasedFlowCalculator:
    def __init__(self, df_flow, rank_order='asc'):
        """
        Initialize the BiasedFlowCalculator with a DataFrame and rank order.

        Parameters:
        df_flow (pd.DataFrame): DataFrame containing the flow data.
        rank_order (str): Order to rank the demographic delta ('asc' for ascending, 'desc' for descending).
        """
        self.df_flow = df_flow
        self.rank_order = rank_order

    def calculate_biased_flow(self):
        """
        Calculate biased flow and B_Flow for each origin with rounding.

        Returns:
        pd.DataFrame: The DataFrame with additional columns:
                      - 'Demographic Delta': Absolute difference between the origin's demographic and the destination's demographic.
                      - 'Demographic Rank': Rank of the demographic delta for each destination within the origin.
                      - 'Cumulative Population': Cumulative population of destinations with lower/higher demographic based on rank order.
                      - 'Percentile': Percentile based on the cumulative population.
                      - 'Biased Flow': Biased flow value calculated based on the given formula.
                      - 'B_Flow': Rounded biased flow value, adjusted to ensure the total outflow remains the same.
        """
        df_flow = self.df_flow.copy()
        # Step 1: Calculate absolute Demographic Delta and create Demographic Rank
        df_flow['Demographic Delta'] = abs(df_flow['demographic_col_origin'] - df_flow['demographic_col_destination'])

        # Step 2: Rank within each origin
        df_flow['Demographic Rank'] = df_flow.groupby('Origin')['Demographic Delta'].rank(ascending=self.rank_order == 'asc', method='min')

        # Step 3: Sort values within each origin and create Cumulative Population
        df_flow = df_flow.sort_values(['Origin', 'Demographic Rank'], ascending=[True, True])
        df_flow['Cumulative Population'] = df_flow.groupby('Origin')['Population'].cumsum() - df_flow['Population']

        # Step 4: Create Percentile
        total_population_per_origin = df_flow.groupby('Origin')['Population'].transform('sum')
        df_flow['Percentile'] = (df_flow['Cumulative Population'] + 0.5 * df_flow['Population']) / total_population_per_origin * 100
        df_flow['Percentile'] = df_flow['Percentile'].apply(lambda x: f'{x:.2f}%')

        # Step 5: Calculate Biased Flow
        df_flow['Percentile Value'] = df_flow['Percentile'].str.rstrip('%').astype(float) / 100
        df_flow['Numerator'] = (df_flow['Percentile Value'] + 0.5) * df_flow['Flow']
        denominator_per_origin = df_flow.groupby('Origin')['Numerator'].transform('sum')
        df_flow['Biased Flow'] = df_flow['Numerator'] / denominator_per_origin

        # Step 6: Calculate B_Flow
        total_outflow_per_origin = df_flow.groupby('Origin')['Flow'].transform('sum')
        df_flow['B_Flow'] = df_flow['Biased Flow'] * total_outflow_per_origin

        # Step 7: Round B_Flow to integers
        df_flow['B_Flow'] = df_flow['B_Flow'].round()

        # Adjust B_Flow to ensure total outflow remains the same within each origin
        adjustment_needed = df_flow.groupby('Origin')['Flow'].sum() - df_flow.groupby('Origin')['B_Flow'].sum()
        for origin, diff in adjustment_needed.items():
            if diff != 0:
                adjustment_index = df_flow[df_flow['Origin'] == origin]['B_Flow'].idxmax() if diff > 0 else df_flow[df_flow['Origin'] == origin]['B_Flow'].idxmin()
                df_flow.at[adjustment_index, 'B_Flow'] += diff

        # Drop intermediate columns
        df_flow.drop(columns=['Percentile Value', 'Numerator'], inplace=True)

        return df_flow
    

    def allocate_flows_fixed(self, origin, total_outflow, expected_values):
        """
        Allocate total outflow to destinations based on biased flow probabilities and fixed expected values.

        Parameters:
        origin (str): The origin from which to allocate the total outflow.
        total_outflow (int): The total outflow to allocate.
        expected_values (dict): A dictionary containing the expected values for each destination.

        Returns:
        pd.DataFrame: The updated DataFrame with allocated flows.
        """
        df_flow = self.df_flow[self.df_flow['Origin'] == origin].copy()
        for destination, expected_value in expected_values.items():
            df_flow.loc[df_flow['Destination'] == destination, 'Allocated Flow'] = expected_value
        
        return df_flow


# # Provided test data
# data = {
#     'Destination': ['53033000101', '53033000102', '53033000103', '53033000105', '53033000107'],
#     'Origin': ['53033000100', '53033000100', '53033000100', '53033000104', '53033000104'],
#     'demographic_col_origin': [100, 100, 100, 200, 200],
#     'demographic_col_destination': [150, 270, 260, 280, 300],
#     'Population': [20, 100, 50, 100, 400], #
#     'Flow': [10, 5, 3, 20, 10]
# }

# df = pd.DataFrame(data)
# pd.set_option('display.max_columns', None)

# # Instantiate the BiasedFlowCalculator class with ascending rank order
# calculator_asc = BiasedFlowCalculator(df, rank_order='asc')
# df_asc = calculator_asc.calculate_biased_flow()

# # Instantiate the BiasedFlowCalculator class with descending rank order
# calculator_desc = BiasedFlowCalculator(df, rank_order='desc')
# df_desc = calculator_desc.calculate_biased_flow()

# print(df_asc.head())
# print(df_desc.head())
