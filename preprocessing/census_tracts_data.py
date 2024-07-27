import censusdata
import pandas as pd


def get_census_data(tables, state, county='*', year=2022):
    """
    Download census data for a given state and optional county.
    """
    # Download the data
    data = censusdata.download('acs5', year,  # Use 2022 ACS 5-year estimates
                               censusdata.censusgeo([('state', state), ('county', county), ('tract', '*')]),
                               list(tables.keys()))

    # Rename the columns
    data.rename(columns=tables, inplace=True)

    # Extract GEOID information
    data['GEOID'] = data.index.to_series().apply(lambda x: x.geo[0][1] + x.geo[1][1] + x.geo[2][1])
    data.reset_index(drop=True, inplace=True)
    data = data[['GEOID'] + list(tables.values())]
    return data


def get_tract_data(state_fips, county_fips=None, save_to_csv=False, filename='census_data.csv'):
    """
    Fetch census data (total population and other features) for a given state and optional county, and save to CSV if specified.
    """
    # Define the tables of features
    tables = {
        'B19013_001E': 'MedianIncome',
        'B19013_001M': 'MedianIncome_MoE',
        'B01003_001E': 'TotalPopulation',
        'B01003_001M': 'TotalPopulation_MoE',
        'B01002_001E': 'MedianAge',
        'B01002_001M': 'MedianAge_MoE',
        'B17001_002E': 'PopulationBelowPovertyLevel',
        'B17001_002M': 'PopulationBelowPovertyLevel_MoE',
        'B02001_002E': 'PopulationWhiteAlone',
        'B02001_003E': 'PopulationBlackAlone',
        'B02001_004E': 'PopulationAmericanIndianAlaskaNativeAlone',
        'B02001_005E': 'PopulationAsianAlone',
        'B02001_006E': 'PopulationNativeHawaiianPacificIslanderAlone',
        'B02001_007E': 'PopulationSomeOtherRaceAlone',
        'B02001_008E': 'PopulationTwoOrMoreRaces',
        'B03002_003E': 'PopulationNotHispanicWhiteAlone',
        'B03003_003E': 'PopulationHispanic',
        'B25064_001E': 'MedianGrossRent',
        'B25077_001E': 'MedianHomeValue',
        'B25035_001E': 'MedianYearStructureBuilt',
        'B25001_001E': 'TotalHousingUnits',
        'B25004_001E': 'TotalVacantHousingUnits',
        'B25003_002E': 'OccupiedHousingUnitsOwnerOccupied',
        'B25003_003E': 'OccupiedHousingUnitsRenterOccupied',
        'B27001_005E': 'PopulationNoHealthInsuranceCoverage',
    }

    # Use '*' for county if not provided
    county_fips = county_fips if county_fips is not None else '*'

    data = get_census_data(tables, state_fips, county_fips)
    if save_to_csv:
        data.to_csv(filename, index=False)

    return data


# Example usage:
# Fetch the data for Washington state (state FIPS code is '53') and save it to a CSV file
washington_data = get_tract_data('53', save_to_csv=True, filename='../data/washington_census_data.csv')

# Fetch the data for King County, Washington state (state FIPS code is '53', county FIPS code is '033')
# king_county_data = get_tract_data('53', '033', save_to_csv=True, filename='../data/king_county_census_data.csv')