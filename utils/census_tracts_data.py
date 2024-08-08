import censusdata
import pandas as pd
import geopandas as gpd


def get_census_data(tables, state, year, county='*'):
    """
    Download census data for a given state and optional county.
    Default year: 2022, use 2022 ACS 5-year estimates
    """
    # Download the census data
    data = censusdata.download('acs5', year,
                               censusdata.censusgeo([('state', state), ('county', county), ('tract', '*')]),
                               list(tables.keys()))

    data.rename(columns=tables, inplace=True)

    # Extract required GEOID information
    data['GEOID'] = data.index.to_series().apply(lambda x: x.geo[0][1] + x.geo[1][1] + x.geo[2][1])
    data.reset_index(drop=True, inplace=True)
    data = data[['GEOID'] + list(tables.values())]
    return data


def get_census_tract_geom(state_fips, year, county_fips=None):
    """
    Download the census tract geometries for a given state and optional county.
    """
    # Download the census tract shapefiles
    tracts = gpd.read_file(f'https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip')

    # Filter to the specific county if provided
    if county_fips and county_fips != '*':
        tracts = tracts[tracts['COUNTYFP'] == county_fips]

    # Convert to EPSG:4326
    tracts = tracts.to_crs('EPSG:4326')

    # Create GEOID and set as index
    tracts['GEOID'] = tracts['STATEFP'] + tracts['COUNTYFP'] + tracts['TRACTCE']
    tracts = tracts[['GEOID', 'geometry']]
    tracts.set_index('GEOID', inplace=True)

    return tracts


def get_census_tract_data(state_fips, year, county_fips=None, save_to_csv=False, filename='census_data.csv'):
    """
    Fetch census data (total population and other features) for a given state and optional county, and save to CSV if specified.
    """
    # Define the tables of features to fetch
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

    data = get_census_data(tables, state_fips, year, county_fips)
    tract_geom_gdf = get_census_tract_geom(state_fips, year, county_fips)
    merged_data = data.merge(tract_geom_gdf, on='GEOID')
    merged_gdf = gpd.GeoDataFrame(merged_data, geometry='geometry')

    # Save to CSV if specified
    if save_to_csv:
        merged_gdf.to_csv(filename, index=False)

    return merged_gdf


# Example usage:
# Fetch the data for Washington state (state FIPS code is '53') and save it to a CSV file
# washington_data = get_census_tract_data('53', 2020, save_to_csv=True, filename='../data/washington_census_data.csv')
# washington_data[['GEOID', 'TotalPopulation']].to_csv('../data/washington_census_tracts_population_data.csv', index=False)

# Fetch the data for King County, Washington state (state FIPS code is '53', county FIPS code is '033')
# king_county_data = get_census_tract_data('53', '033', save_to_csv=True, filename='../data/king_county_census_data.csv')

