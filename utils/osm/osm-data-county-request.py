import os

# Create a directory for county results
output_dir = 'county_results_csv_2'
os.makedirs(output_dir, exist_ok=True)  # This creates the directory if it doesn't already exist

import geopandas as gpd
import osmnx as ox

def fetch_data_for_county(county_geometry, tags, county_name, output_dir):
    try:
        # Fetch data within the county boundary
        osm_data = ox.features_from_polygon(county_geometry, tags)

        # Convert geometry to WKT string within the DataFrame
        osm_data['geometry'] = osm_data['geometry'].apply(lambda x: x.wkt)

        # Save to CSV in the specific county folder
        filename = os.path.join(output_dir, f'{county_name}.csv')
        osm_data.to_csv(filename, index=False)
        print(f"Data for {county_name} saved to {filename}")
    except Exception as e:
        print(f"Failed to fetch or save data for {county_name}: {e}")

# Load county boundaries
counties = gpd.read_file('/Users/apoorvasheera/Documents/DSSG/Crowd Flow/osm/WA_County_Boundaries/WA_County_Boundaries.shp')
counties = counties.to_crs(epsg=4326)  # Make sure CRS matches OSM data

# Define the tags for the data you're interested in
tags = {'landuse': ['residential', 'commercial', 'industrial', 'retail', 'forest']
        , 'highway': ['residential', 'primary', 'secondary', 'tertiary', 'unclassified']
        , 'amenity': ['parking', 'bar', 'pub', 'cafe', 'restaurant', 'clinic', 'hospital', 'pharmacy','college','school','kindergarten','university']
        , 'public_transport': 'stop_position'
        , 'shop': True
        }

# Iterate over each county and fetch OSM data
for index, county in counties.iterrows():
    county_geometry = county.geometry
    county_name = county['JURISDIC_5']  # Adjust field name as necessary
    fetch_data_for_county(county_geometry, tags, county_name,output_dir)
