import os
import geopandas as gpd
import osmnx as ox
from shapely.geometry import box
from shapely.ops import split
import logging

# Set up logging for errors
logging.basicConfig(
    filename='county_data_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def split_geometry_into_grid(geometry, n_rows, n_cols):
    """Divide a polygon geometry into a grid of n_rows x n_cols smaller polygons."""
    minx, miny, maxx, maxy = geometry.bounds
    width = (maxx - minx) / n_cols
    height = (maxy - miny) / n_rows
    grid_cells = []
    for i in range(n_cols):
        for j in range(n_rows):
            cell = box(minx + i * width, miny + j * height,
                       minx + (i + 1) * width, miny + (j + 1) * height)
            if geometry.intersects(cell):  # Only keep cells that intersect the original geometry
                grid_cells.append(cell.intersection(geometry))
    return grid_cells

def fetch_data_for_county(county_geometry, tags, county_name, output_dir):
    try:
        # Create subdirectory for county
        county_dir = os.path.join(output_dir, str(county_name))
        os.makedirs(county_dir, exist_ok=True)

        # Split county geometry into a grid (adjust grid size as needed)
        grid_cells = split_geometry_into_grid(county_geometry, n_rows=5, n_cols=5)
        
        # Fetch data for each grid cell
        for idx, cell in enumerate(grid_cells):
            try:
                osm_data = ox.features_from_polygon(cell, tags)
                
                # Convert geometry to WKT string within the DataFrame
                osm_data['geometry'] = osm_data['geometry'].apply(lambda x: x.wkt)
                
                # Save each cell's data to a CSV file
                cell_filename = os.path.join(county_dir, f'{county_name}_cell_{idx}.csv')
                osm_data.to_csv(cell_filename, index=False)
                print(f"Data for {county_name} cell {idx} saved to {cell_filename}")
                
            except Exception as e:
                logging.error(f"Failed to fetch data for {county_name}, cell {idx}: {e}")
    
    except Exception as e:
        logging.error(f"Failed to fetch or save data for {county_name}: {e}")

# Main code to load county boundaries and loop over them
output_dir = 'county_results_ny'
counties = gpd.read_file('NYS_Civil_Boundaries/Counties.shp')
counties = counties.to_crs(epsg=4326)

# Define the tags for data
# Define the tags for the data you're interested in
tags = {'landuse': ['residential', 'commercial', 'industrial', 'retail']
        , 'natural': True
        , 'highway': ['residential', 'trunk','primary', 'secondary', 'tertiary', 'unclassified']
        , 'amenity': ['parking', 'bar', 'pub', 'cafe', 'restaurant', 'clinic', 'hospital', 'pharmacy','college','school','kindergarten','university']
        , 'public_transport': 'stop_position'
        , 'shop': True
        }

# Iterate over counties
for index, county in counties.iterrows():
    county_geometry = county.geometry
    county_name = county['FIPS_CODE']  # Adjust field name as necessary
    if county_name == '36053' or county_name == '36115':
        fetch_data_for_county(county_geometry, tags, county_name,output_dir)
