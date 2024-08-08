import os
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads

# Load census tracts shapefile
tracts = gpd.read_file("/Users/apoorvasheera/Documents/DSSG/Crowd Flow/osm/tract20/tract20.shp")

# Convert tracts to a the same CRS as the data
tracts = tracts.to_crs(epsg=4326)  

# Directory containing your CSV files
directory = "/Users/apoorvasheera/Documents/DSSG/Crowd Flow/osm/county_results_csv_2"

# Function to process each county file
def process_county_files(directory, tracts):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            county_id = filename.split(".")[0]
            df = pd.read_csv(os.path.join(directory, filename))
            df['geometry'] = df['geometry'].apply(loads)
            gdf = gpd.GeoDataFrame(df, geometry='geometry',crs='EPSG:4326')

            # Filter tracts that belong to the current county
            relevant_tracts = tracts[tracts['GEOID20'].str.startswith(county_id)]
            
            for _, tract in relevant_tracts.iterrows():
                tract_id = tract['GEOID20']
                tract_geometry = gpd.GeoSeries(tract['geometry'],crs='EPSG:4326')
                
                # Intersect tract with data
                gdf_tract = gpd.clip(gdf, tract_geometry)

                # Convert the clipped data to a projected CRS for accurate area and length calculations
                gdf_tract = gdf_tract.to_crs(epsg=26910)
                
                # Calculate area in km2 for different land uses
                def calc_area(landuse):
                    if 'landuse' in gdf_tract.columns:
                        return gdf_tract[gdf_tract['landuse'] == landuse]['geometry'].area.sum() / 1e6
                    else:
                        print(f"No landuse information for {filename}")
                        return 0

                # Calculate road lengths in km
                def calc_road_length(road_type):
                    if 'highway' in gdf_tract.columns:
                        roads = gdf_tract[gdf_tract['highway'] == road_type]
                        return roads['geometry'].length.sum() / 1000
                    else:
                        print(f"No highway information for {filename}")
                        return 0

                result = {
                    'GEODID': tract_id,
                    'residential_landuse': calc_area('residential'),
                    'commercial_landuse': calc_area('commercial'),
                    'industrial_landuse': calc_area('industrial'),
                    'retail_landuse': calc_area('retail'),
                    'forest_landuse': calc_area('forest'),
                    'road_residential': calc_road_length('residential'),
                    'road_primary': calc_road_length('primary'),
                    'road_other': calc_road_length('secondary') + calc_road_length('tertiary') + calc_road_length('unclassified'),
                    'transport_point': gdf_tract[gdf_tract['public_transport'] == 'stop_position'].shape[0] if 'public_transport' in gdf_tract.columns else 0 + gdf_tract[gdf_tract['amenity'] == 'parking'].shape[0],
                    'food_point': gdf_tract[gdf_tract['amenity'].isin(['bar', 'pub', 'cafe', 'restaurant'])].shape[0],
                    'health_point': gdf_tract[gdf_tract['amenity'].isin(['clinic', 'hospital', 'pharmacy'])].shape[0],
                    'education_point': gdf_tract[gdf_tract['amenity'].isin(['college', 'school', 'kindergarten', 'university'])].shape[0],
                    'retail_point': gdf_tract['shop'].notna().sum() if 'shop' in gdf_tract.columns else 0
                }
                results.append(result)
    
    # Convert results to DataFrame and return
    return pd.DataFrame(results)

# Running the function
final_data = process_county_files(directory, tracts)
final_data.to_csv("/Users/apoorvasheera/Documents/DSSG/Crowd Flow/osm/aggregated-data.csv", index=False)
print("Data aggregation complete and saved.")
