import os
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads

# Load census tracts shapefile
tracts = gpd.read_file("tl_2020_36_tract/tl_2020_36_tract.shp")

# Convert tracts to the same CRS as the data
tracts = tracts.to_crs(epsg=4326)

# Directory containing your CSV files
directory = "county_results_ny"

# Function to process each county file
def process_county_files(directory, tracts):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            county_id = filename.split(".")[0]
            df = pd.read_csv(os.path.join(directory, filename))
            df['geometry'] = df['geometry'].apply(loads)
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

            # Filter tracts that belong to the current county
            relevant_tracts = tracts[tracts['GEOID'].str.startswith(county_id)]
            
            for _, tract in relevant_tracts.iterrows():
                tract_id = tract['GEOID']
                tract_geometry = gpd.GeoSeries(tract['geometry'], crs='EPSG:4326')
                
                # Intersect tract with data
                gdf_tract = gpd.clip(gdf, tract_geometry)

                # Convert the clipped data to a projected CRS for accurate area and length calculations
                gdf_tract = gdf_tract.to_crs(epsg=26910)

                # General function to calculate area in km² for given filter criteria
                def calc_area(column_name, values):
                    if column_name in gdf_tract.columns:
                        if isinstance(values, list):
                            filtered_gdf = gdf_tract[gdf_tract[column_name].isin(values)]
                        else:
                            filtered_gdf = gdf_tract[gdf_tract[column_name] == values]
                        return filtered_gdf['geometry'].area.sum() / 1e6  # Calculate area in km²
                    else:
                        print(f"No information for {column_name} in {filename}")
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
                    'GEOID': tract_id,
                    'residential_landuse': calc_area('landuse', 'residential'),
                    'commercial_landuse': calc_area('landuse', 'commercial'),
                    'industrial_landuse': calc_area('landuse', 'industrial'),
                    'retail_landuse': calc_area('landuse', 'retail'),
                    'natural_landuse': calc_area('natural', True),  # Calculate area where 'natural' == True
                    'road_residential': calc_road_length('residential'),
                    'road_primary': calc_road_length('primary') + calc_road_length('trunk'),
                    'road_other': calc_road_length('secondary') + calc_road_length('tertiary') + calc_road_length('unclassified'),
                    'transport_point': gdf_tract[gdf_tract['public_transport'] == 'stop_position'].shape[0] if 'public_transport' in gdf_tract.columns else 0 + gdf_tract[gdf_tract['amenity'] == 'parking'].shape[0],
                    'transport_area': calc_area('public_transport', 'stop_position') + calc_area('amenity', 'parking'),  # Area of transport points
                    'food_point': gdf_tract[gdf_tract['amenity'].isin(['bar', 'pub', 'cafe', 'restaurant'])].shape[0],
                    'food_area': calc_area('amenity', ['bar', 'pub', 'cafe', 'restaurant']),  # Area of food points
                    'health_point': gdf_tract[gdf_tract['amenity'].isin(['clinic', 'hospital', 'pharmacy'])].shape[0],
                    'health_area': calc_area('amenity', ['clinic', 'hospital', 'pharmacy']),  # Area of health points
                    'education_point': gdf_tract[gdf_tract['amenity'].isin(['college', 'school', 'kindergarten', 'university'])].shape[0],
                    'education_area': calc_area('amenity', ['college', 'school', 'kindergarten', 'university']),  # Area of education points
                    'retail_point': gdf_tract['shop'].notna().sum() if 'shop' in gdf_tract.columns else 0,
                    'retail_area': calc_area('shop', True)  # Area of retail points
                }
                results.append(result)
    
    # Convert results to DataFrame and return
    return pd.DataFrame(results)

# Running the function
final_data = process_county_files(directory, tracts)
final_data.to_csv("aggregated-data-ny.csv", index=False)
print("Data aggregation complete and saved.")
