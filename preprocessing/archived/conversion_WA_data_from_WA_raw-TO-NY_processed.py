import json

import geopandas
import pandas as pd
import os
import pickle
from tqdm import tqdm
import csv
import pprint

root_path = "data/washington"
os.makedirs(os.path.join(root_path, "processed"), exist_ok=True)


'''
# mapping geoids in flow.csv to polygon geoids in tessellation.geojson
# read flow and tessellation file
flow = pd.read_csv(root_path + "/flow.csv")
tessellation = geopandas.read_file(os.path.join(root_path, "tessellation.geojson"))

# add columns to the tessellation dataframe callled new_geoid_o and new_geoid_d
flow["new_geoid_o"] = None
flow["new_geoid_d"] = None
from shapely.geometry import Point

progress = 0

progress_bar = tqdm.tqdm_notebook(total=len(flow))
for i, row in flow.iterrows():
    progress_bar.update(1)
    # print("progress", progress,"of", len(flow))
    lng_o = row["lng_o"]
    lat_o = row["lat_o"]
    geoid_o_old = str(row["geoid_o"])
    lng_d = row["lng_d"]
    lat_d = row["lat_d"]
    geoid_d_old = str(row["geoid_d"])

    for j, standard_row in tessellation.iterrows():
        if standard_row["geometry"].contains(Point(lng_o, lat_o)):
            flow.at[i, "new_geoid_o"] = standard_row["GEOID"]

        if standard_row["geometry"].contains(Point(lng_d, lat_d)):
            flow.at[i, "new_geoid_d"] = standard_row["GEOID"]

flow.to_csv('data/washington/flow_GEOIDadjusted.csv', index=False)
'''









# test_tiles.csv
test_tiles = pd.read_csv(root_path + "/test_region_index.csv")
test_tiles.to_csv(root_path + "/processed/test_tiles.csv", header=False, index=False)
print("Output -> test_tiles.csv, len is", len(test_tiles))
# len(test_tiles) 120


# train_tiles.csv
train_tiles = pd.read_csv(root_path + "/train_region_index.csv")
train_tiles.to_csv(root_path + "/processed/train_tiles.csv", header=False, index=False)
print("Output -> train_tiles.csv, len is", len(train_tiles))
# len(train_tiles) 120










# Original data
flow = pd.read_csv(root_path + "/flow_GEOIDadjusted.csv")
demographics = pd.read_csv(os.path.join(root_path, "demographics.csv"))
tessellation = geopandas.read_file(os.path.join(root_path, "tessellation.geojson"))
features = pd.read_csv(os.path.join(root_path, "features.csv"))


# find the intersection of GEOIDs
GEOIDs_flow = set()
for i, row in flow.iterrows():
    geoid_o = str(int(row["new_geoid_o"]))
    geoid_d = str(int(row["new_geoid_d"]))
    GEOIDs_flow.add(geoid_o)
    GEOIDs_flow.add(geoid_d)

GEOIDs_demographics = set()
for i, row in demographics.iterrows():
    geoid = str(int(row["geoid"]))
    GEOIDs_demographics.add(geoid)


GEOIDs_features = set()
for i, row in features.iterrows():
    geoid = str(int(row["GEOID"]))
    GEOIDs_features.add(geoid)

GEOIDs_tessellation = set()
for i, row in tessellation.iterrows():
    geoid = str(int(row["GEOID"]))
    GEOIDs_tessellation.add(geoid)

GEOIDs_intersected = GEOIDs_flow & GEOIDs_demographics & GEOIDs_features & GEOIDs_tessellation
# len(GEOIDs_intersected) 1429






# TO -> od2flow_new_york.csv.zip, res 138623 rows
flows_oa = flow.rename(columns={"new_geoid_o": "residence", "new_geoid_d": "workplace", "pop_flows": "commuters"})
filtered_flows_oa = pd.DataFrame(columns=["residence", "workplace", "commuters"])

# for i, row in flows_oa.iterrows():
for i, row in tqdm(flows_oa.iterrows(), total=len(flows_oa), desc= "-> od2flow_new_york.csv.zip"):
    residence = str(row["residence"])
    workplace = str(row["workplace"])

    if residence in GEOIDs_intersected and workplace in GEOIDs_intersected:
        filtered_flows_oa = pd.concat([filtered_flows_oa, pd.DataFrame([row])], ignore_index=True)

filtered_flows_oa_grouped = filtered_flows_oa.groupby(['residence', 'workplace'], as_index=False).sum()

filtered_flows_oa_grouped = filtered_flows_oa_grouped[["residence", "workplace", "commuters"]]
filtered_flows_oa_grouped.to_csv(root_path + "/processed/od2flow_new_york.csv.zip", index=False)
print("Output -> od2flow_new_york.csv.zip, len is", len(filtered_flows_oa_grouped))







# TO -> od2flow.pkl, res 138623 rows
od2flow = {(str(int(row['residence'])), str(int(row['workplace']))): row['commuters'] for _, row in filtered_flows_oa_grouped.iterrows()}

with open(root_path + '/processed/od2flow.pkl', 'wb') as f:
    pickle.dump(od2flow, f)
print("Output -> od2flow.pkl, len is", len(od2flow))









# TO -> oa2centroid.pkl, res 1429 rows
oa2centroid = dict()

# for i, row in flow.iterrows():
for i, row in tqdm(flow.iterrows(), total=len(flow), desc= "-> oa2centroid.pkl"):
    geoid_o = str(row["new_geoid_o"])
    geoid_d = str(row["new_geoid_d"])

    if geoid_o in GEOIDs_intersected:
        oa2centroid[geoid_o] = [row["lng_o"], row["lat_o"]]

    if geoid_d in GEOIDs_intersected:
        oa2centroid[geoid_d] = [row["lng_d"], row["lat_d"]]

with open(root_path + "/processed/oa2centroid.pkl", "wb") as f:
    pickle.dump(oa2centroid, f)
print("Output -> oa2centroid.pkl, len is", len(oa2centroid))










# TO -> oa2features.pkl, res 1429 rows
oa2features = dict()

# for i, row in features.iterrows():
for i, row in tqdm(features.iterrows(), total=len(features), desc= "-> oa2features.pkl"):
    geoid = str(int(row["GEOID"]))

    if geoid in GEOIDs_intersected:
        feature = [
            row["residential_landuse"], row["commercial_landuse"], row["industrial_landuse"],
            row["retail_landuse"], row["forest_landuse"], row["road_residential"],
            row["road_primary"], row["road_other"], row["transport_point"],
            row["food_point"], row["health_point"], row["education_point"], row["retail_point"]
        ]

        oa2features[geoid] = feature

with open(root_path + "/processed/oa2features.pkl", "wb") as f:
    pickle.dump(oa2features, f)
print("Output -> oa2features.pkl, len is", len(oa2features))













# TO -> oa_gdf.csv.gz, res 1429 rows
oa_gdf = []
# for i, geo_code in enumerate(GEOIDs_intersected):
for i, geo_code in tqdm(enumerate(GEOIDs_intersected), total=len(GEOIDs_intersected), desc= "-> oa_gdf.csv.gz"):
    # Default values
    centroid = None
    area_km2 = None

    # Check if geo_code exists in the tessellation DataFrame
    tessellation_match = tessellation[tessellation['GEOID'] == geo_code]
    if not tessellation_match.empty:
        lng = tessellation_match.iloc[0]['lng']
        lat = tessellation_match.iloc[0]['lat']
        centroid = [lng, lat]  # Store centroid as a tuple
    else:
        print(f"Tessellation data not found for GEOID {geo_code}")

    # Check if geo_code exists in the demographics DataFrame
    demographics_match = demographics[demographics['geoid'] == int(geo_code)]
    if not demographics_match.empty:
        area_sqmi = demographics_match.iloc[0]['AREA_SQMI']
        area_km2 = area_sqmi * 2.58999  # Convert sq mi to km²
    else:
        print(f"Demographics data not found for GEOID {geo_code}")

    # Append the data as a dictionary
    oa_gdf.append({
        "Unnamed: 0": i,  # Index value
        "geo_code": geo_code,
        "centroid": centroid,
        "area_km2": area_km2
    })

# Create the final DataFrame from the collected data
oa_gdf = pd.DataFrame(oa_gdf)
oa_gdf.to_csv(root_path + "/processed/oa_gdf.csv.gz", index=False)
print("Output -> oa_gdf.csv.gz, len is", len(oa_gdf))




















# TO -> tileid2oa2handmade_features.json
features_mapping = {}
for index, row in tqdm(features.iterrows(), total=len(features), desc="-> tileid2oa2handmade_features.json"):
    geoid = int(row["GEOID"])

    # Proceed only if geoid is in the intersected set
    if str(geoid) in GEOIDs_intersected:
        feature_data = row.drop("GEOID").to_dict()

        # Convert each feature's value to a list containing that value
        feature_data = {key: [value] for key, value in feature_data.items()}
        features_mapping[str(geoid)] = feature_data

for geoid in GEOIDs_intersected:
    if str(geoid) not in features_mapping:
        features_mapping[str(geoid)] = {}

print("features_mapping size", len(features_mapping))




region_dict = {}
with open(root_path + "/test_tile_geoids.csv", mode='r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        region_index = int(row[1])  # Convert region_index to integer
        geoids = row[2].split(',')  # Split the geoids into a list
        used_geoids = []
        for geoid in geoids:
            if geoid in GEOIDs_intersected:
                used_geoids.append(geoid)
        region_dict[region_index] = used_geoids  # Store in the dictionary

with open(root_path + "/train_tile_geoids.csv", mode='r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        region_index = int(row[1])  # Convert region_index to integer
        geoids = row[2].split(',')  # Split the geoids into a list
        used_geoids = []
        for geoid in geoids:
            if geoid in GEOIDs_intersected:
                used_geoids.append(geoid)
        region_dict[region_index] = used_geoids
tileid2oa2handmade_features = {}





for region_id, zone_ids in region_dict.items():
    tileid2oa2handmade_features[region_id] = {}
    for zone_id in zone_ids:
        zone_features = features_mapping.get(zone_id, {})
        tileid2oa2handmade_features[region_id][zone_id] = zone_features
# write to processed/tileid2oa2handmade_features.json
with open(os.path.join(root_path, "processed", "tileid2oa2handmade_features.json"), "w") as f:
    json.dump(tileid2oa2handmade_features, f)
print("Output -> tileid2oa2handmade_features.json, len is", len(tileid2oa2handmade_features))