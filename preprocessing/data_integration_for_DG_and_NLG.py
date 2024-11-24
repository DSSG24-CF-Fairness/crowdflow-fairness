import multiprocessing
from IPython.display import display, HTML, Image, Markdown
import geopandas
from tqdm import tqdm
import csv
import geopandas as gpd
import pandas as pd
import json
import os
import pickle
import shutil

def run_data_integration_form_DG_and_NLG(folder_name):
    for dg_nlg_data_folder_id in range(21):

        if folder_name == "NY":
            dg_nlg_data_folder_name = "new_york" + str(dg_nlg_data_folder_id)
        elif folder_name == "NY_NEW":
            dg_nlg_data_folder_name = "new_york_new" + str(dg_nlg_data_folder_id)
        elif folder_name == "WA":
            dg_nlg_data_folder_name = "washington" + str(dg_nlg_data_folder_id)
        elif folder_name == "WA_NEW":
            dg_nlg_data_folder_name = "washington_new" + str(dg_nlg_data_folder_id)
        else:
            raise ValueError("folder_name is not valid")

        raw_data_root_folder_path = os.path.join("../data", folder_name)
        processed_data_root_folder_path = os.path.join("../processed_data", folder_name)
        dg_nlg_data_root_folder_path = os.path.join(processed_data_root_folder_path, dg_nlg_data_folder_name)
        dg_nlg_data_DG_folder_path = os.path.join(dg_nlg_data_root_folder_path, "processed_DG")
        dg_nlg_data_NLG_folder_path = os.path.join(dg_nlg_data_root_folder_path, "processed_NLG")

        os.makedirs(raw_data_root_folder_path, exist_ok=True)
        os.makedirs(processed_data_root_folder_path, exist_ok=True)
        os.makedirs(dg_nlg_data_root_folder_path, exist_ok=True)
        os.makedirs(dg_nlg_data_DG_folder_path, exist_ok=True)
        os.makedirs(dg_nlg_data_NLG_folder_path, exist_ok=True)



        # copy file from src to tar
        def copy_file(src_folder_path, tar_folder_path, file_name):
            src_file_path = os.path.join(src_folder_path, file_name)
            tar_file_path = os.path.join(tar_folder_path, file_name)
            shutil.copyfile(src_file_path, tar_file_path)
            print(f"Copy {file_name} file from {src_file_path} to {tar_file_path}")


        copy_file(raw_data_root_folder_path, dg_nlg_data_root_folder_path, "demographics.csv")
        copy_file(raw_data_root_folder_path, dg_nlg_data_root_folder_path, "features.csv")
        copy_file(raw_data_root_folder_path, dg_nlg_data_root_folder_path, "tessellation.geojson")

        copy_file(processed_data_root_folder_path, dg_nlg_data_root_folder_path, "test_region_index.csv")
        copy_file(processed_data_root_folder_path, dg_nlg_data_root_folder_path, "test_tile_geoids.csv")
        copy_file(processed_data_root_folder_path, dg_nlg_data_root_folder_path, "train_region_index.csv")
        copy_file(processed_data_root_folder_path, dg_nlg_data_root_folder_path, "train_tile_geoids.csv")


        # Merge train and test flow data

        train_flow_files = os.listdir(os.path.join(processed_data_root_folder_path, "train"))
        train_flow_files = [file for file in train_flow_files if "flow" in file]
        train_flow_files.sort()
        print(train_flow_files)

        # read processed_data_root_folder_path/train/train_flow.csv
        train_flow_file_path = os.path.join(processed_data_root_folder_path, "train",
                                            train_flow_files[dg_nlg_data_folder_id - 1])
        print(train_flow_file_path)
        train_flow_df = pd.read_csv(train_flow_file_path)
        print(train_flow_df.shape)

        # read test
        test_flow_file_path = os.path.join(processed_data_root_folder_path, "test", "test_flow.csv")
        test_flow_df = pd.read_csv(test_flow_file_path)
        print(test_flow_df.shape)

        # combine train and test flow
        flow_df = pd.concat([train_flow_df, test_flow_df])
        print(flow_df.shape)
        flow_df.to_csv(os.path.join(dg_nlg_data_root_folder_path, "flow.csv"), index=False)




        # convert to DG Data


        # test_tiles.csv
        test_tiles = pd.read_csv(os.path.join(dg_nlg_data_root_folder_path, "test_region_index.csv"))
        test_tiles.to_csv(os.path.join(dg_nlg_data_DG_folder_path, "test_tiles.csv"), header=False, index=False)
        print("Output -> test_tiles.csv, len is", len(test_tiles))

        # train_tiles.csv
        train_tiles = pd.read_csv(os.path.join(dg_nlg_data_root_folder_path, "train_region_index.csv"))
        train_tiles.to_csv(os.path.join(dg_nlg_data_DG_folder_path, "train_tiles.csv"), header=False, index=False)
        print("Output -> train_tiles.csv, len is", len(train_tiles))



        # Original data
        flow = pd.read_csv(os.path.join(dg_nlg_data_root_folder_path, "flow.csv"))
        demographics = pd.read_csv(os.path.join(dg_nlg_data_root_folder_path, "demographics.csv"))
        tessellation = geopandas.read_file(os.path.join(dg_nlg_data_root_folder_path, "tessellation.geojson"))
        features = pd.read_csv(os.path.join(dg_nlg_data_root_folder_path, "features.csv"))

        # find the intersection of GEOIDs
        GEOIDs_flow = set()
        for i, row in flow.iterrows():
            geoid_o = str(int(row["origin"]))
            geoid_d = str(int(row["destination"]))
            GEOIDs_flow.add(geoid_o)
            GEOIDs_flow.add(geoid_d)

        GEOIDs_demographics = set()
        for i, row in demographics.iterrows():
            geoid = str(int(row["geoid"]))
            GEOIDs_demographics.add(geoid)

        GEOIDs_features = set()
        for i, row in features.iterrows():
            geoid = str(int(row["geoid"]))
            GEOIDs_features.add(geoid)

        GEOIDs_tessellation = set()
        for i, row in tessellation.iterrows():
            geoid = str(int(row["GEOID"]))
            GEOIDs_tessellation.add(geoid)

        GEOIDs_intersected = GEOIDs_flow & GEOIDs_demographics & GEOIDs_features & GEOIDs_tessellation
        print(len(GEOIDs_intersected))


        # TO -> od2flow_new_york.csv.zip, res 138623 rows
        flows_oa = flow.rename(columns={"origin": "residence", "destination": "workplace", "flow": "commuters"})
        filtered_flows_oa = pd.DataFrame(columns=["residence", "workplace", "commuters"])

        # Initialize an empty list to collect rows
        filtered_rows = []

        # Iterate over each row in the DataFrame
        for i, row in tqdm(flows_oa.iterrows(), total=len(flows_oa), desc="-> flows_oa.csv.zip"):
            residence = str(int(row["residence"]))
            workplace = str(int(row["workplace"]))

            # Check if both residence and workplace are in GEOIDs_intersected
            if residence in GEOIDs_intersected and workplace in GEOIDs_intersected:
                # Append the row data as a tuple to the list
                filtered_rows.append([residence, workplace, row["commuters"]])

        # Convert the list of rows to a DataFrame at once
        filtered_flows_oa = pd.DataFrame(filtered_rows, columns=["residence", "workplace", "commuters"])

        filtered_flows_oa_grouped = filtered_flows_oa.groupby(['residence', 'workplace'], as_index=False).sum()
        filtered_flows_oa_grouped = filtered_flows_oa_grouped[["residence", "workplace", "commuters"]]
        print("Output -> flows_oa.csv.zip, len is", len(filtered_flows_oa_grouped))
        filtered_flows_oa_grouped.to_csv(os.path.join(dg_nlg_data_DG_folder_path, "flows_oa.csv.zip"), index=False)
        filtered_flows_oa_grouped

        # TO -> od2flow.pkl, res 138623 rows
        od2flow = {(str(int(row['residence'])), str(int(row['workplace']))): row['commuters'] for _, row in
                   filtered_flows_oa_grouped.iterrows()}

        with open(os.path.join(dg_nlg_data_DG_folder_path, 'od2flow.pkl'), 'wb') as f:
            pickle.dump(od2flow, f)
        print("Output -> od2flow.pkl, len is", len(od2flow))
        od2flow

        # TO -> oa2centroid.pkl, res 1429 rows
        oa2centroid = dict()
        # Extracting lng and lat of each GEOID into a dictionary

        for i, row in tqdm(tessellation.iterrows(), total=len(tessellation), desc="-> oa2centroid.pkl"):
            geoid = str(int(row["GEOID"]))
            if geoid in GEOIDs_intersected:
                oa2centroid[geoid] = [row["lng"], row["lat"]]
        with open(os.path.join(dg_nlg_data_DG_folder_path, "oa2centroid.pkl"), "wb") as f:
            pickle.dump(oa2centroid, f)
        print("Output -> oa2centroid.pkl, len is", len(oa2centroid))
        oa2centroid

        # TO -> oa2features.pkl
        oa2features = dict()

        # for i, row in features.iterrows():
        for i, row in tqdm(features.iterrows(), total=len(features), desc="-> oa2features.pkl"):
            geoid = str(int(row["geoid"]))

            if geoid in GEOIDs_intersected:
                # Select all columns except the first one
                feature = row.iloc[1:].tolist()

                oa2features[geoid] = feature

                oa2features[geoid] = feature

        with open(os.path.join(dg_nlg_data_DG_folder_path, "oa2features.pkl"), "wb") as f:
            pickle.dump(oa2features, f)
        print("Output -> oa2features.pkl, len is", len(oa2features))
        oa2features



        # TO -> oa_gdf.csv.gz
        oa_gdf = []
        # for i, geo_code in enumerate(GEOIDs_intersected):
        for i, geo_code in tqdm(enumerate(GEOIDs_intersected), total=len(GEOIDs_intersected), desc="-> oa_gdf.csv.gz"):
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
        oa_gdf.to_csv(os.path.join(dg_nlg_data_DG_folder_path, "oa_gdf.csv.gz"), index=False)
        print("Output -> oa_gdf.csv.gz, len is", len(oa_gdf))
        oa_gdf



        # TO -> tileid2oa2handmade_features.json
        features_mapping = {}
        for index, row in tqdm(features.iterrows(), total=len(features), desc="featrues -> tileid2oa2handmade_features.json"):
            geoid = int(row["geoid"])

            # Proceed only if geoid is in the intersected set
            if str(geoid) in GEOIDs_intersected:
                feature_data = row.drop("geoid").to_dict()

                # Convert each feature's value to a list containing that value
                feature_data = {key: [value] for key, value in feature_data.items()}
                features_mapping[str(geoid)] = feature_data

        for geoid in GEOIDs_intersected:
            if str(geoid) not in features_mapping:
                features_mapping[str(geoid)] = {}

        print("features_mapping size", len(features_mapping))

        region_dict = {}

        with open(os.path.join(dg_nlg_data_root_folder_path, "train_region_index.csv"), mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                region_dict[int(row[0])] = {}

        with open(os.path.join(dg_nlg_data_root_folder_path, "test_region_index.csv"), mode='r') as file:
            # the test_region_index is a csv file has one column called region_index, store all region_index to a list except for the header
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                region_dict[int(row[0])] = {}

        # sort keys in region_dict
        region_dict = dict(sorted(region_dict.items()))

        valid_geoids = set()
        invalid_geoids = set()
        with open(os.path.join(dg_nlg_data_root_folder_path, "test_tile_geoids.csv"), mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                region_index = int(row[0])  # Convert region_index to integer
                geoids = row[1].split(',')  # Split the geoids into a list
                used_geoids = []
                for geoid in geoids:
                    geoid = geoid.strip()
                    if geoid in GEOIDs_intersected:
                        valid_geoids.add(geoid)
                        used_geoids.append(geoid)
                    else:
                        invalid_geoids.add(geoid)
                region_dict[region_index] = used_geoids  # Store in the dictionary

        with open(os.path.join(dg_nlg_data_root_folder_path, "train_tile_geoids.csv"), mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                region_index = int(row[0])  # Convert region_index to integer
                geoids = row[1].split(',')  # Split the geoids into a list
                used_geoids = []
                for geoid in geoids:
                    geoid = geoid.strip()
                    if geoid in GEOIDs_intersected:
                        valid_geoids.add(geoid)
                        used_geoids.append(geoid)
                    else:
                        invalid_geoids.add(geoid)
                region_dict[region_index] = used_geoids

        tileid2oa2handmade_features = {}
        used_geoids = set()

        for region_id, zone_ids in region_dict.items():
            tileid2oa2handmade_features[region_id] = {}
            for zone_id in zone_ids:
                zone_features = features_mapping.get(zone_id, {})
                tileid2oa2handmade_features[region_id][zone_id] = zone_features
                used_geoids.add(zone_id)
        # write to processed/tileid2oa2handmade_features.json
        with open(os.path.join(dg_nlg_data_DG_folder_path, "tileid2oa2handmade_features.json"), "w") as f:
            json.dump(tileid2oa2handmade_features, f)

        print("Output -> tileid2oa2handmade_features.json, used geoid counts is", len(used_geoids))
        print("Output -> tileid2oa2handmade_features.json, total used tile counts is", len(tileid2oa2handmade_features))

        tileid2oa2handmade_features


        # convert to NLG Data


        tessellation_data = gpd.read_file(os.path.join(dg_nlg_data_root_folder_path, "tessellation.geojson"))
        tessellation_data.to_file(os.path.join(dg_nlg_data_root_folder_path, "processed_NLG", "tessellation.geojson"),
                                  driver="GeoJSON")

        # clean handmade features
        with open(os.path.join(dg_nlg_data_root_folder_path, "processed_DG", "tileid2oa2handmade_features.json"), "r") as f:
            tileid2oa2handmade_features = json.load(f)

        geoid_to_population = {}
        for tileid in tileid2oa2handmade_features:
            for oa in tileid2oa2handmade_features[tileid]:
                geoid_to_population[oa] = tileid2oa2handmade_features[tileid][oa]['total_population']
                tileid2oa2handmade_features[tileid][oa] = {'total_population': geoid_to_population[oa]}
        with open(os.path.join(dg_nlg_data_root_folder_path, "processed_NLG", "tileid2oa2handmade_features.json"), "w") as f:
            json.dump(tileid2oa2handmade_features, f)

        # clean features
        oa2features = pickle.load(open(os.path.join(dg_nlg_data_root_folder_path, "processed_DG", "oa2features.pkl"), "rb"))

        for oa in oa2features:
            oa2features[oa] = geoid_to_population[oa]
        pickle.dump(oa2features, open(os.path.join(dg_nlg_data_root_folder_path, "processed_NLG", "oa2features.pkl"), "wb"))

        # copy rest of the files to processed_NLG
        flows_oa = pd.read_csv(os.path.join(dg_nlg_data_root_folder_path, "processed_DG", "flows_oa.csv.zip"))
        flows_oa.to_csv(os.path.join(dg_nlg_data_root_folder_path, "processed_NLG", "flows_oa.csv.zip"), index=False)

        od2flow = pickle.load(open(os.path.join(dg_nlg_data_root_folder_path, "processed_DG", "od2flow.pkl"), "rb"))
        pickle.dump(od2flow, open(os.path.join(dg_nlg_data_root_folder_path, "processed_NLG", "od2flow.pkl"), "wb"))

        oa2centroid = pickle.load(open(os.path.join(dg_nlg_data_root_folder_path, "processed_DG", "oa2centroid.pkl"), "rb"))
        pickle.dump(oa2centroid, open(os.path.join(dg_nlg_data_root_folder_path, "processed_NLG", "oa2centroid.pkl"), "wb"))

        oa_gdf = pd.read_csv(os.path.join(dg_nlg_data_root_folder_path, "processed_DG", "oa_gdf.csv.gz"))
        oa_gdf.to_csv(os.path.join(dg_nlg_data_root_folder_path, "processed_NLG", "oa_gdf.csv.gz"), index=False)

        # copy test_tiles.csv and train_tiles.csv to processed_NLG
        shutil.copy(os.path.join(dg_nlg_data_root_folder_path, "processed_DG", "test_tiles.csv"),
                    os.path.join(dg_nlg_data_root_folder_path, "processed_NLG", "test_tiles.csv"))
        shutil.copy(os.path.join(dg_nlg_data_root_folder_path, "processed_DG", "train_tiles.csv"),
                    os.path.join(dg_nlg_data_root_folder_path, "processed_NLG", "train_tiles.csv"))


if __name__ == '__main__':
    datasets = ["NY_NEW", "WA_NEW"]
    # Create a pool of processes, one for each dataset
    processes = []
    for dataset in datasets:
        p = multiprocessing.Process(target=run_data_integration_form_DG_and_NLG, args=(dataset,))
        processes.append(p)
        p.start()  # Start each process

    # Wait for all processes to complete
    for p in processes:
        p.join()