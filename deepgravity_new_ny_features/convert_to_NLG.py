def convert_to_NLG(dataset_name):
    import geopandas as gpd
    import pandas as pd
    import json
    import os
    import pickle
    import shutil
    from IPython.display import display

    root_path = os.path.join("data", dataset_name)
    os.makedirs(os.path.join(root_path, "processed_NLG"), exist_ok=True)

    # import geojson for population data
    # if "washington" in root_path:
    tessellation_data = gpd.read_file(os.path.join(root_path, "tessellation.geojson"))
    #
    # if "new_york" in root_path:
    #     population_data = pd.read_csv(os.path.join(root_path, "population_new_york.csv"))



    # create a mapping of GEOID -> total_population using tessellation_data
    geoid_to_population = {}
    # if "washington" in root_path:
    for i in tessellation_data.index:
        geoid = str(int(tessellation_data.at[i, "GEOID"]))
        population = int(tessellation_data.at[i, "total_population"])
        geoid_to_population[geoid] = population
    # if "new_york" in root_path:
    #     for i, row in population_data.iterrows():
    #         geoid = str(int(row["geoid"]))
    #         population = row["total_population"] # it is format as 1,000, convert to int
    #         population = int(population.replace(",", ""))
    #         geoid_to_population[geoid] = population



    # clean handmade features
    with open(os.path.join(root_path, "processed_DG", "tileid2oa2handmade_features.json"), "r") as f:
        tileid2oa2handmade_features = json.load(f)


    for tileid in tileid2oa2handmade_features:
        for oa in tileid2oa2handmade_features[tileid]:
            tileid2oa2handmade_features[tileid][oa] = [geoid_to_population[oa]]
    with open(os.path.join(root_path, "processed_NLG", "tileid2oa2handmade_features.json"), "w") as f:
        json.dump(tileid2oa2handmade_features, f)



    # clean features
    oa2features = pickle.load(open(os.path.join(root_path, "processed_DG", "oa2features.pkl"), "rb"))

    for oa in oa2features:
        oa2features[oa] = [geoid_to_population[oa]]
    pickle.dump(oa2features, open(os.path.join(root_path, "processed_NLG", "oa2features.pkl"), "wb"))






    # copy rest of the files to processed_NLG
    flows_oa = pd.read_csv(os.path.join(root_path, "processed_DG", "flows_oa.csv.zip"))
    flows_oa.to_csv(os.path.join(root_path, "processed_NLG", "flows_oa.csv.zip"), index=False)

    od2flow = pickle.load(open(os.path.join(root_path, "processed_DG", "od2flow.pkl"), "rb"))
    pickle.dump(od2flow, open(os.path.join(root_path, "processed_NLG", "od2flow.pkl"), "wb"))

    oa2centroid = pickle.load(open(os.path.join(root_path, "processed_DG", "oa2centroid.pkl"), "rb"))
    pickle.dump(oa2centroid, open(os.path.join(root_path, "processed_NLG", "oa2centroid.pkl"), "wb"))

    oa_gdf = pd.read_csv(os.path.join(root_path, "processed_DG", "oa_gdf.csv.gz"))
    oa_gdf.to_csv(os.path.join(root_path, "processed_NLG", "oa_gdf.csv.gz"), index=False)


    # copy test_tiles.csv and train_tiles.csv to processed_NLG
    shutil.copy(os.path.join(root_path, "processed_DG", "test_tiles.csv"), os.path.join(root_path, "processed_NLG", "test_tiles.csv"))
    shutil.copy(os.path.join(root_path, "processed_DG", "train_tiles.csv"), os.path.join(root_path, "processed_NLG", "train_tiles.csv"))