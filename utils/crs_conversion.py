import json
import pyproj
from pyproj import Transformer

def convert_crs_epsg(input_file, output_file, input_crs_epsg, output_crs_epsg):
    """
    Convert coordinates in a GeoJSON file from one CRS to another.

    Parameters:
    input_file (str): Path to the input GeoJSON file.
    output_file (str): Path to the output GeoJSON file.
    input_crs_epsg (str): EPSG code for the input CRS.
    output_crs_epsg (str): EPSG code for the output CRS.

    Returns:
    GeoJSON file (.geojson): Longitude latitude converted GeoJSON file saved in the output_file.
    """
    # Load the GeoJSON data
    with open(input_file, 'r') as f:
        geojson_data = json.load(f)

    # Define the input and output CRS
    input_crs = pyproj.CRS.from_epsg(input_crs_epsg)
    output_crs = pyproj.CRS.from_epsg(output_crs_epsg)

    # Create a transformer object for the coordinate conversion
    transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)

    # Function to transform coordinates
    def transform_coordinates(coords):
        transformed_coords = []
        for coord in coords:
            x, y = coord
            newX, newY = transformer.transform(x, y)
            transformed_coords.append([newX, newY])
        return transformed_coords

    # Prepare the final GeoJSON data
    final_geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # Iterate over features and transform coordinates
    for feature in geojson_data['features']:
        # Extract and transform XCOORD and YCOORD
        xcoord = feature['properties']['XCOORD']
        ycoord = feature['properties']['YCOORD']
        lng, lat = transformer.transform(xcoord, ycoord)

        # Transform geometry coordinates
        geometry_type = feature['geometry']['type']
        if geometry_type == 'Polygon':
            coordinates = feature['geometry']['coordinates']
            new_coordinates = [transform_coordinates(ring) for ring in coordinates]
        elif geometry_type == 'MultiPolygon':
            coordinates = feature['geometry']['coordinates']
            new_coordinates = []
            for polygon in coordinates:
                temp = []
                for ring in polygon:
                    temp.append(transform_coordinates(ring))
                new_coordinates.append(temp)

            # new_coordinates = [transform_coordinates(ring) for polygon in coordinates for ring in polygon]
        else:
            new_coordinates = []

        # Add the new feature to the final GeoJSON
        new_feature = {
            "type": "Feature",
            "properties": {
                "GEOID": feature['properties']['GEOID'],
                "lng": lng,
                "lat": lat,
                "total_population": feature['properties']['total_population']
            },
            "geometry": {
                "type": geometry_type,
                "coordinates": new_coordinates
            }
        }

        final_geojson['features'].append(new_feature)

    # Save the updated GeoJSON to a new file
    with open(output_file, 'w') as f:
        json.dump(final_geojson, f, indent=2)

    print("GeoJSON coordinates and properties have been transformed and saved.")
