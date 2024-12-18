import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.spatial import ConvexHull
import re

def plot_unfairness_vs_performance(steepness_factor, location_name, performance_type, metric_type):
    results_G = pd.read_csv(f"../evaluation/{steepness_factor}/{location_name}_G/{performance_type}/{location_name}_G_log.csv")
    results_DG = pd.read_csv(f"../evaluation/{steepness_factor}/{location_name}_DG/{performance_type}/{location_name}_DG_log.csv")
    results_NLG = pd.read_csv(f"../evaluation/{steepness_factor}/{location_name}_NLG/{performance_type}/{location_name}_NLG_log.csv")

    data = {
        f"{location_name}_G": (results_G['performance'], results_G['unfairness'], results_G['file_name']),
        f"{location_name}_DG": (results_DG['performance'], results_DG['unfairness'], results_DG['file_name']),
        f"{location_name}_NLG": (results_NLG['performance'], results_NLG['unfairness'], results_NLG['file_name'])
    }

    hull_colors = {f"{location_name}_G": "pink", f"{location_name}_DG": "green", f"{location_name}_NLG": "orange"}
    point_colors = {"unbiased": "red", "ascending": "blue", "descending": "grey"}
    point_shapes = {"0": "o", "1": "s", "2": "^"}

    plt.figure(figsize=(10, 6))



    def get_label(filename):
        if "gravity" in filename:
            if "1_ascending" in filename:
                return "ascending", "1"
            elif "1_descending" in filename:
                return "descending", "1"
            elif "2_ascending" in filename:
                return "ascending", "2"
            elif "2_descending" in filename:
                return "descending", "2"
            else:
                return "unbiased", "0"
        elif "od2flow" in filename:
            numbers = re.findall(r'\d+', filename)
            numbers = [int(num) for num in numbers]
            index = None
            if len(numbers) == 2:
                index = numbers[1]
            if index == 0:
                return "unbiased", "0"
            elif 1 <= index <= 6:
                return "ascending", "1"
            elif 13 <= index <= 18:
                return "ascending", "2"
            elif 7 <= index <= 12:
                return "descending", "1"
            elif 19 <= index <= 24:
                return "descending", "2"
            else:
                return "unknown", "unknown"





    hull_metrics = {}
    # Calculate height, width, and area of the hull
    def calculate_hull_metrics(points, label_suffix, store_metrics=True):
        if len(points) > 2:
            points = np.array(points)
            hull = ConvexHull(points)

            min_x, max_x = points[hull.vertices, 0].min(), points[hull.vertices, 0].max()
            min_y, max_y = points[hull.vertices, 1].min(), points[hull.vertices, 1].max()

            width = max_x - min_x
            height = max_y - min_y
            area = hull.volume

            if store_metrics:
                hull_metrics[f"{label}_{label_suffix}"] = {
                    "height": height,
                    "width": width,
                    "area": area
                }

            return hull, points, {"height": height, "width": width, "area": area}
        return None, None, None

    for label, (performance, unfairness, file_names) in data.items():
        grey_and_unbiased_points = []
        blue_and_unbiased_points = []

        for i, row in zip(range(len(file_names)), file_names):
            point_order, point_method = get_label(row)
            plt.scatter(performance[i], unfairness[i], color=point_colors[point_order], marker=point_shapes[point_method], alpha=0.7)
            if point_order in {"descending", "unbiased"}:
                grey_and_unbiased_points.append((performance[i], unfairness[i]))
            if point_order in {"ascending", "unbiased"}:
                blue_and_unbiased_points.append((performance[i], unfairness[i]))

        grey_metrics = None
        blue_metrics = None

        # Grey and unbiased points
        hull, points, grey_metrics = calculate_hull_metrics(grey_and_unbiased_points, "Descending and Unbiased")

        if hull:
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], linestyle="dotted", color=hull_colors[label])

        # Blue and unbiased points
        hull, points, blue_metrics = calculate_hull_metrics(blue_and_unbiased_points, "Ascending and Unbiased")

        if hull:
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], linestyle="solid", color=hull_colors[label])

        # Calculate the "all" metrics as the mean of grey and blue metrics
        if grey_metrics and blue_metrics:
            mean_metrics = {
                "height": (grey_metrics["height"] + blue_metrics["height"]) / 2,
                "width": (grey_metrics["width"] + blue_metrics["width"]) / 2,
                "area": (grey_metrics["area"] + blue_metrics["area"]) / 2
            }
            hull_metrics[f"{label}_all"] = mean_metrics

        # Overall hull for all points
        all_points = np.column_stack((performance, unfairness))
        hull, points, _ = calculate_hull_metrics(all_points, "all", store_metrics=False)
        if hull:
            # for simplex in hull.simplices:
            #     plt.plot(points[simplex, 0], points[simplex, 1], color=hull_colors[label])
            plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=hull_colors[label], alpha=0.2)

    # Display hull metrics for each model and shape
    for label, metrics in hull_metrics.items():
        print(f"Metrics for {label}:")
        print(f"  Height: {metrics['height']}")
        print(f"  Width: {metrics['width']}")
        print(f"  Area: {metrics['area']}\n")


    def sep_description(description):
        res = list(description.split("_"))
        if res[0] == "NY":
            res[0] = "New York"
        elif res[0] == "WA":
            res[0] = "Washington"
        if res[1] == "G":
            res[1] = "Gravity"
        elif res[1] == "DG":
            res[1] = "Deep Gravity"
        elif res[1] == "NLG":
            res[1] = "Non-Linear Gravity"
        if res[2] == "all":
            res[2] = "All (Mean)"
        return res


    dataset_locs = []
    model_types = []
    descriptions = []

    for key in hull_metrics:
        dataset_loc, model_type, description = sep_description(key)
        dataset_locs.append(dataset_loc)
        model_types.append(model_type)
        descriptions.append(description)

    df = pd.DataFrame({
        "Dataset Location": dataset_locs,
        "Model Type": model_types,
        "Description": descriptions,
        "Fairness Sensitivity (Height)": [metrics["height"] for metrics in hull_metrics.values()],
        "Performance Sensitivity (Width)": [metrics["width"] for metrics in hull_metrics.values()],
        "Robustness (Area)": [metrics["area"] for metrics in hull_metrics.values()]
    })

    pd.set_option("display.max_rows", None)  # Shows all rows
    pd.set_option("display.max_columns", None)  # Shows all columns
    pd.set_option("display.width", None)  # Adjusts width for better formatting
    print(df)
    df.to_csv(f"../evaluation/{steepness_factor}/{location_name}_{performance_type}_{metric_type}_{steepness_factor}.csv", index = False)





    # Define legend handles for hull colors
    hull_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=10, label=f"{location_name}_G Hull"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label=f"{location_name}_DG Hull"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label=f"{location_name}_NLG Hull")
    ]

    # Define legend handles for point combinations (color and shape)
    point_handles = [
        mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label="Unbiased"),
        mlines.Line2D([], [], color='blue', marker='s', linestyle='None', markersize=8, label="Ascending Method 1"),
        mlines.Line2D([], [], color='blue', marker='^', linestyle='None', markersize=8, label="Ascending Method 2"),
        mlines.Line2D([], [], color='grey', marker='s', linestyle='None', markersize=8, label="Descending Method 1"),
        mlines.Line2D([], [], color='grey', marker='^', linestyle='None', markersize=8, label="Descending Method 2")
    ]

    # Define legend handles for line styles
    line_handles = [
        mlines.Line2D([], [], color='black', linestyle='dotted', linewidth=2, label="Descending Method Hull"),
        mlines.Line2D([], [], color='black', linestyle='solid', linewidth=2, label="Ascending Method Hull")
    ]

    legend_handles = hull_handles + point_handles + line_handles
    plt.legend(handles=legend_handles, title="Legend", loc='upper left', bbox_to_anchor=(1, 1), borderpad=1.5, handlelength=2)

    plt.xlabel(f'Performance (Mean {performance_type})', fontsize=14, fontweight='bold')
    plt.ylabel(f'Unfairness ({metric_type})', fontsize=14, fontweight='bold')
    plt.title(f'Unfairness vs. Performance for {location_name}', fontsize=14, fontweight='bold')
    plt.grid(True)

    plt.savefig(f"../evaluation/{steepness_factor}/{location_name}_{performance_type}_{metric_type}_{steepness_factor}_plot.png", bbox_inches='tight')

    plt.show()



