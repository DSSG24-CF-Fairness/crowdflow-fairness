import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.spatial import ConvexHull
import re






def plot_fairness_vs_accuracy(location_name, accuracy_type, metric_type):
    results_G = pd.read_csv(f"../evaluation/{location_name}_G/{accuracy_type}/{metric_type}/{location_name}_G_log.csv")
    results_DG = pd.read_csv(f"../evaluation/{location_name}_DG/{accuracy_type}/{metric_type}/{location_name}_DG_log.csv")
    results_NLG = pd.read_csv(f"../evaluation/{location_name}_NLG/{accuracy_type}/{metric_type}/{location_name}_NLG_log.csv")

    data = {
        f"{location_name}_G": (results_G['accuracy'], results_G['fairness']),
        f"{location_name}_DG": (results_DG['accuracy'], results_DG['fairness']),
        f"{location_name}_NLG": (results_NLG['accuracy'], results_NLG['fairness'])
    }

    # display(data)

    hull_colors = {f"{location_name}_G": "pink", f"{location_name}_DG": "green", f"{location_name}_NLG": "orange"}
    point_colors = {"unbiased": "red", "ascending": "blue", "descending": "grey"}
    point_shapes = {"0": "o", "1": "s", "2": "^"}

    plt.figure(figsize=(10, 6))

    def get_label(filename):
        # Gravity log files
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

        # DG/NLG log files
        elif "od2flow" in filename:
            # Extract the index from the filename using a regular expression
            numbers = re.findall(r'\d+', filename)
            # Convert numbers to integers (optional)
            numbers = [int(num) for num in numbers]

            index = None
            if len(numbers) == 2:
                index = numbers[1]

            if index == 0:
                return "unbiased", "0"
            elif 1 <= index <= 5:
                return "ascending", "1"
            elif 11 <= index <= 15:
                return "ascending", "2"
            elif 6 <= index <= 10:
                return "descending", "1"
            elif 16 <= index <= 20:
                return "descending", "2"
            else:
                return "unknown", "unknown"


    # Plot points for each dataset
    for label, (accuracy, fairness) in data.items():
        # Loop through the dataset and apply get_label based on each filename
        for i, row in results_G.iterrows():
            point_order, point_method = get_label(row['file_name'])
            plt.scatter(row['accuracy'], row['fairness'], color=point_colors[point_order], marker=point_shapes[point_method], label=point_order if i == 0 else "", alpha=0.7)

        for i, row in results_DG.iterrows():
            point_order, point_method = get_label(row['file_name'])  # Pass the filename to get_label
            plt.scatter(row['accuracy'], row['fairness'], color=point_colors[point_order], marker=point_shapes[point_method], label=point_order if i == 0 else "", alpha=0.7)

        for i, row in results_NLG.iterrows():
            point_order, point_method = get_label(row['file_name'])
            plt.scatter(row['accuracy'], row['fairness'], color=point_colors[point_order], marker=point_shapes[point_method], label=point_order if i == 0 else "", alpha=0.7)


        # Convex hull for each dataset
        points = np.column_stack((accuracy, fairness))
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], color=hull_colors[label],
                     linewidth=1)  # Hull edges in dataset color
        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=hull_colors[label], alpha=0.2)  # Hull fill

    hull_metrics = {}  # To store hull measurements for each dataset
    for label, (accuracy, fairness) in data.items():
        points = np.column_stack((accuracy, fairness))
        hull = ConvexHull(points)

        # Calculate height, width, and area of the hull
        min_x, max_x = points[hull.vertices, 0].min(), points[hull.vertices, 0].max()
        min_y, max_y = points[hull.vertices, 1].min(), points[hull.vertices, 1].max()

        width = max_x - min_x
        height = max_y - min_y
        area = hull.volume  # ConvexHull's `volume` property gives the 2D area for 2D points

        hull_metrics[label] = {
            "height": height,
            "width": width,
            "area": area
        }
    # Display hull metrics for each model
    for label, metrics in hull_metrics.items():
        print(f"Metrics for {label}:")
        print(f"  Height: {metrics['height']}")
        print(f"  Width: {metrics['width']}")
        print(f"  Area: {metrics['area']}\n")

    plt.xlabel(f'Performance (Mean {accuracy_type})')
    plt.ylabel(f'Fairness ({metric_type})')
    plt.title(f'Fairness vs. Performance for {location_name}')
    plt.ylim(plt.ylim())
    plt.xlim(plt.xlim())



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

    # Combine all legend handles
    legend_handles = hull_handles + point_handles

    # Add the legend to the plot
    plt.legend(handles=legend_handles, title="Legend", loc='upper left', bbox_to_anchor=(1, 1), borderpad=1.5, handlelength=2)



    plt.grid(True)

    plt.savefig(f"../evaluation/{location_name}_{accuracy_type}_{metric_type}_plot.png", bbox_inches='tight')

    plt.show()




location_name = 'WA'
accuracy_type = 'CPC'
metric_type = 'kl_divergence'
plot_fairness_vs_accuracy(location_name, accuracy_type, metric_type)