import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

def plot_fairness_vs_accuracy(location_name, file_name):
    results_G = pd.read_csv(f"../evaluation/{location_name}_G_log.csv")
    results_DG = pd.read_csv(f"../evaluation/{location_name}_DG_log.csv")
    results_NLG = pd.read_csv(f"../evaluation/{location_name}_NLG_log.csv")

    data = {
        f"{location_name}_G": (results_G['accuracy'], results_G['fairness']),
        f"{location_name}_DG": (results_DG['accuracy'], results_DG['fairness']),
        f"{location_name}_NLG": (results_NLG['accuracy'], results_NLG['fairness'])
    }

    hull_colors = {f"{location_name}_G": "yellow", f"{location_name}_DG": "green", f"{location_name}_NLG": "orange"}
    point_colors = {"unbiased": "red", "ascending": "blue", "descending": "grey"}

    plt.figure(figsize=(10, 6))

    # Function to determine label based on filename
    def get_label(filename):
        index = int(filename.replace(f"{file_name}", ""))
        if index == 0:
            return "unbiased"
        elif 1 <= index <= 5 or 11 <= index <= 15:
            return "ascending"
        elif 6 <= index <= 10 or 16 <= index <= 20:
            return "descending"

    # Create handles for the legend (for both point and hull colors)
    legend_handles = {
        "hull_G": plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label=f"{location_name}_G"),
        "hull_DG": plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label=f"{location_name}_DG"),
        "hull_NLG": plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label=f"{location_name}_NLG"),
        "unbiased": plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label="Unbiased"),
        "ascending": plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label="Ascending"),
        "descending": plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label="Descending")
    }

    # Plot points for each dataset
    for label, (accuracy, fairness) in data.items():
        for i, (acc, fair) in enumerate(zip(accuracy, fairness)):
            # Determine label for the points (unbiased, ascending, or descending)
            filename = f"{file_name}{i}"
            point_label = get_label(filename)

            # Plot the points with the appropriate color for each label
            plt.scatter(acc, fair, color=point_colors[point_label], label=point_label if i == 0 else "", alpha=0.7)

        # Convex hull for each dataset
        points = np.column_stack((accuracy, fairness))
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], color=hull_colors[label], linewidth=1)  # Hull edges in dataset color
        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=hull_colors[label], alpha=0.2)  # Hull fill

    plt.xlabel('Accuracy (Mean CPC)')
    plt.ylabel('Fairness (KL Divergence for CPC)')
    plt.title(f'Fairness vs. Accuracy for {location_name}')
    plt.ylim(plt.ylim())
    plt.xlim(plt.xlim())

    # Add the custom legend with padding
    plt.legend(handles=[
        legend_handles["hull_G"],
        legend_handles["hull_DG"],
        legend_handles["hull_NLG"],
        legend_handles["unbiased"],
        legend_handles["ascending"],
        legend_handles["descending"]
    ], title="Legend", loc='upper left', bbox_to_anchor=(1, 1), borderpad=1.5, handlelength=2)

    plt.grid(True)

    plt.savefig(f"../evaluation/{location_name}_plot.png", bbox_inches='tight')

    plt.show()




plot_fairness_vs_accuracy('WA', 'washington')
plot_fairness_vs_accuracy('NY', 'new_york')
