import matplotlib.pyplot as plt

fractions = [25, 50, 75, 100]

data = {
    "GraphSAGE": {
        "forward_memory": [4.47, 8.24, 12.01, 15.77],
        "backward_memory": [5.17, 8.94, 12.72, 16.47],
        "epoch_time": [0.0118, 0.0192, 0.0325, 0.0403],
        "accuracy": [0.6705, 0.7559, 0.7632, 0.7890],
    },
    "GraphSAINT": {
        "forward_memory": [1.96, 3.05, 3.35, 3.46],
        "backward_memory": [2.66, 3.76, 4.06, 4.17],
        "epoch_time": [0.0202, 0.0301, 0.0334, 0.0325],
        "accuracy": [0.7326, 0.7598, 0.7315, 0.8130],
    },
    "Cluster-GCN": {
        "forward_memory": [1.14, 1.56, 1.99, 2.42],
        "backward_memory": [1.84, 2.26, 2.70, 3.13],
        "epoch_time": [0.0301, 0.0429, 0.0629, 0.0689],
        "accuracy": [0.7054, 0.7559, 0.7553, 0.8030],
    },
}


def make_line_plot(metric_key: str, ylabel: str, title: str, filename: str):
    plt.figure(figsize=(8, 5))

    for model_name, metrics in data.items():
        plt.plot(fractions, metrics[metric_key], marker="o", label=model_name)

    plt.xlabel("Graph Size (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(fractions)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


make_line_plot(
    metric_key="forward_memory",
    ylabel="Forward Memory (MB)",
    title="Forward Computational Memory vs Graph Size",
    filename="forward_memory_vs_graph_size.png",
)

make_line_plot(
    metric_key="backward_memory",
    ylabel="Backward Memory (MB)",
    title="Backward Computational Memory vs Graph Size",
    filename="backward_memory_vs_graph_size.png",
)

make_line_plot(
    metric_key="epoch_time",
    ylabel="Average Epoch Time (s)",
    title="Training Time vs Graph Size",
    filename="epoch_time_vs_graph_size.png",
)

make_line_plot(
    metric_key="accuracy",
    ylabel="Best Test Accuracy",
    title="Accuracy vs Graph Size",
    filename="accuracy_vs_graph_size.png",
)

