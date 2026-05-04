import matplotlib.pyplot as plt

# ============================================================
# FINAL RESULT PLOTTING SCRIPT
# Generates:
# 1. Cora metric plots
# 2. PubMed-tripled metric plots
# 3. Cora trade-off plot
# 4. PubMed trade-off plot
# 5. Bias-variance plots
# ============================================================

# ------------------------------------------------------------
# CORA RESULTS
# ------------------------------------------------------------

cora_fractions = [25, 50, 75, 100]
cora_nodes = [677, 1354, 2031, 2708]

cora_data = {
    "GraphSAGE": {
        "forward_memory": [4.47, 8.24, 12.01, 15.77],
        "backward_memory": [5.17, 8.94, 12.72, 16.47],
        "epoch_time": [0.0112, 0.0191, 0.0275, 0.0477],
        "accuracy": [0.6705, 0.7583, 0.7619, 0.7900],
        "accuracy_std": [0.0248, 0.0093, 0.0108, 0.0116],
        "bias_sq": [0.457220, 0.366618, 0.407386, 0.304367],
        "variance": [0.043654, 0.008982, 0.013003, 0.036117],
    },
    "GraphSAINT": {
        "forward_memory": [1.96, 3.05, 3.35, 3.46],
        "backward_memory": [2.66, 3.76, 4.06, 4.17],
        "epoch_time": [0.0269, 0.0378, 0.0393, 0.0510],
        "accuracy": [0.6938, 0.7266, 0.7741, 0.7750],
        "accuracy_std": [0.0157, 0.0107, 0.0082, 0.0049],
        "bias_sq": [0.438264, 0.359692, 0.322117, 0.299606],
        "variance": [0.032371, 0.046101, 0.026298, 0.033968],
    },
    "Cluster-GCN": {
        "forward_memory": [1.14, 1.56, 1.99, 2.42],
        "backward_memory": [1.84, 2.26, 2.70, 3.13],
        "epoch_time": [0.0480, 0.0600, 0.0717, 0.0941],
        "accuracy": [0.7074, 0.7539, 0.7794, 0.7975],
        "accuracy_std": [0.0104, 0.0091, 0.0154, 0.0038],
        "bias_sq": [0.400354, 0.316158, 0.308887, 0.276215],
        "variance": [0.031866, 0.047360, 0.023633, 0.023924],
    },
    "GAT": {
        "forward_memory": [8.55, 16.77, 24.99, 33.14],
        "backward_memory": [8.90, 17.12, 25.35, 33.49],
        "epoch_time": [0.0097, 0.0137, 0.0186, 0.0220],
        "accuracy": [0.6705, 0.7632, 0.7698, 0.7975],
        "accuracy_std": [0.0407, 0.0227, 0.0228, 0.0144],
        "bias_sq": [0.475132, 0.481200, 0.451738, 0.470470],
        "variance": [0.009021, 0.006705, 0.015599, 0.015581],
        "attention_memory": [0.11, 0.24, 0.36, 0.46],
    },
}

# ------------------------------------------------------------
# PUBMED TRIPLED-FEATURE RESULTS
# ------------------------------------------------------------

pubmed_fractions = [25, 50, 75, 100]
pubmed_nodes = [4929, 9858, 14788, 19717]

pubmed_data = {
    "GraphSAGE": {
        "forward_memory": [29.13, 57.69, 86.42, 115.34],
        "backward_memory": [29.87, 58.43, 87.16, 116.08],
        "epoch_time": [0.0642, 0.1542, 0.2731, 0.4257],
        "accuracy": [0.5896, 0.6890, 0.7141, 0.7547],
        "accuracy_std": [0.0207, 0.0114, 0.0110, 0.0084],
        "bias_sq": [0.504440, 0.424047, 0.399767, 0.365364],
        "variance": [0.017172, 0.004535, 0.003120, 0.004106],
    },
    "GraphSAINT": {
        "forward_memory": [3.55, 3.87, 4.00, 4.05],
        "backward_memory": [4.29, 4.61, 4.73, 4.79],
        "epoch_time": [0.0314, 0.0381, 0.0395, 0.0383],
        "accuracy": [0.4882, 0.6365, 0.6675, 0.6562],
        "accuracy_std": [0.0162, 0.0142, 0.0132, 0.0575],
        "bias_sq": [0.585036, 0.476373, 0.433376, 0.432936],
        "variance": [0.033521, 0.027733, 0.013694, 0.028246],
    },
    "Cluster-GCN": {
        "forward_memory": [3.96, 7.23, 10.51, 13.80],
        "backward_memory": [4.70, 7.97, 11.24, 14.53],
        "epoch_time": [0.0826, 0.1730, 0.3646, 0.5607],
        "accuracy": [0.6664, 0.7450, 0.7524, 0.7745],
        "accuracy_std": [0.0210, 0.0140, 0.0034, 0.0080],
        "bias_sq": [None, None, None, 0.327143],
        "variance": [None, None, None, 0.011703],
    },
    "GAT": {
        "forward_memory": [61.32, 123.13, 185.81, 249.49],
        "backward_memory": [61.69, 123.50, 186.17, 249.86],
        "epoch_time": [0.0459, 0.0587, 0.0965, 0.1872],
        "accuracy": [0.6624, 0.7100, 0.7199, 0.7450],
        "accuracy_std": [0.0139, 0.0162, 0.0094, 0.0067],
        "bias_sq": [None, None, None, 0.380635],
        "variance": [None, None, None, 0.001520],
        "attention_memory": [0.36, 1.09, 2.19, 3.72],
    },
}


# ============================================================
# PLOT FUNCTIONS
# ============================================================

def make_line_plot(x_values, data, metric_key, xlabel, ylabel, title, filename):
    plt.figure(figsize=(8, 5))

    for model_name, metrics in data.items():
        if metric_key in metrics:
            plt.plot(
                x_values,
                metrics[metric_key],
                marker="o",
                linewidth=2,
                markersize=6,
                label=model_name,
            )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x_values)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def make_accuracy_plot_with_error(x_values, data, xlabel, title, filename):
    plt.figure(figsize=(8, 5))

    for model_name, metrics in data.items():
        y = metrics["accuracy"]
        yerr = metrics.get("accuracy_std", None)

        if yerr is not None:
            plt.errorbar(
                x_values,
                y,
                yerr=yerr,
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=4,
                label=model_name,
            )
        else:
            plt.plot(
                x_values,
                y,
                marker="o",
                linewidth=2,
                markersize=6,
                label=model_name,
            )

    plt.xlabel(xlabel)
    plt.ylabel("Test Accuracy")
    plt.title(title)
    plt.xticks(x_values)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def make_tradeoff_plot(data, title, filename, index=-1):
    plt.figure(figsize=(8, 5))

    for model_name, metrics in data.items():
        x = metrics["forward_memory"][index]
        y = metrics["accuracy"][index]

        plt.scatter(x, y, s=90, label=model_name)
        plt.text(x, y, f" {model_name}", fontsize=9, va="center")

    plt.xlabel("Forward Computational Memory (MB)")
    plt.ylabel("Test Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def make_bias_variance_plot(data, title, filename, index=-1):
    plt.figure(figsize=(8, 5))

    for model_name, metrics in data.items():
        bias_values = metrics.get("bias_sq", None)
        variance_values = metrics.get("variance", None)

        if bias_values is None or variance_values is None:
            continue

        bias = bias_values[index]
        variance = variance_values[index]

        if bias is None or variance is None:
            continue

        plt.scatter(bias, variance, s=90, label=model_name)
        plt.text(bias, variance, f" {model_name}", fontsize=9, va="center")

    plt.xlabel("Empirical Bias²")
    plt.ylabel("Empirical Variance")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def make_attention_plot(x_values, data, xlabel, title, filename):
    plt.figure(figsize=(8, 5))

    for model_name, metrics in data.items():
        if "attention_memory" in metrics:
            plt.plot(
                x_values,
                metrics["attention_memory"],
                marker="o",
                linewidth=2,
                markersize=6,
                label=model_name,
            )

    plt.xlabel(xlabel)
    plt.ylabel("Attention Memory (MB)")
    plt.title(title)
    plt.xticks(x_values)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# GENERATE CORA PLOTS
# ============================================================

make_line_plot(
    cora_fractions,
    cora_data,
    "forward_memory",
    "Graph Size (%)",
    "Forward Memory (MB)",
    "Cora: Forward Computational Memory vs Graph Size",
    "cora_forward_memory_vs_graph_size.png",
)

make_line_plot(
    cora_fractions,
    cora_data,
    "backward_memory",
    "Graph Size (%)",
    "Backward Memory (MB)",
    "Cora: Backward Computational Memory vs Graph Size",
    "cora_backward_memory_vs_graph_size.png",
)

make_line_plot(
    cora_fractions,
    cora_data,
    "epoch_time",
    "Graph Size (%)",
    "Average Epoch Time (s)",
    "Cora: Training Time vs Graph Size",
    "cora_epoch_time_vs_graph_size.png",
)

make_accuracy_plot_with_error(
    cora_fractions,
    cora_data,
    "Graph Size (%)",
    "Cora: Test Accuracy Across Graph Sizes",
    "cora_accuracy_vs_graph_size.png",
)

make_attention_plot(
    cora_fractions,
    cora_data,
    "Graph Size (%)",
    "Cora: GAT Attention Memory vs Graph Size",
    "cora_attention_memory_vs_graph_size.png",
)

make_tradeoff_plot(
    cora_data,
    "Cora: Memory–Accuracy Trade-off at 100% Graph Size",
    "cora_memory_accuracy_tradeoff_100.png",
)

make_bias_variance_plot(
    cora_data,
    "Cora: Bias–Variance Trade-off at 100% Graph Size",
    "cora_bias_variance_100.png",
)


# ============================================================
# GENERATE PUBMED PLOTS
# ============================================================

make_line_plot(
    pubmed_fractions,
    pubmed_data,
    "forward_memory",
    "Graph Size (%)",
    "Forward Memory (MB)",
    "PubMed Tripled: Forward Computational Memory vs Graph Size",
    "pubmed_forward_memory_vs_graph_size.png",
)

make_line_plot(
    pubmed_fractions,
    pubmed_data,
    "backward_memory",
    "Graph Size (%)",
    "Backward Memory (MB)",
    "PubMed Tripled: Backward Computational Memory vs Graph Size",
    "pubmed_backward_memory_vs_graph_size.png",
)

make_line_plot(
    pubmed_fractions,
    pubmed_data,
    "epoch_time",
    "Graph Size (%)",
    "Average Epoch Time (s)",
    "PubMed Tripled: Training Time vs Graph Size",
    "pubmed_epoch_time_vs_graph_size.png",
)

make_accuracy_plot_with_error(
    pubmed_fractions,
    pubmed_data,
    "Graph Size (%)",
    "PubMed Tripled: Test Accuracy Across Graph Sizes",
    "pubmed_accuracy_vs_graph_size.png",
)

make_attention_plot(
    pubmed_fractions,
    pubmed_data,
    "Graph Size (%)",
    "PubMed Tripled: GAT Attention Memory vs Graph Size",
    "pubmed_attention_memory_vs_graph_size.png",
)

make_tradeoff_plot(
    pubmed_data,
    "PubMed Tripled: Memory–Accuracy Trade-off at 100% Graph Size",
    "pubmed_memory_accuracy_tradeoff_100.png",
)

make_bias_variance_plot(
    pubmed_data,
    "PubMed Tripled: Bias–Variance Trade-off at 100% Graph Size",
    "pubmed_bias_variance_100.png",
)

print("All final plots generated successfully.")