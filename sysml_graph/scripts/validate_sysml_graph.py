import csv
import json
from pathlib import Path
from collections import Counter, defaultdict, deque

OUTPUT_DIR = Path("sysml_graph/output")


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def weakly_connected_components(node_ids, edges):
    graph = defaultdict(set)

    for edge in edges:
        src = edge["source_id"]
        tgt = edge["target_id"]
        graph[src].add(tgt)
        graph[tgt].add(src)

    visited = set()
    sizes = []

    for node in node_ids:
        if node in visited:
            continue

        queue = deque([node])
        visited.add(node)
        size = 0

        while queue:
            current = queue.popleft()
            size += 1

            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        sizes.append(size)

    return sizes


def main():
    nodes_path = OUTPUT_DIR / "nodes.csv"
    edges_path = OUTPUT_DIR / "edges.csv"

    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)

    node_ids = [n["node_id"] for n in nodes]
    node_set = set(node_ids)

    duplicate_node_count = len(node_ids) - len(node_set)

    dangling_edges = [
        e for e in edges
        if e["source_id"] not in node_set or e["target_id"] not in node_set
    ]

    node_type_counts = Counter(n["node_type"] for n in nodes)
    edge_type_counts = Counter(e["edge_type"] for e in edges)

    component_sizes = weakly_connected_components(node_set, edges)
    largest_component = max(component_sizes) if component_sizes else 0
    isolated_nodes = sum(1 for size in component_sizes if size == 1)

    report = {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "duplicate_node_ids": duplicate_node_count,
        "dangling_edges": len(dangling_edges),
        "node_types": dict(node_type_counts),
        "edge_types": dict(edge_type_counts),
        "weakly_connected_components": len(component_sizes),
        "largest_component_size": largest_component,
        "isolated_nodes": isolated_nodes,
        "validation_passed": duplicate_node_count == 0,
        "note": "Dangling reference edges may occur if referenced elements are outside the selected exported files."
    }

    with open(OUTPUT_DIR / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(OUTPUT_DIR / "extraction_report.md", "w", encoding="utf-8") as f:
        f.write("# SysML Graph Extraction Report\n\n")
        f.write(f"- Nodes: {len(nodes)}\n")
        f.write(f"- Edges: {len(edges)}\n")
        f.write(f"- Duplicate node IDs: {duplicate_node_count}\n")
        f.write(f"- Dangling edges: {len(dangling_edges)}\n")
        f.write(f"- Weakly connected components: {len(component_sizes)}\n")
        f.write(f"- Largest component size: {largest_component}\n")
        f.write(f"- Isolated nodes: {isolated_nodes}\n")
        f.write(f"- Validation passed: {report['validation_passed']}\n\n")

        f.write("## Node Types\n")
        for key, value in node_type_counts.most_common():
            f.write(f"- {key}: {value}\n")

        f.write("\n## Edge Types\n")
        for key, value in edge_type_counts.most_common():
            f.write(f"- {key}: {value}\n")

        if dangling_edges:
            f.write("\n## Note on Dangling Edges\n")
            f.write(
                "Some dangling edges may refer to model elements outside the selected exported files. "
                "This is expected when only a subset of the SysML library or model repository is extracted.\n"
            )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
