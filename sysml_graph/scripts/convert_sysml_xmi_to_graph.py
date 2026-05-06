import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

INPUT_DIR = Path("sysml_graph/input/exported_xmi")
OUTPUT_DIR = Path("sysml_graph/output")

NODE_FIELDS = ["node_id", "name", "node_type", "qualified_name", "source_file", "raw_type"]
EDGE_FIELDS = ["source_id", "target_id", "edge_type", "source_file"]


def clean_tag(tag):
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    return tag


def get_xmi_id(elem):
    for key, value in elem.attrib.items():
        if key.endswith("id"):
            return value
    return None


def get_xmi_type(elem):
    for key, value in elem.attrib.items():
        if key.endswith("type"):
            return value.split(":")[-1]
    return clean_tag(elem.tag)


def get_name(elem):
    return elem.attrib.get("name") or elem.attrib.get("declaredName") or ""


def add_node(nodes, elem, source_file):
    node_id = get_xmi_id(elem)
    if not node_id:
        return None

    node_type = get_xmi_type(elem)

    nodes[node_id] = {
        "node_id": node_id,
        "name": get_name(elem),
        "node_type": node_type,
        "qualified_name": elem.attrib.get("qualifiedName", ""),
        "source_file": source_file,
        "raw_type": clean_tag(elem.tag),
    }

    return node_id


def add_edge(edges, source_id, target_id, edge_type, source_file):
    if not source_id or not target_id:
        return
    if source_id == target_id:
        return

    edges.append({
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": edge_type,
        "source_file": source_file,
    })


def infer_edge_type(attr_name):
    name = attr_name.lower()

    if "type" in name:
        return "typed_by"
    if "special" in name:
        return "specializes"
    if "owner" in name:
        return "owned_by"
    if "import" in name:
        return "imports"
    if "target" in name or "source" in name or "related" in name:
        return "references"

    return "references"


def maybe_reference_tokens(value):
    tokens = []

    for token in str(value).replace("\n", " ").split():
        token = token.strip()

        if not token:
            continue

        if token.startswith("#"):
            token = token[1:]

        # XMI references are often IDs, fragments, or long generated identifiers.
        if token.startswith("_") or len(token) > 5:
            tokens.append(token)

    return tokens


def walk_xml(elem, nodes, edges, source_file, parent_id=None):
    current_id = add_node(nodes, elem, source_file)

    if parent_id and current_id:
        add_edge(edges, parent_id, current_id, "contains", source_file)

    active_parent = current_id or parent_id

    # Attribute-based references.
    for attr_name, attr_value in elem.attrib.items():
        lower_attr = attr_name.lower()

        if lower_attr.endswith("id") or lower_attr.endswith("type"):
            continue

        if any(keyword in lower_attr for keyword in ["owner", "type", "target", "source", "related", "special", "import"]):
            edge_type = infer_edge_type(lower_attr)

            for token in maybe_reference_tokens(attr_value):
                add_edge(edges, current_id, token, edge_type, source_file)

    for child in list(elem):
        walk_xml(child, nodes, edges, source_file, active_parent)


def write_outputs(nodes, edges):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NODE_FIELDS)
        writer.writeheader()
        writer.writerows(nodes.values())

    with open(OUTPUT_DIR / "edges.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EDGE_FIELDS)
        writer.writeheader()
        writer.writerows(edges)

    stats = {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "node_types": dict(Counter(n["node_type"] for n in nodes.values())),
        "edge_types": dict(Counter(e["edge_type"] for e in edges)),
    }

    with open(OUTPUT_DIR / "graph_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


def main():
    nodes = {}
    edges = []

    xmi_files = list(INPUT_DIR.glob("*.sysmlx")) + list(INPUT_DIR.glob("*.xmi"))

    if not xmi_files:
        raise FileNotFoundError(f"No .sysmlx or .xmi files found in {INPUT_DIR}")

    for path in xmi_files:
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            walk_xml(root, nodes, edges, path.name)
        except ET.ParseError as e:
            print(f"Skipping {path.name}: XML parse error: {e}")

    write_outputs(nodes, edges)


if __name__ == "__main__":
    main()
