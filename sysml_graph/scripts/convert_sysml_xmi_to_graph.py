import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

INPUT_DIR = Path("sysml_graph/input/exported_xmi")
OUTPUT_DIR = Path("sysml_graph/output")

NODE_FIELDS = ["node_id", "name", "node_type", "qualified_name", "source_file", "raw_type"]
EDGE_FIELDS = ["source_id", "target_id", "edge_type", "source_file"]


# Relationship nodes in SysML/KerML XMI are often represented as model elements.
# This map adds semantic edge labels based on the type of the relationship element.
NODE_TYPE_TO_SEMANTIC_EDGE = {
    "OwningMembership": "owns",
    "Membership": "has_member",
    "MembershipImport": "imports",

    "FeatureMembership": "has_feature",
    "EndFeatureMembership": "has_end",
    "ParameterMembership": "has_parameter",
    "ReturnParameterMembership": "has_return_parameter",
    "SubjectMembership": "has_subject",
    "ObjectiveMembership": "has_objective",
    "RequirementConstraintMembership": "constrains",

    "FeatureTyping": "typed_by",
    "Subclassification": "specializes",
    "Redefinition": "redefines",
    "Subsetting": "subsets",
    "ReferenceSubsetting": "subsets",

    "FeatureValue": "has_value",
    "FeatureChaining": "chains",
    "FeatureChainExpression": "chains",
    "FeatureReferenceExpression": "references",

    "SatisfyRequirementUsage": "satisfies",
    "ConnectionUsage": "connects",
    "BindingConnectorAsUsage": "connects",
    "FlowUsage": "flows_to",
    "SuccessionAsUsage": "precedes",

    "ActionUsage": "has_action",
    "StateUsage": "has_state",
    "TransitionUsage": "has_transition",
    "ConstraintUsage": "has_constraint",
    "RequirementUsage": "has_requirement",
    "UseCaseUsage": "has_use_case",
    "ViewUsage": "has_view",
    "RenderingUsage": "has_rendering",
    "ViewpointUsage": "has_viewpoint",
    "AttributeUsage": "has_attribute",
    "OccurrenceUsage": "has_occurrence",
    "EventOccurrenceUsage": "has_event",
    "AcceptActionUsage": "has_accept_action",
    "AssignmentActionUsage": "has_assignment_action",
    "WhileLoopActionUsage": "has_loop",
    "PerformActionUsage": "performs_action",
    "InvocationExpression": "invokes",
    "IndexExpression": "indexes",
    "OperatorExpression": "has_operator",
    "ResultExpressionMembership": "has_result",
}


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


def add_edge(edges, edge_set, source_id, target_id, edge_type, source_file):
    if not source_id or not target_id:
        return
    if source_id == target_id:
        return

    key = (source_id, target_id, edge_type, source_file)
    if key in edge_set:
        return

    edge_set.add(key)
    edges.append({
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": edge_type,
        "source_file": source_file,
    })


def infer_edge_type_from_attribute(attr_name):
    name = attr_name.lower()

    if "type" in name:
        return "typed_by"
    if "special" in name:
        return "specializes"
    if "redef" in name:
        return "redefines"
    if "subset" in name:
        return "subsets"
    if "owner" in name:
        return "owned_by"
    if "import" in name:
        return "imports"
    if "source" in name:
        return "source_ref"
    if "target" in name:
        return "target_ref"
    if "related" in name:
        return "references"
    if "member" in name:
        return "has_member"
    if "feature" in name:
        return "has_feature"
    if "parameter" in name:
        return "has_parameter"
    if "subject" in name:
        return "has_subject"
    if "objective" in name:
        return "has_objective"

    return "references"


def maybe_reference_tokens(value):
    tokens = []

    for token in str(value).replace("\n", " ").split():
        token = token.strip().strip('"').strip("'")

        if not token:
            continue

        # XMI references sometimes appear as URI fragments.
        if "#" in token:
            token = token.split("#")[-1]

        if token.startswith("#"):
            token = token[1:]

        # Ignore obvious non-reference values.
        if token in {"true", "false", "public", "private", "protected"}:
            continue

        # XMI IDs are often generated identifiers or long references.
        if token.startswith("_") or len(token) > 5:
            tokens.append(token)

    return tokens


def is_reference_attribute(attr_name):
    name = attr_name.lower()

    # Skip actual identity/type/name fields.
    if name.endswith("id") or name.endswith("type"):
        return False

    skip_keywords = [
        "name",
        "declaredname",
        "qualifiedname",
        "visibility",
        "isabstract",
        "isderived",
        "isordered",
        "isunique",
    ]
    if any(keyword in name for keyword in skip_keywords):
        return False

    reference_keywords = [
        "owner",
        "type",
        "target",
        "source",
        "related",
        "special",
        "import",
        "member",
        "feature",
        "parameter",
        "subject",
        "objective",
        "subset",
        "redef",
    ]

    return any(keyword in name for keyword in reference_keywords)


def walk_xml(elem, nodes, edges, edge_set, source_file, parent_id=None):
    current_id = add_node(nodes, elem, source_file)
    current_type = get_xmi_type(elem)

    if parent_id and current_id:
        # Generic structural edge.
        add_edge(edges, edge_set, parent_id, current_id, "contains", source_file)

        # Additional semantic edge based on the relationship/model element type.
        semantic_edge = NODE_TYPE_TO_SEMANTIC_EDGE.get(current_type)
        if semantic_edge:
            add_edge(edges, edge_set, parent_id, current_id, semantic_edge, source_file)

    active_parent = current_id or parent_id

    # Attribute-based references. These capture typed_by, specializes, imports, etc.,
    # whenever the structured XMI stores them as references.
    if current_id:
        for attr_name, attr_value in elem.attrib.items():
            if not is_reference_attribute(attr_name):
                continue

            edge_type = infer_edge_type_from_attribute(attr_name)

            for token in maybe_reference_tokens(attr_value):
                add_edge(edges, edge_set, current_id, token, edge_type, source_file)

    for child in list(elem):
        walk_xml(child, nodes, edges, edge_set, source_file, active_parent)


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
    edge_set = set()

    xmi_files = list(INPUT_DIR.glob("*.sysmlx")) + list(INPUT_DIR.glob("*.xmi"))

    if not xmi_files:
        raise FileNotFoundError(f"No .sysmlx or .xmi files found in {INPUT_DIR}")

    for path in xmi_files:
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            walk_xml(root, nodes, edges, edge_set, path.name)
        except ET.ParseError as e:
            print(f"Skipping {path.name}: XML parse error: {e}")

    write_outputs(nodes, edges)


if __name__ == "__main__":
    main()
