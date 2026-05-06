# SysML v2 Graph Extraction Pipeline

This module implements a structured SysML v2 graph extraction pipeline.

The goal is to convert structured SysML v2 model exports into graph data suitable for future graph-learning experiments.

## Input

The current implementation uses `.sysmlx` / XMI-style structured model files from public SysML v2 model resources.

Input files should be placed in:

`sysml_graph/input/exported_xmi/`

## Output

The pipeline produces:

- `nodes.csv`
- `edges.csv`
- `graph_stats.json`
- `validation_report.json`
- `extraction_report.md`
- `sysml_graph.pt`

## Graph Schema

SysML model elements are represented as graph nodes.

Examples:

- packages
- definitions
- usages
- requirements
- ports
- attributes
- constraints
- actions
- flows
- connections
- use cases
- views

SysML relationships are represented as typed graph edges.

Current extracted edge types:

- `contains`
- `typed_by`

Semantic SysML constructs such as requirement satisfaction, connections, actions, and flows are currently represented primarily as typed nodes. Extracting them into richer semantic edge types is left as future work.

## PyTorch Geometric Export

The extracted graph is converted into a PyTorch Geometric `Data` object:

- `x`: one-hot node-type feature matrix
- `edge_index`: graph connectivity
- `edge_type`: encoded edge type labels

The generated file is:

`sysml_graph/output/sysml_graph.pt`

## Scope

This pipeline does not implement a complete SysML textual parser. Instead, it uses structured SysML/XMI exports and converts model elements and relationships into graph format.

Full GNN benchmarking on the extracted SysML graph is considered future work.
