# GNN Scalability Benchmark

This repository contains the source code used for the RP:

**Analyzing the scalability of graph neural networks when encoding large knowledge graphs**

The implementation evaluates the scalability behaviour of selected Graph Neural Network architectures under controlled graph-size growth. The empirical experiments are conducted on Cora and PubMed-tripled using graph-size fractions of 25%, 50%, 75%, and 100%.

## Implemented Models

The repository includes implementations and experiment scripts for:

- GraphSAGE
- GraphSAINT-style subgraph mini-batch training
- Cluster-GCN-style cluster mini-batch training
- Graph Attention Network (GAT)

## Datasets

The experiments use:

- **Cora**: citation graph benchmark used as the smaller baseline dataset.
- **PubMed-tripled**: PubMed citation graph with the original feature matrix concatenated three times to create a 1,500-dimensional feature representation.

The datasets are loaded using PyTorch Geometric's Planetoid dataset interface.

## Evaluation Metrics

The experiments collect the following metrics:

- Test accuracy
- Forward computational memory
- Average epoch time
- Memory–accuracy trade-off
- Multi-seed accuracy standard deviation
- Empirical bias and variance

Some scripts also compute backward memory and attention-related memory as additional diagnostic values, but the final thesis discussion focuses mainly on forward computational memory.

## Repository Structure

```text
src/
  data.py                      # Cora loading and graph-fraction construction
  data_pubmed_tripled.py       # PubMed-tripled loading and subgraph construction
  models.py                    # model definitions
  gat_model.py                 # GAT model definition
  trainers.py                  # Training and computational memory functions
  gat_trainer.py               # GAT training and memory functions
  profiling.py                 # RSS/process memory profiling utilities

scripts/
  run_cora_benchmark.py                  # Cora benchmark for GraphSAGE, GraphSAINT, Cluster-GCN
  run_cora_gat_benchmark.py              # Cora benchmark for GAT
  run_pubmed_tripled_benchmark.py        # PubMed-tripled benchmark for GraphSAGE, GraphSAINT, Cluster-GCN
  run_pubmed_tripled_gat_benchmark.py    # PubMed-tripled benchmark for GAT
  run_cora_multiseed_summary.py          # Cora multi-seed stability and bias–variance analysis
  run_pubmed_tripled_multiseed_summary.py # PubMed-tripled multi-seed stability and bias–variance analysis
  plot_results.py                        # Generates the final thesis plots from aggregated result values


  NOTE: GraphSAGE, GraphSAINT, and Cluster-GCN are run using run_cora_benchmark.py and run_pubmed_tripled_benchmark.py. GAT is run using separate GAT benchmark scripts.

  NOTE: The plotting script contains the final aggregated result values used for the thesis figures.



  ## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt

NOTE: If torch-geometric does not install correctly, install PyTorch and PyTorch Geometric according to the official instructions for your system.



Running the Main Experiments




Cora: GraphSAGE, GraphSAINT, Cluster-GCN

python scripts/run_cora_benchmark.py --epochs 20 --memory-mode computational


Cora: GAT

python scripts/run_cora_gat_benchmark.py --epochs 20 --memory-mode computational





PubMed-tripled: GraphSAGE, GraphSAINT, Cluster-GCN

python scripts/run_pubmed_tripled_benchmark.py --epochs 20 --memory-mode computational

PubMed-tripled: GAT

python scripts/run_pubmed_tripled_gat_benchmark.py --epochs 20 --memory-mode computational






Multi-seed Stability and Bias–Variance Analysis

Cora

python scripts/run_cora_multiseed_summary.py --epochs 20 --models graphsage graphsaint clustergcn gat



PubMed-tripled

python scripts/run_pubmed_tripled_multiseed_summary.py --epochs 20 --models graphsage graphsaint clustergcn gat