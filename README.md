# Cora GNN Benchmark

A simple benchmark repo for node classification on the Cora dataset using PyTorch Geometric.

Included models:
- GraphSAGE
- GraphSAINT-style mini-batch GraphSAGE
- ClusterGCN-style mini-batch GraphSAGE

Included profiling:
- process RAM (RSS) for this Python script
- snapshots before/after forward, backward, optimizer step
- optional forward-pass step snapshots inside the model

## Install

```bash
pip install -r requirements.txt
```

## Run one model

```bash
python scripts/run_cora_model.py --model graphsage --fraction 1.0 --epochs 200
python scripts/run_cora_model.py --model graphsaint --fraction 0.5 --epochs 100
python scripts/run_cora_model.py --model clustergcn --fraction 0.5 --epochs 100
```

## Run benchmark sweep

```bash
python scripts/run_cora_benchmark.py --epochs 100
```
