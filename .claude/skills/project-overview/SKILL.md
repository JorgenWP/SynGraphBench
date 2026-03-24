---
name: project-overview
description: High-level introduction to SynGraphBench and quick-reference heuristics for agents. Entry point — links to detailed skills.
---

# SynGraphBench — Overview

**SynGraphBench** is a benchmarking suite designed to evaluate the trade-off between data utility and privacy in synthetic graph generation. It measures how well synthetic graph data performs compared to real data on downstream machine learning tasks, and analyzes how strict privacy guarantees (like k-anonymity) degrade this performance.

The project is an amalgamation of three distinct, previously published research repositories, orchestrated together via custom shell scripts to form a complete benchmarking pipeline.

## Detailed Skills

| Skill | What it covers |
|---|---|
| `/generative-models` | BiGG and CGT — paradigms, mechanisms, output formats, key files, Conda envs |
| `/evaluation-framework` | GADBench — anomaly detection, link prediction extension, design details |
| `/execution-flow` | Pipeline steps, shell script CLI args and defaults, env setup |
| `/project-structure` | Folder layout, dataset naming conventions |

## Quick Heuristics

1. **Which synthetic type to use?** BiGG → `--synthetic_type graph`. CGT → `--synthetic_type comp-graph`. They are not interchangeable.
2. **Changing anomaly detection hyperparameters?** `CGT/args.py` or `GADBench/benchmark.py`.
3. **Changing link prediction hyperparameters?** `GADBench/link_benchmark.py` (epochs, patience) and `GADBench/models/link_prediction/link_predictor.py` (decoder architecture).
4. **Fixing C++ compilation errors?** `scripts/env_setups/bigg.sh` and `bigg/bigg/model/tree_clib/Makefile`. Modern CUDA architectures are patched via `sed` in the setup script.
5. **Adding a new dataset?** Place original in `datasets/original/`. Update loading utilities in `CGT/task/utils/utils.py`, `GADBench/benchmark.py`, or `GADBench/link_utils.py` as appropriate.
6. **Conflicting dependencies across sub-repos?** Always activate the correct Conda env: `bigg`, `CGT`, or `GADBench` depending on which sub-repo you're working in.
