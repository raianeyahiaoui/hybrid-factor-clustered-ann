# Hybrid Factor-Clustered Vector Search PoC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A proof-of-concept demonstrating a **hybrid vector indexing pipeline** that combines:

- **Factor Analysis** via PCA (dimensionality reduction + whitening)
- **IVF clustering** using FAISS for fast approximate nearest neighbor (ANN) search
- **DEA-inspired efficiency scoring** to identify the mathematically optimal recall-speed trade-off

This work is relevant to research in advanced vector databases and cluster-shaped indexing techniques.

## Key Results

On a structured manifold dataset (100,000 × 128-dimensional vectors generated with `make_blobs`):

## System Performance Evaluation

**Pipeline Execution:**
Generating structured manifold data (d=128, 100,000 vectors)...
Applying Factor Analysis via PCA (reducing to 64 dimensions + whitening)...
Building Hybrid Factor-Clustered IVF Index...
Computing exact ground truth...


**Evaluation Results Across nprobe Values:**

| nprobe | Recall@10 | Throughput (QPS) | DEA Efficiency Score |
|--------|-----------|-----------------|--------------------|
| 1      | 0.709     | 17,917.8        | 3.0173             |
| 2      | 0.922     | 6,681.5         | 3.5284             |
| 4      | 0.999     | 5,841.7         | 3.7620             |
| 8      | 1.000     | 3,476.8         | 3.5412             |
| 16     | 1.000     | 1,878.7         | 3.2739             |
| 32     | 1.000     | 767.9           | 2.8853             |
| 64     | 1.000     | 470.8           | 2.6728             |



**Mathematically Optimal Configuration** (maximum DEA efficiency score):

==================================================
RESEARCH RESULT: MATHEMATICALLY OPTIMAL CONFIGURATION
==================================================
nprobe             : 4
Recall@10          : 99.88%
Throughput         : 5,841.7 queries/second
DEA Efficiency Score: 3.7620 (maximum)
==================================================



![Efficient Frontier Plot](plot.png)

The Pareto frontier shows rapid convergence to near-perfect recall, with the DEA-inspired metric correctly identifying **nprobe=4** as the optimal operating point.

## How to Run

# Install dependencies
pip install faiss-cpu scikit-learn matplotlib numpy

# Run the experiment
python main.py

## About the Author
$Yahiaoui Raiane$
$Telecommunication Systems Engineer | AI Researcher$
$Email: yahiaoui.raiane7@gmail.com$
$LinkedIn: Yahiaoui Raiane$

$This project is licensed under the MIT License – see the LICENSE file for details.$



