# 📐 Data Shape Changes under Neural Networks

Short demonstration of how the shape of a data cloud simplifies as it progresses through the layers of a neural network. A brief section on persistent homology is included. Inspired by the paper "Topology of deep neural networks" available at [https://arxiv.org/abs/2004.06093](https://arxiv.org/abs/2004.06093).

## 🔗 Live Report
The full interactive data storytelling report—complete with narrative analysis and dynamic visualizations—is hosted live on my portfolio:
👉 **[View the Interactive Portfolio Report](https://andratx-bellmunt.github.io/portfolio/projects/nn_shape_changes.html)**


## 🛠️ Project Architecture

This repository is engineered using modular, industry-standard Python data pipelines. Heavy plotting logic and helper functions are isolated from the core narrative to keep code execution clean and readable.

```text
.
├── data/                  # Source CSV datasets
├── docs/                  # Clean production artifacts (rendered HTML)
├── src/
│   ├── notebooks/         # Narrative-driven analysis and modeling notebooks
│   └── utils/             # Modular Python utility scripts (e.g., plots.py)
├── templates/             # HTML injection fragments (analytics & tracking tokens)
├── _quarto.yml            # Automated Quarto build configurations
├── pyproject.toml         # Fast, reproducible dependency configurations (via uv)
└── uv.lock                # Deterministic lockfile for exact environment state
```


## 🚀 Quick Start & Reproducibility

This project utilizes `uv` for Python dependency management. Follow these steps to clone the repo and run the environment locally:

### Clone the Repository

```bash
git clone https://github.com/andratx-bellmunt/shape-changes-under-nn.git
cd shape-changes-under-nn
```

### Set Up the Environment

Ensure you have `uv` installed, then synchronize the environment (this will automatically create a virtual environment and install all pinned versions):
```bash
uv sync
```

### Explore the Code

To run the notebook with your active `uv` environment:

```bash
uv run jupyter notebook src/notebooks/nba.ipynb
```

## 📊 Methodology & Key Findings

We train a Neural Network to perform a Binary Classification task. A thorough tracking of how the data shape changes as it is transformed by the network layers brings on many learnings. In particular:

* **Encoding Shape:** complexity of point cloud shapes can be encoded through Betti numbers
* **Topology:** Betti numbers can be computed with persistent homology
* **Neural Networks Action:** Each new layer on a neural network simplifies Betti numbers (i.e. the data shape)
* **Classification Power:** Once shape is simplified, it is easier to perform some tasks, such as biary classification


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

