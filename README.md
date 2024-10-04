# Genen

Genen is a project that utilizes Graph Neural Networks (GNNs) to analyze PPI, GO, and GO+PPI networks to predict various gene attributes such as solubility. The project compares the performance of different networks and models in predicting these attributes.

## Project Structure

- `PPI.ipynb`: Uses GCN and GAT to predict gene solubility in the PPI network.
- `GO.ipynb`: Uses GCN and GAT to predict gene solubility in the GO network.
- `GO+PPI.ipynb`: Uses GCN and GAT to predict gene solubility in the GO+PPI network and includes hidden node feature testing.
- `regression.ipynb`: Uses GCN for regression to predict gene conservation.
- `PPI_hidding.ipynb`: Tests hidden node features in the PPI network.
- `other.ipynb`: Uses GCN to predict other gene attributes, including:
  - Dosage Sensitivity
  - BivalentVs Lys4 Methylated
  - BivalentVs Non Methylated
  - Tf range
  - Tf target type
  - Solubility
  - Subcellular localization

- `GNN/`: Directory containing all the datasets used in this project.
- `GCN.pth` and `GAT.pth`: Model parameters trained on the GO+PPI network.

## Results

The prediction accuracy on the GO+PPI network for solubility is generally higher compared to the PPI and GO networks. The trained GCN and GAT network parameters are saved in the `GCN.pth` and `GAT.pth` files.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Genen.git
   cd Genen

## Data
All data files are stored in the [google drive](https://drive.google.com/drive/folders/1MEtsO0q5-nw6gvtPT2rZJwocCdcSJYxl?usp=sharing). Ensure that the data is correctly placed before running the notebooks.
