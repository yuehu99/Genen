# Genen

Genen is a project that utilizes Graph Neural Networks (GNNs) to analyze PPI, GO, and GO+PPI networks to predict various gene attributes such as solubility. The project compares the performance of different networks and models in predicting these attributes.

## Project Structure

- `PPI.ipynb`: Uses GCN and GAT to predict gene solubility in the PPI network.
- `GO.ipynb`: Uses GCN and GAT to predict gene solubility in the GO network.
- `GO+PPI.ipynb`: Uses GCN and GAT to predict various gene attributes, including:
  - Dosage Sensitivity
  - BivalentVs Lys4 Methylated
  - BivalentVs Non Methylated
  - Tf range
  - Tf target type
  - Solubility
  - Subcellular localization
  - Hidden node feature testing

- `regression.ipynb`: Uses GCN for regression to predict gene conservation.
- `PPI_hidding.ipynb`: Tests hidden node features in the PPI network.
- `Data/`: Directory containing all the datasets used in this project.
- `GCN.pth` and `GAT.pth`: Model parameters trained on the GO+PPI network.

## Results

The prediction accuracy on the GO+PPI network is generally higher compared to the PPI and GO networks. The trained GCN and GAT network parameters are saved in `GCN.pth` and `GAT.pth` files.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Genen.git
   cd Genen
## Data
All data files are stored in the Data directory. Ensure that the data is correctly placed before running the notebooks.
## Contact
For any questions or inquiries, please contact yuhu@salud.unm.edu.
