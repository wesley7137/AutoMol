import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GINConv, GraphSAGE, TransformerConv, EdgeConv, global_mean_pool
from torch.optim import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

EXPECTED_COLUMNS = ['accession', 'name', 'sequence', 'length', 'function',
                    'protein_name', 'gene_names', 'organism',
                    'subcellular_location', 'go_terms',
                    'keywords', 'disease', 'pathway']

class AdvancedGraphNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AdvancedGraphNeuralNetwork, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.gin1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
        )
        self.sage = GraphSAGE(hidden_dim, hidden_dim, num_layers=2)
        self.transformer = TransformerConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.edge_conv = EdgeConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(2 * hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
        )
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gat2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gin1(x, edge_index))
        x = F.relu(self.sage(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.transformer(x, edge_index))
        x = F.relu(self.edge_conv(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x




def load_and_preprocess_protein_data(file_path, batch_size=20, max_rows=None):
    logging.info(f"Starting to read data from {file_path}")
    graph_data = []
    total_rows = 0
    error_count = 0
    try:
        for chunk in tqdm(pd.read_csv(file_path, chunksize=batch_size, low_memory=False)):
            for col in EXPECTED_COLUMNS:
                if col not in chunk.columns:
                    chunk[col] = ""
            for _, row in chunk.iterrows():
                try:
                    sequence = str(row.get('sequence', ''))
                    function = row.get('function', '')
                    if pd.isna(function) or not isinstance(function, (int, float)):
                        function = len(sequence)
                    else:
                        function = float(function)
                    x, edge_index = protein_to_graph_features(sequence)
                    if x.numel() == 0 or edge_index.numel() == 0:
                        raise ValueError("Empty feature tensor or edge index")
                    label = torch.tensor([function], dtype=torch.float)
                    graph_data.append(Data(x=x, edge_index=edge_index, y=label))
                    total_rows += 1
                    if max_rows and total_rows >= max_rows:
                        logging.info(f"Reached maximum number of rows: {max_rows}")
                        break
                except Exception as e:
                    error_count += 1
                    logging.error(f"Error processing row {total_rows + 1}: {str(e)}")
                    logging.error(f"Problematic row data: {row}")
                    logging.error(f"Sequence: {sequence}")
                    logging.error(f"Function: {function}")
            if max_rows and total_rows >= max_rows:
                break
        logging.info(f"Processed {total_rows} rows. Errors: {error_count}")
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
    if not graph_data:
        raise ValueError("No valid data was loaded")
    return train_test_split(graph_data, test_size=0.2, random_state=42)

def protein_to_graph_features(sequence):
    amino_acids = list(sequence)
    num_nodes = len(amino_acids)
    node_features = [aa_to_features(aa) for aa in amino_acids]
    edges = [[i, i+1] for i in range(num_nodes-1)]
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index

def aa_to_features(amino_acid):
    aa_properties = {
        'A': [89.09, 6.00, 1.8, 0.0, 88.6, 0.0, 0.0, 1.0, 0.62],
        'R': [174.20, 10.76, -4.5, 1.0, 173.4, 1.0, 0.0, 0.0, 0.64],
        'N': [132.12, 5.41, -3.5, 0.0, 114.1, 0.5, 0.0, 0.0, 0.49],
        'D': [133.10, 2.77, -3.5, -1.0, 111.1, 0.5, 0.0, 0.0, 0.48],
        'C': [121.15, 5.07, 2.5, 0.0, 108.5, 0.0, 0.0, 0.0, 0.62],
        'E': [147.13, 3.22, -3.5, -1.0, 138.4, 0.5, 0.0, 0.0, 0.54],
        'Q': [146.15, 5.65, -3.5, 0.0, 143.8, 0.5, 0.0, 0.0, 0.56],
        'G': [75.07, 5.97, -0.4, 0.0, 60.1, 0.0, 0.0, 1.0, 0.48],
        'H': [155.16, 7.59, -3.2, 0.0, 153.2, 0.5, 0.0, 0.0, 0.61],
        'I': [131.17, 6.02, 4.5, 0.0, 166.7, 0.0, 0.0, 0.0, 0.73],
        'L': [131.17, 5.98, 3.8, 0.0, 166.7, 0.0, 0.0, 0.0, 0.69],
        'K': [146.19, 9.74, -3.9, 1.0, 168.6, 1.0, 0.0, 0.0, 0.52],
        'M': [149.21, 5.74, 1.9, 0.0, 162.9, 0.0, 0.0, 0.0, 0.70],
        'F': [165.19, 5.48, 2.8, 0.0, 189.9, 0.0, 1.0, 0.0, 0.80],
        'P': [115.13, 6.30, -1.6, 0.0, 112.7, 0.0, 0.0, 0.0, 0.36],
        'S': [105.09, 5.68, -0.8, 0.0, 89.0, 0.5, 0.0, 0.0, 0.41],
        'T': [119.12, 5.60, -0.7, 0.0, 116.1, 0.5, 0.0, 0.0, 0.48],
        'W': [204.23, 5.89, -0.9, 0.0, 227.8, 0.0, 1.0, 0.0, 0.85],
        'Y': [181.19, 5.66, -1.3, 0.0, 193.6, 0.5, 1.0, 0.0, 0.76],
        'V': [117.15, 5.96, 4.2, 0.0, 140.0, 0.0, 0.0, 0.0, 0.64],
    }
    return aa_properties.get(amino_acid, [0.0] * 9)

def train_gnn(train_data, val_data, input_dim, hidden_dim, output_dim, epochs=10, patience=10):
    model = AdvancedGraphNeuralNetwork(input_dim, hidden_dim, output_dim)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            logging.info(f"Training loss: {loss.item()}")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch)
                val_loss += criterion(out.squeeze(), batch.y).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_gnn_model.pth')
        else:
            counter += 1
            if counter >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
if __name__ == "__main__":
    try:
        print("Starting protein data processing...")
        logging.info("Loading and preprocessing protein data...")
        train_data, val_data = load_and_preprocess_protein_data('protein.csv', max_rows=5000)
        print(f"Data loaded. Train set size: {len(train_data)}, Validation set size: {len(val_data)}")
        logging.info(f"Data loaded. Train set size: {len(train_data)}, Validation set size: {len(val_data)}")

        if len(train_data) == 0 or len(val_data) == 0:
            raise ValueError("No valid data was loaded for training or validation")

        print("Displaying sample data:")
        for i in range(min(5, len(train_data))):
            print(f"Sample {i}: x shape: {train_data[i].x.shape}, edge_index shape: {train_data[i].edge_index.shape}, y: {train_data[i].y}")
            logging.info(f"Sample {i}: x shape: {train_data[i].x.shape}, edge_index shape: {train_data[i].edge_index.shape}, y: {train_data[i].y}")

        print("Starting GNN training...")
        logging.info("Starting GNN training...")
        train_gnn(train_data, val_data, input_dim=9, hidden_dim=256, output_dim=1)
        print("GNN training completed.")
        logging.info("GNN training completed.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"An error occurred: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        logging.error(traceback.format_exc())
