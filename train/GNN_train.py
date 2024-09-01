import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.optim import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNeuralNetwork, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)  # Global pooling
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_graph_data(df: pd.DataFrame) -> List[Data]:
    # Example: converting DataFrame rows into graph data
    graph_data = []
    for _, row in df.iterrows():
        # Assume nodes have 4 features, and edges are represented as pairs
        x = torch.tensor(row['node_features'], dtype=torch.float)
        edge_index = torch.tensor(row['edges'], dtype=torch.long)
        y = torch.tensor(row['label'], dtype=torch.float)  # Replace with the appropriate target
        data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
        graph_data.append(data)
    return graph_data

def train_gnn(model, train_loader, optimizer, criterion, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

def main():
    # Assuming df contains graph-related data for training
    df = pd.read_csv('graph_data.csv')  # Example file containing graph data
    graph_data = create_graph_data(df)
    
    # Splitting data into train and test sets
    train_data, test_data = train_test_split(graph_data, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    input_dim = 4  # Example input dimension (node feature size)
    hidden_dim = 64
    output_dim = 1  # Example output dimension (e.g., prediction of a single value)

    model = GraphNeuralNetwork(input_dim, hidden_dim, output_dim)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()  # Change this depending on your task

    train_gnn(model, train_loader, optimizer, criterion)

    # Save the trained model
    torch.save(model.state_dict(), 'gnn_model.pth')

    # Integration into the pipeline
    integrate_gnn_in_pipeline(model)

def integrate_gnn_in_pipeline(model):
    # Example function showing how the GNN can be used in your pipeline
    # Assuming 'pipeline_data' contains new data to be processed
    pipeline_data = pd.read_csv('new_data.csv')
    graph_data = create_graph_data(pipeline_data)

    # Load and process using trained GNN
    model.eval()
    for data in graph_data:
        with torch.no_grad():
            prediction = model(data)
            print(f"Predicted value: {prediction.item()}")

if __name__ == "__main__":
    main()
