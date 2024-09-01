import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.optim import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any
from tqdm import tqdm
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import KMeans
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNeuralNetwork, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class OmicsRAGPipeline:
    def __init__(self, vector_store_path: str):
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store_path = self._sanitize_path(vector_store_path)
        self.db = None
        self.qa_chain = None
        self.gnn_model = None
        print("OmicsRAGPipeline initialized.")
        print(f"Vector store path: {self.vector_store_path}")

    def _sanitize_path(self, path: str) -> str:
        # Replace backslashes with forward slashes
        path = path.replace("\\", "/")
        # Remove drive letter if present
        path = path.split(":")[-1]
        # Remove leading slash if present
        path = path.lstrip("/")
        return path

    def load_vector_store(self):
        print("Loading vector store...")
        print(f"Vector store path: {self.vector_store_path}")
        self.db = DeepLake(dataset_path=self.vector_store_path, embedding=self.embeddings, read_only=True)
        
        
        
    def create_rag_pipeline(self):
        if self.db is None:
            raise ValueError("Vector store is not loaded. Call load_vector_store() first.")

        retriever = self.db.as_retriever(search_kwargs={"k": 5})

        custom_prompt_template = """
        You are an AI assistant specialized in omics data analysis. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. {context}
        Question: {question}

        Answer:"""

        PROMPT = PromptTemplate(
            template=custom_prompt_template, input_variables=["question", "context"]
        )

        llm = ChatOllama(
            model="mathstral:7b-v0.1-q6_K",
            temperature=0.2,
            max_tokens=512,
            top_p=0.5,
        )
        print("Creating RAG pipeline...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def initialize_gnn(self, input_dim, hidden_dim, output_dim):
        self.gnn_model = GraphNeuralNetwork(input_dim, hidden_dim, output_dim)
        print("GNN model initialized.")

    def train_gnn(self, train_data, val_data, epochs=50, patience=10):
        if self.gnn_model is None:
            raise ValueError("GNN model is not initialized. Call initialize_gnn() first.")

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

        optimizer = Adam(self.gnn_model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()  # Adjust based on your task

        best_val_loss = float('inf')
        counter = 0
        for epoch in range(epochs):
            self.gnn_model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                optimizer.zero_grad()
                out = self.gnn_model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            self.gnn_model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    out = self.gnn_model(batch)
                    val_loss += criterion(out, batch.y).item()

            print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(self.gnn_model.state_dict(), 'best_gnn_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

    def analyze_protein_sequence(self, protein_sequence):
        if self.gnn_model is None:
            raise ValueError("GNN model is not trained. Call train_gnn() first.")

        graph_data = self.protein_to_graph(protein_sequence)

        self.gnn_model.eval()
        with torch.no_grad():
            embedding = self.gnn_model(graph_data)

        return embedding.numpy()

    def protein_to_graph(self, protein_sequence):
        # Convert protein sequence to a graph
        amino_acids = list(protein_sequence)
        num_nodes = len(amino_acids)
        
        # Create node features
        node_features = []
        for aa in amino_acids:
            features = self.aa_to_features(aa)
            node_features.append(features)
        
        # Create edges (connecting each amino acid to its neighbors)
        edge_index = []
        for i in range(num_nodes):
            if i > 0:
                edge_index.append([i-1, i])
            if i < num_nodes - 1:
                edge_index.append([i, i+1])
        
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create a dummy target (you may want to replace this with actual labels if available)
        y = torch.tensor([0.0], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, y=y)


    def query(self, question: str) -> Dict[str, Any]:
        if self.qa_chain is None:
            raise ValueError("RAG pipeline is not created. Call create_rag_pipeline() first.")
        print("Question: ", question)
        result = self.qa_chain({"query": question, "context": ""})
        print("Result: ", result)
        return result

def generate_protein_sequence(prompt, max_length=512, min_length=100):
    model_name = "nferruz/ProtGPT2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        min_length=min_length,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2
    )

    generated_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
    protein_sequence = generated_sequence[len(prompt):].strip()

    return protein_sequence

def load_and_preprocess_protein_data(file_path):
    # Load protein data from the CSV file
    df = pd.read_csv(file_path)

    # Preprocess the data
    graph_data = []
    for _, row in df.iterrows():
        sequence = row['sequence']
        function = row['function']  # We're using the function as a label

        # Convert sequence to graph
        x, edge_index = protein_to_graph_features(sequence)

        # Create Data object
        # Use the function value directly if it's a number, or its length if it's a string
        label = function if isinstance(function, (int, float)) else len(str(function))
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))
        graph_data.append(data)

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(graph_data, test_size=0.2, random_state=42)

    return train_data, val_data





def protein_to_graph_features(sequence):
    # Convert protein sequence to graph features
    amino_acids = list(sequence)
    num_nodes = len(amino_acids)
    
    # Create node features
    node_features = []
    for aa in amino_acids:
        features = aa_to_features(aa)
        node_features.append(features)
    
    # Create edges (connecting each amino acid to its neighbors)
    edge_index = []
    for i in range(num_nodes):
        if i > 0:
            edge_index.append([i-1, i])
        if i < num_nodes - 1:
            edge_index.append([i, i+1])
    
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return x, edge_index

def aa_to_features(amino_acid):
    # Extended properties for each amino acid
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
    # Features: [MW, pKa, hydrophobicity, charge, volume, polarity, aromaticity, flexibility, alpha-helix propensity]
    return aa_properties.get(amino_acid, [0]*9)  # Default to [0, 0, 0, 0, 0, 0, 0, 0, 0] for unknown amino acids


def main():
    # Initialization
    vector_store_path = "/mnt/c/users/wes/vector_data_good/omics_vector_store"
    rag_pipeline = OmicsRAGPipeline(vector_store_path)
    rag_pipeline.load_vector_store()
    rag_pipeline.create_rag_pipeline()

    # Initialize GNN
    input_dim = 9  # Based on the number of features per amino acid
    hidden_dim = 64
    output_dim = 32  # Embedding dimension
    rag_pipeline.initialize_gnn(input_dim, hidden_dim, output_dim)    # Load and preprocess protein data for GNN training
    # You'll need to create this CSV file with protein sequences and labels
    train_data, val_data = load_and_preprocess_protein_data('protein_data_combined.csv')
    
    # Train GNN
    rag_pipeline.train_gnn(train_data, val_data)

    # Generate protein sequence
    prompt = "Design a protein sequence that enhances mitochondrial function and increases lifespan. The protein sequence is: "
    protein_sequence = generate_protein_sequence(prompt)
    print(f"Generated protein sequence: {protein_sequence}")

    # Analyze the generated protein sequence using GNN
    protein_embedding = rag_pipeline.analyze_protein_sequence(protein_sequence)

    # Use the embedding in your RAG pipeline query
    context = f"Protein embedding: {protein_embedding.tolist()}"
    question = f"""Your objective is to take the user's initial query, take the context and data you are given, 
    as well as the search results supplied to you, and offer any specific details about the viability, 
    implementation, data findings, or potential novel benefits/attributes that the predicted structure might have 
    according to your dataset findings. User Input: {prompt} Sequence: {protein_sequence} Context: {context}?"""
    
    result = rag_pipeline.query(question)
    print(f"Question: {question}")
    print(f"Answer: {result['result']}")
    print("Source documents:")
    for doc in result['source_documents']:
        print(f"- {doc.metadata['id']}: {doc.page_content[:100]}...")

if __name__ == "__main__":
    main()