import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from torch_geometric.data import DataLoader
from typing import Dict, Any
from torch_geometric.nn import GATConv, GINConv, GraphSAGE, TransformerConv, EdgeConv, global_mean_pool
import torch.nn.functional as F


class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNeuralNetwork, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        # Add batch normalization layers
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
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
        # Apply batch normalization after activation
        x = self.bn1(F.relu(self.gat1(x, edge_index)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.gat2(x, edge_index)))
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







class OmicsRAGPipeline:
    def __init__(self, vector_store_path: str):
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store_path = self._sanitize_path(vector_store_path)
        self.db = None
        self.qa_chain = None
        print("OmicsRAGPipeline initialized.")
        print(f"Vector store path: {self.vector_store_path}")

    def _sanitize_path(self, path: str) -> str:
        path = path.replace("\\", "/")
        path = path.split(":")[-1]
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

    def query(self, question: str) -> Dict[str, Any]:
        if self.qa_chain is None:
            raise ValueError("RAG pipeline is not created. Call create_rag_pipeline() first.")
        print("Question: ", question)
        result = self.qa_chain({"query": question})
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


def analyze_protein_sequence(protein_sequence):
    if gnn_model is None:
        raise ValueError("GNN model is not trained. Call train_gnn() first.")

    x, edge_index = protein_to_graph(protein_sequence)
    
    # Create a Data object
    from torch_geometric.data import Data
    graph_data = Data(x=x, edge_index=edge_index)

    gnn_model.eval()
    with torch.no_grad():
        embedding = gnn_model(graph_data)

    return embedding.numpy()

def protein_to_graph(sequence):
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
    return aa_properties.get(amino_acid, [0.0] * 9)  # Return a zero vector for unknown amino acids


if __name__ == "__main__":
    vector_store_path = "./omics_vector_store"
    rag_pipeline = OmicsRAGPipeline(vector_store_path)
    # Load the trained GNN model
    gnn_model = GraphNeuralNetwork(input_dim=9, hidden_dim=256, output_dim=1)
    gnn_model.load_state_dict(torch.load('./models/simple_gnn.pth'))
    gnn_model.eval()  # Set the model to evaluation mode
    print("GNN model loaded successfully.")
    
    rag_pipeline.load_vector_store()
    rag_pipeline.create_rag_pipeline()
    print("RAG pipeline created.")

    prompt = "Design a protein sequence that enhances mitochondrial function and enhances mitophagy and Cellular Quality Control. The protein sequence is: "
    protein_sequence = generate_protein_sequence(prompt)
    print(f"Generated protein sequence: {protein_sequence}")

    protein_embedding = analyze_protein_sequence(protein_sequence)


    context = f"Protein embedding: {protein_embedding.tolist()}"
    question = f"""Your objective is to take the user's initial query, take the context and data you are given, as well as the search results supplied to you, and offer any specific details about the viability, implementation, data findings, or potential novel benefits/attributes that the predicted structure might have according to your dataset findings. User Input: {prompt} Sequence: {protein_sequence} Context: {context}?"""
    
    result = rag_pipeline.query(question)
    print(f"Question: {question}")
    print(f"Answer: {result['result']}")
    print("Source documents:")
    for doc in result['source_documents']:
        print(f"- {doc.metadata['id']}: {doc.page_content[:100]}...")

    


