import os
import pandas as pd
import logging
from typing import List
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import xml.etree.ElementTree as ET
import torch
from uuid import uuid4


class DataLoader:
    @staticmethod
    def load_uniprot_data(file_path):
        if file_path.endswith('.xml'):
            return DataLoader.load_uniprot_xml(file_path)
        elif file_path.endswith('.tsv'):
            return DataLoader.load_uniprot_tsv(file_path)
        else:
            raise ValueError("Unsupported file format. Please use XML or TSV.")

    @staticmethod
    def load_uniprot_xml(file_path):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            data = []
            for entry in root.findall('{http://uniprot.org/uniprot}entry'):
                accession = entry.find('{http://uniprot.org/uniprot}accession')
                name = entry.find('{http://uniprot.org/uniprot}name')
                sequence = entry.find('{http://uniprot.org/uniprot}sequence')

                if accession is None or name is None or sequence is None:
                    continue  # Skip entries with missing required fields

                function = ""
                for comment in entry.findall('{http://uniprot.org/uniprot}comment'):
                    if comment.get('type') == 'function':
                        function_text = comment.find('{http://uniprot.org/uniprot}text')
                        if function_text is not None:
                            function = function_text.text
                            break

                data.append({
                    'Entry': accession.text,
                    'Entry Name': name.text,
                    'Sequence': sequence.text,
                    'Function': function
                })

            return pd.DataFrame(data)
        except ET.ParseError as e:
            logging.error(f"Error parsing XML file {file_path}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def load_uniprot_tsv(file_path):
        try:
            df = pd.read_csv(file_path, sep='\t', low_memory=False)
            logging.info(f"Columns in {file_path}: {df.columns.tolist()}")

            # Map the available columns to our required structure
            column_mapping = {
                'Entry': 'Accessions',
                'Entry Name': 'GeneGroupSymbol',
                'Sequence': 'Peptide',
                'Function': None  # We don't have a direct mapping for this
            }

            new_df = pd.DataFrame()
            for our_col, their_col in column_mapping.items():
                if their_col in df.columns:
                    new_df[our_col] = df[their_col]
                else:
                    new_df[our_col] = ''

            # If 'Accessions' column contains multiple entries, split them
            if 'Entry' in new_df.columns and new_df['Entry'].str.contains(';').any():
                new_df['Entry'] = new_df['Entry'].str.split(';').str[0]

            # Add an empty 'Function' column if it doesn't exist
            if 'Function' not in new_df.columns:
                new_df['Function'] = ''

            logging.info(f"Processed {len(new_df)} entries from {file_path}")
            return new_df

        except Exception as e:
            logging.error(f"Error loading TSV file {file_path}: {str(e)}")
            return pd.DataFrame()

class OmicsDataProcessor:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        self.data_loader = DataLoader()
        self.vector_store_manager = VectorStoreManager()
        self.db = None

    def process_file(self, file_path: str) -> List[Document]:
        data = self.data_loader.load_uniprot_data(file_path)
        if data.empty:
            logging.warning(f"No valid data found in {file_path}")
            return []
        
        self.print_sample_entries(data, file_path)
        
        docs = self.create_documents(data)
        return docs

    def print_sample_entries(self, data: pd.DataFrame, file_path: str, num_samples: int = 3):
        logging.info(f"\nSample entries from {os.path.basename(file_path)}:")
        logging.info(f"Total entries: {len(data)}")
        logging.info(f"Columns: {data.columns.tolist()}")
        
        sample = data.sample(min(num_samples, len(data)))
        
        for idx, row in sample.iterrows():
            logging.info(f"\nSample Entry {idx}:")
            for col in data.columns:
                value = str(row[col])
                if len(value) > 50:
                    value = value[:47] + "..."
                logging.info(f"  {col}: {value}")

    def create_documents(self, df: pd.DataFrame) -> List[Document]:
        docs = []
        for _, row in df.iterrows():
            doc = Document(
                page_content=row.get('Sequence', ''),
                metadata={
                    'Entry': row.get('Entry', ''),
                    'Entry Name': row.get('Entry Name', ''),
                    'Function': row.get('Function', '')
                }
            )
            docs.append(doc)
        return docs

    def process_directory(self):
        all_docs = []
        for filename in os.listdir(self.directory_path):
            if filename.endswith(('.xml', '.tsv')):
                file_path = os.path.join(self.directory_path, filename)
                logging.info(f"\nProcessing file: {filename}")

                try:
                    docs = self.process_file(file_path)
                    all_docs.extend(docs)
                    logging.info(f"Processed {len(docs)} entries from {filename}")
                except Exception as e:
                    logging.error(f"Error processing {filename}: {str(e)}")

        if all_docs:
            self.create_or_update_vector_store(all_docs)
            logging.info("Vector store created/updated successfully.")
        else:
            logging.warning("No documents were processed. Vector store was not updated.")

    def create_or_update_vector_store(self, docs: List[Document]):
        if self.db is None:
            self.db = self.vector_store_manager.create_vector_store(docs, "./omics_vector_store")
        else:
            self.db.add_documents(docs)

class VectorStoreManager:
    def __init__(self):
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {device}")

        # Initialize HuggingFaceEmbeddings with CUDA if available
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': device}
        )

    def create_vector_store(self, docs, dataset_path):
        logging.info(f"Creating vector store with {len(docs)} documents.")
        
        batch_size = 500  # Adjust this value based on your system's capabilities
        db = None
        steps = 0
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            if db is None:
                db = DeepLake.from_documents(
                    batch, 
                    self.embeddings, 
                    dataset_path=dataset_path, 
                    overwrite=False
                )
            else:
                db.add_documents(batch)
        if steps % 2000 == 0:
            print(f"Processed {i+len(batch)} documents")
        
        return db

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Print CUDA availability information
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    directory_path = r"./"
    processor = OmicsDataProcessor(directory_path)
    processor.process_directory()

if __name__ == "__main__":
    main()