import os
import pandas as pd
import logging
from typing import List, Dict, Any
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import xml.etree.ElementTree as ET
import torch


class DataLoader:
    @staticmethod
    def load_uniprot_data(file_path: str) -> pd.DataFrame:
        """Load UniProt data from XML or TSV file."""
        if file_path.endswith('.xml'):
            return DataLoader.load_uniprot_xml(file_path)
        elif file_path.endswith('.tsv'):
            return DataLoader.load_uniprot_tsv(file_path)
        else:
            raise ValueError("Unsupported file format. Please use XML or TSV.")

    @staticmethod
    def load_uniprot_xml(file_path: str) -> pd.DataFrame:
        """Parse UniProt XML file into a DataFrame with extended fields."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            data = []
            for entry in root.findall('{http://uniprot.org/uniprot}entry'):
                entry_data = {}
                try:
                    entry_data['Entry'] = entry.find('{http://uniprot.org/uniprot}accession').text
                    entry_data['Entry Name'] = entry.find('{http://uniprot.org/uniprot}name').text
                    protein_name = entry.find('{http://uniprot.org/uniprot}protein/{http://uniprot.org/uniprot}recommendedName/{http://uniprot.org/uniprot}fullName')
                    entry_data['Protein names'] = protein_name.text if protein_name is not None else ''
                    entry_data['Gene names'] = ', '.join([gene.text for gene in entry.findall('{http://uniprot.org/uniprot}gene/{http://uniprot.org/uniprot}name[@type="primary"]')])
                    organism = entry.find('{http://uniprot.org/uniprot}organism/{http://uniprot.org/uniprot}name[@type="scientific"]')
                    entry_data['Organism'] = organism.text if organism is not None else ''
                    sequence = entry.find('{http://uniprot.org/uniprot}sequence')
                    entry_data['Length'] = sequence.get('length') if sequence is not None else ''
                    entry_data['Sequence'] = sequence.text if sequence is not None else ''
                    entry_data['Function'] = next((comment.find('{http://uniprot.org/uniprot}text').text for comment in entry.findall('{http://uniprot.org/uniprot}comment') if comment.get('type') == 'function'), '')
                    entry_data['Subcellular location [CC]'] = next((comment.find('{http://uniprot.org/uniprot}text').text for comment in entry.findall('{http://uniprot.org/uniprot}comment') if comment.get('type') == 'subcellular location'), '')
                    entry_data['Disease'] = next((comment.find('{http://uniprot.org/uniprot}text').text for comment in entry.findall('{http://uniprot.org/uniprot}comment') if comment.get('type') == 'disease'), '')
                    entry_data['GO'] = ', '.join([dbReference.get('id') for dbReference in entry.findall('{http://uniprot.org/uniprot}dbReference[@type="GO"]')])
                    entry_data['Keywords'] = ', '.join([keyword.text for keyword in entry.findall('{http://uniprot.org/uniprot}keyword')])
                    entry_data['Pathway'] = next((comment.find('{http://uniprot.org/uniprot}text').text for comment in entry.findall('{http://uniprot.org/uniprot}comment') if comment.get('type') == 'pathway'), '')
                    
                    data.append(entry_data)
                except AttributeError as e:
                    logging.warning(f"Skipping entry due to missing attribute: {str(e)}")
                    continue

            return pd.DataFrame(data)
        except ET.ParseError as e:
            logging.error(f"Error parsing XML file {file_path}: {str(e)}")
            return pd.DataFrame()
        
        
    @staticmethod
    def load_uniprot_tsv(file_path: str) -> pd.DataFrame:
        """Parse UniProt TSV file into a DataFrame with extended fields."""
        try:
            df = pd.read_csv(file_path, sep='\t', low_memory=False)
            logging.info(f"Columns in {file_path}: {df.columns.tolist()}")

            column_mapping = {
                'Entry': 'Accessions',
                'Entry Name': 'GeneGroupSymbol',
                'Sequence': 'Peptide',
                'Protein names': 'GeneGroupSymbol',
                'Gene names': 'GeneGroupSymbol',
                'Organism': '',
                'Length': '',
                'Function': '',
                'Subcellular location [CC]': '',
                'Disease': '',
                'GO': '',
                'Keywords': '',
                'Pathway': ''
            }

            new_df = pd.DataFrame()
            for our_col, their_col in column_mapping.items():
                if their_col in df.columns:
                    new_df[our_col] = df[their_col]
                else:
                    new_df[our_col] = ''

            # If 'Entry' contains multiple accessions, take the first one
            if 'Entry' in new_df.columns and new_df['Entry'].str.contains(';').any():
                new_df['Entry'] = new_df['Entry'].str.split(';').str[0]

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
        """Process a single file and convert it to a list of Documents."""
        data = self.data_loader.load_uniprot_data(file_path)
        if data.empty:
            logging.warning(f"No valid data found in {file_path}")
            return []

        self.print_sample_entries(data, file_path)
        docs = self.create_documents(data)
        return docs


    def create_or_update_vector_store(self, docs: List[Document]):
        """Create or update the vector store with processed documents."""
        if self.db is None:
            self.db = self.vector_store_manager.create_vector_store(docs, "./omics_vector_store")
        else:
            self.db.add_documents(docs)
        logging.info(f"Vector store updated with {len(docs)} documents.")

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
        """Convert DataFrame rows to a list of standardized Documents for vector storage."""
        docs = []
        steps = 0
        for _, row in df.iterrows():
            standardized_entry = self.standardize_entry(row.to_dict())
            doc = Document(
                page_content=standardized_entry['sequences']['protein']['sequence'],
                metadata=standardized_entry
            )
            docs.append(doc)
            if steps == 200:
                logging.info(f"Processed {len(docs)} entries...")
        return docs


    def process_directory(self):
        """Process all valid files in the directory and update the vector store."""
        processed_files = set()
        try:
            # Load previously processed files if the file exists
            if os.path.exists('processed_files.txt'):
                with open('processed_files.txt', 'r') as f:
                    processed_files = set(f.read().splitlines())
        except Exception as e:
            logging.error(f"Error loading processed files list: {str(e)}")

        for filename in os.listdir(self.directory_path):
            if filename.endswith(('.xml', '.tsv')) and filename not in processed_files:
                file_path = os.path.join(self.directory_path, filename)
                logging.info(f"\nProcessing file: {filename}")

                try:
                    docs = self.process_file(file_path)
                    if docs:
                        self.create_or_update_vector_store(docs)
                        logging.info(f"Processed and added {len(docs)} entries from {filename}")
                        processed_files.add(filename)
                        # Save the updated list of processed files
                        with open('processed_files.txt', 'w') as f:
                            f.write('\n'.join(processed_files))
                    else:
                        logging.warning(f"No valid documents found in {filename}")
                except Exception as e:
                    logging.error(f"Error processing {filename}: {str(e)}")
                    # You might want to add a "problematic_files.txt" to keep track of files that couldn't be processed
                    with open('problematic_files.txt', 'a') as f:
                        f.write(f"{filename}: {str(e)}\n")

        if not processed_files:
            logging.warning("No files were processed. Vector store was not updated.")
        else:
            logging.info(f"Completed processing. Total files processed: {len(processed_files)}")

    def standardize_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize an entry to the required schema, mapping UniProt fields."""
        standardized = {
            'id': entry.get('Entry', ''),
            'type': 'protein',
            'full_description': entry.get('Function', ''),
            'short_description': entry.get('Protein names', ''),
            'source': 'UniProtKB',
            'confidence': 'high' if entry.get('Function') else 'low',
            'version': 1,
            'created': '',
            'last_modified': '',
            'metadata': {
                'organism': entry.get('Organism', ''),
                'taxonomy_id': '',
                'proteome_id': '',
                'gene_name': entry.get('Gene names', ''),
                'protein_name': entry.get('Protein names', ''),
                'alternative_names': []
            },
            'sequences': {
                'protein': {
                    'sequence': entry.get('Sequence', ''),
                    'length': int(entry.get('Length', 0)) if entry.get('Length') else 0
                },
                'nucleotide': {
                    'sequence': '',
                    'length': 0
                }
            },
            'molecular_properties': {},
            'annotations': [
                {'type': 'subcellular_location', 'value': entry.get('Subcellular location [CC]', '')},
                {'type': 'disease', 'value': entry.get('Disease', '')}
            ],
            'cross_references': [
                {'database': 'GO', 'id': go_id} for go_id in entry.get('GO', '').split(',') if go_id.strip()
            ],
            'genomic_mapping': {},
            'interactions': [],
            'structure': {},
            'expression': [],
            'pathways': [entry.get('Pathway', '')] if entry.get('Pathway') else [],
            'post_translational_modifications': [],
            'variants': [],
            'phenotypes': [],
            'ontologies': [
                {'type': 'go', 'terms': [term.strip() for term in entry.get('GO', '').split(',') if term.strip()]}
            ],
            'description_metadata': {
                'keywords': [kw.strip() for kw in entry.get('Keywords', '').split(',') if kw.strip()]
            }
        }
        return standardized

class VectorStoreManager:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {device}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': device}
        )

    def create_or_update_vector_store(self, docs: List[Document]):
        """Create or update the vector store with processed documents."""
        if self.db is None:
            self.db = self.vector_store_manager.create_vector_store(docs, "./omics_vector_store")
        else:
            self.db.add_documents(docs)
        logging.info(f"Vector store updated with {len(docs)} documents.")
        
        
    def create_vector_store(self, docs: List[Document], dataset_path: str):
        """Create a new vector store or update an existing one with new documents."""
        logging.info(f"Creating vector store with {len(docs)} documents.")
        
        batch_size = 500
        db = None
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            if db is None:
                db = DeepLake.from_documents(
                    batch, 
                    self.embeddings, 
                    dataset_path=dataset_path, 
                    overwrite=False
                )
            else:
                db.add_documents(batch)

        return db



def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    directory_path = r"./"
    processor = OmicsDataProcessor(directory_path)
    processor.process_directory()


if __name__ == "__main__":
    main()