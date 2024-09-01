import csv
import os
import sys
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem
from langchain_ollama import ChatOllama
import time

# Set a large field size limit, but not sys.maxsize
def set_max_csv_field_size():
    max_int = sys.maxsize
    decrement = True
    while decrement:
        try:
            csv.field_size_limit(max_int)
            decrement = False
        except OverflowError:
            max_int = int(max_int/10)
    return max_int

# Set the maximum field size
max_field_size = set_max_csv_field_size()
print(f"Set maximum CSV field size to: {max_field_size}")

# Define ONLY the fields we need
fields_to_extract = ['PDB_ID', 'Protein_Name', 'PDB_Content']

# Load MolT5 model on GPU 0
device_molt5 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
molt5_tokenizer = AutoTokenizer.from_pretrained("laituan245/t5-v1_1-small-smiles2caption-ft-from-pretrained-c4")
molt5_model = AutoModelForSeq2SeqLM.from_pretrained("laituan245/t5-v1_1-small-smiles2caption-ft-from-pretrained-c4").to(device_molt5)

# Load ProtT5 model on GPU 1
device_prott5 = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
prott5_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", use_fast=False)
prott5_model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device_prott5)

# Initialize the ChatOllama model
chat_model = ChatOllama(
    model="mathstral:7b-v0.1-q6_K",
    temperature=0.3,
    max_tokens=2048,
)

def generate_caption(molt5_model, molt5_tokenizer, smiles_sequence, device_molt5):
    print("Starting generate_caption")
    start_time = time.time()
    try:
        input_ids = molt5_tokenizer(smiles_sequence, return_tensors="pt").input_ids.to(device_molt5)
        outputs = molt5_model.generate(input_ids, num_beams=5, max_length=1024)
        molt5_description = molt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated caption in {time.time() - start_time:.2f} seconds")
        return molt5_description
    except Exception as e:
        print(f"Error in generate_caption: {str(e)}")
        return None

import time
from rdkit import Chem
from rdkit.Chem import AllChem

def protein_to_smiles(protein_sequence):
    print(f"Starting protein_to_smiles for sequence of length {len(protein_sequence)}")
    
    try:
        # Create a simple SMILES-like representation of the protein sequence
        smiles_sequence = '[' + '].peptide.'.join(protein_sequence) + ']'
        print(f"Generated SMILES-like representation: {smiles_sequence[:50]}... (truncated)")
        return smiles_sequence
    except Exception as e:
        print(f"Error in protein_to_smiles: {str(e)}")
        return None
    
    
def generate_description(model, tokenizer, sequence, device, max_length=1024):
    print("Starting generate_description")
    start_time = time.time()
    max_input_length = 512
    inputs = tokenizer(sequence, return_tensors="pt", max_length=max_input_length, truncation=True).to(device)
    
    try:
        outputs = model.generate(**inputs, max_length=max_length)
        description = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated description in {time.time() - start_time:.2f} seconds")
        return description
    except Exception as e:
        print(f"Error in generate_description: {str(e)}")
        return None

def process_protein_entry(extracted_row):
    print(f"Processing entry for PDB ID: {extracted_row['PDB_ID']}")
    pdb_id = extracted_row['PDB_ID']
    resolution = extracted_row.get('Resolution', 'N/A')
    pdb_content = extracted_row.get('PDB_Content', '')
    
    protein_sequence = extract_sequence_from_pdb(pdb_content)
    
    if not protein_sequence:
        print(f"Warning: Could not extract protein sequence for PDB ID {pdb_id}")
        return "", ""
    
    smiles_sequence = protein_to_smiles(protein_sequence)
    
    if not smiles_sequence:
        print(f"Warning: Could not generate SMILES for PDB ID {pdb_id}")
        return "", ""
    
    molt5_description = generate_caption(molt5_model, molt5_tokenizer, smiles_sequence, device_molt5)
    prott5_description = generate_description(prott5_model, prott5_tokenizer, protein_sequence, device_prott5)
    
    print("molt5_description: ", molt5_description)
    print("prott5_description: ", prott5_description)
    
    combined_description = f"""
    PDB ID: {pdb_id}
    Resolution: {resolution} Angstroms
    Protein Sequence: {protein_sequence[:50]}... (truncated)
    SMILES: {smiles_sequence[:50]}... (truncated)
    MolT5 (SMILES): {molt5_description}
    ProtT5 (Protein): {prott5_description}
    """
    
    prompt = f"""Using the context provided, write a detailed description of the protein with PDB ID {pdb_id}. Below is an example as a template for you to generate the description. Ensure it's written in natural language at a 12th-grade level
    The protein structure has a resolution of {resolution} Angstroms.
    Include information about its structure, function, and any notable features. 
    Use the following context to enhance your description:

    Context:
    {combined_description}

    Include information about:
    1. The protein's structure (e.g., alpha helices, beta sheets, domains)
    2. Its function and role in biological processes
    3. Any notable features or characteristics
    4. Its interactions with other molecules or proteins
    5. Any known mutations or variants and their effects
    6. Its relevance in research or medical applications

    <example_description>
    [Your example description here]
    </example_description>
    """
    
    try:
        final_description = chat_model.invoke(prompt).content
        return final_description, smiles_sequence
    except Exception as e:
        print(f"Error generating final description for PDB ID {pdb_id}: {str(e)}")
        return "", smiles_sequence

def extract_sequence_from_pdb(pdb_content):
    seqres_lines = [line for line in pdb_content.split('\n') if line.startswith('SEQRES')]
    
    three_letter_sequence = ' '.join(line[19:].strip() for line in seqres_lines)
    three_letter_codes = three_letter_sequence.split()
    
    # Dictionary to convert 3-letter amino acid codes to 1-letter codes
    aa_dict = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    one_letter_sequence = ''.join(aa_dict.get(code, 'X') for code in three_letter_codes)
    return one_letter_sequence

def process_csv_file(csv_file_path, output_file_path):
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file, \
         open(output_file_path, 'w', newline='', encoding='utf-8') as output_file:
        csv_reader = csv.DictReader(csv_file)
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(['PDB_ID', 'Description', 'SMILES'])
        
        processed_entries = []
        skipped_entries = []
        for i, row in enumerate(csv_reader, 1):
            try:
                extracted_row = {field: row[field] for field in fields_to_extract if field in row}
                
                pdb_id = extracted_row['PDB_ID']
                protein_name = extracted_row.get('Protein_Name', '')
                pdb_content = extracted_row.get('PDB_Content', '')
                
                description, smiles = process_protein_entry(extracted_row)
                if description and smiles:
                    processed_entries.append([pdb_id, description, smiles])
                    print(f"Processed: {pdb_id}")
                else:
                    skipped_entries.append(pdb_id)
                    print(f"Skipped: {pdb_id}")
                
                if i % 10 == 0:
                    csv_writer.writerows(processed_entries)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                    processed_entries = []
                    print(f"Saved {i} entries")
            except Exception as e:
                print(f"Error processing entry {i}: {str(e)}")
                skipped_entries.append(pdb_id)
        
        if processed_entries:
            csv_writer.writerows(processed_entries)
            output_file.flush()
            os.fsync(output_file.fileno())
            print(f"Saved final {len(processed_entries)} entries")
        
        print(f"Total skipped entries: {len(skipped_entries)}")
        print(f"Skipped PDB IDs: {', '.join(skipped_entries)}")
        
        
# Example usage
csv_file_path = r'C:\Users\wes\vectordb_data_good\split_pdb_data_with_img\pdb_dataset_with_images_chunk_4.csv'
output_file_path = r'C:\Users\wes\vectordb_data_good\split_pdb_data_with_img\pdb_dataset_with_images_chunk_4_detailed_nl_descriptions_molt5.csv'

process_csv_file(csv_file_path, output_file_path)