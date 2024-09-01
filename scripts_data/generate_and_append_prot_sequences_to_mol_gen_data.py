import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from biotite.structure.io import pdb
from biotite.sequence import ProteinSequence
from biotite.application import dssp
import numpy as np
from typing import List, Tuple
import torch
from transformers import BertForMaskedLM, BertTokenizer
import numpy as np
import tempfile
import os

def load_protbert_model():
    print("INFO : loading protbert model")
    model_name = "Rostlab/prot_bert"
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.eval()
    print("INFO : protbert model loaded")
    return model, tokenizer


model, tokenizer = load_protbert_model()

def generate_protein_sequence(prompt, max_length=1024, min_length=100):
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
    
    
    if len(protein_sequence) < min_length:
        print(f"Warning: Generated sequence length ({len(protein_sequence)}) is less than min_length ({min_length})")
    return protein_sequence



def analyze_sequence(model, tokenizer, protein_sequence):
    #print("INFO : Analyzing sequence with protbert")
    #print("INFO : Analyze Sequence protein_sequence", protein_sequence)
    inputs = tokenizer(protein_sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    #print("INFO:   : Analyze Sequence sequence analyzed with protbert")
    #print("INFO:   : Analyze Sequence outputs.logits", outputs.logits)
    return outputs.logits

def optimize_sequence(protein_sequence, logits, threshold=0.5):
    #print(f"INFO : Entering Optimize Sequence")
    #print(f"INFO : Optimize Sequence logits {logits}")
    #print(f"INFO : Optimize Sequence threshold {threshold}")
    #print(f"INFO : Optimize Sequence protein_sequence {protein_sequence}")
    optimized_sequence = ""
    for i, aa in enumerate(protein_sequence):
        if i < logits.size(1):
            probs = torch.softmax(logits[0, i], dim=-1)
            #print(f"INFO : Optimize Sequence i {i}")
            #print(f"INFO : Optimize Sequence probs {probs}")
            if probs.max() < threshold:
                optimized_sequence += np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"))
            else:
                optimized_sequence += aa
        else:
            optimized_sequence += aa
        #print(f"INFO : Optimize Sequence optimized_sequence {optimized_sequence}")
    return optimized_sequence


import random

def predict_secondary_structure(protein_sequence):
    """
    Predict the secondary structure of a protein sequence using basic heuristics.
    
    This function provides a more robust prediction by incorporating probabilistic elements
    and considering common patterns observed in actual secondary structures.
    
    Parameters:
    protein_sequence (str): The amino acid sequence of the protein.
    
    Returns:
    str: A string representing the predicted secondary structure, where:
         'H' stands for helix, 'E' for sheet, and 'C' for coil.
    """
    #print("\nINFO:  Predicting secondary structure")
    
    # Simplified probabilities for secondary structure elements based on amino acid type
    helix_formers = {'A', 'L', 'M', 'E', 'Q', 'K', 'H'}
    sheet_formers = {'V', 'I', 'Y', 'F', 'W', 'T'}
    coil_formers = {'G', 'N', 'D', 'S', 'P', 'R'}
    #print("INFO : Predict Secondary Structure protein_sequence", protein_sequence)
    structure = []
    for aa in protein_sequence:
        if aa in helix_formers:
            structure.append('H' if random.random() > 0.2 else 'C')  # Predominantly helix
        elif aa in sheet_formers:
            structure.append('E' if random.random() > 0.2 else 'C')  # Predominantly sheet
        elif aa in coil_formers:
            structure.append('C' if random.random() > 0.2 else 'H')  # Predominantly coil
        else:
            # Randomly assign if the amino acid is not in specific sets (e.g., uncommon or missing)
            structure.append(random.choice(['H', 'E', 'C']))
    
    # Join the list into a string representing the secondary structure prediction
    secondary_structure  = ''.join(structure)
    #print(f"INFO : Predict Secondary Structure secondary_structure", secondary_structure)
    return secondary_structure 





def analyze_and_optimize_protein(protein_sequence, optimization_rounds=5):
    for _ in range(optimization_rounds):
        logits = analyze_sequence(model, tokenizer, protein_sequence)
        #print(f"INFO : Analyze and Optimize Protein Sequence length: {len(protein_sequence)}, Logits shape: {logits.shape}")
        if logits.shape[1] != len(protein_sequence):
            print("INFO : Warning: Logits dimension does not match sequence length")
        protein_sequence = optimize_sequence(protein_sequence, logits)
    #print("INFO : Analyze and Optimize Protein Sequence protein_sequence", protein_sequence)
    secondary_structure = predict_secondary_structure(protein_sequence)
    #print("INFO : Analyze and Optimize Protein Sequence secondary_structure", secondary_structure)
    return protein_sequence, secondary_structure




import json
from tqdm import tqdm


def process_json(input_path='generated_prompts.json', output_path='generated_prompts_with_sequences_and_analysis.json'):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    processed_count = 0
    for entry in tqdm(data, desc="Generating and analyzing protein sequences"):
        user_input = entry.get('user_input')
        if user_input and not entry.get('assistant'):
            max_attempts = 3  # Maximum number of attempts to generate a non-empty sequence
            for attempt in range(max_attempts):
                try:
                    protein_sequence = generate_protein_sequence(user_input)
                    if not protein_sequence:
                        print(f"INFO: Empty sequence generated for '{user_input}'. Attempt {attempt + 1}/{max_attempts}")
                        continue  # Try again if the sequence is empty

                    print(f"INFO: Process Json protein_sequence (Attempt {attempt + 1})", protein_sequence)
                    optimized_sequence, secondary_structure = analyze_and_optimize_protein(protein_sequence)
                    
                    if not optimized_sequence or not secondary_structure:
                        print(f"INFO: Empty optimized sequence or secondary structure. Attempt {attempt + 1}/{max_attempts}")
                        continue  # Try again if either is empty

                    entry['assistant'] = {
                        'original_sequence': protein_sequence,
                        'optimized_sequence': optimized_sequence,
                        'secondary_structure': secondary_structure
                    }
                    print("INFO: Process Json optimized_sequence", optimized_sequence)
                    print("INFO: Process Json secondary_structure", secondary_structure)
                    processed_count += 1
                    break  # Successfully processed, exit the attempt loop
                except Exception as e:
                    print(f"INFO: Error in attempt {attempt + 1}: {str(e)}")
                    if attempt == max_attempts - 1:  # If it's the last attempt
                        entry['assistant'] = {'error': f"Failed after {max_attempts} attempts: {str(e)}"}
            
            # Save progress every 10 generations
            if processed_count % 10 == 0:
                save_progress(data, output_path)
                print(f"INFO: Progress saved after {processed_count} generations")
    
    # Final save
    save_progress(data, output_path)
    print(f"INFO: Process Json Updated data saved to {output_path}")

def save_progress(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

# Run the process_json function
process_json('generated_prompts.json', 'generated_prompts_with_sequences_and_analysis.json')