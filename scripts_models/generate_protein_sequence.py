import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import tempfile
import os
from langchain_ollama import ChatOllama





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


