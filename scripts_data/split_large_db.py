import pandas as pd
import os

def split_csv(input_file, output_dir, chunk_size=1000):
    print(f"Splitting {input_file} into chunks of {chunk_size} rows and saving to {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the CSV file in chunks
    chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)
    
    total_chunks = 0
    total_rows = 0
    
    for i, chunk in enumerate(chunk_iter, 1):
        # Create the output filename
        output_file = os.path.join(output_dir, f'chunk_{i:03d}.csv')
        
        # Write the chunk to a new CSV file
        chunk.to_csv(output_file, index=False)
        
        chunk_rows = len(chunk)
        total_rows += chunk_rows
        total_chunks += 1
        
        print(f"Wrote chunk {i} to {output_file}")
        print(f"Chunk rows: {chunk_rows}")
        print(f"Total rows processed: {total_rows}")
        
    print(f"\nSplitting complete.")
    print(f"Total chunks created: {total_chunks}")
    print(f"Total rows processed: {total_rows}")

# Usage
input_file = r"C:\Users\wes\vectordb_data_good\scripts\pdb_dataset_with_images.csv"
output_dir = r"C:\Users\wes\vectordb_data_good\scripts\split_pdb_db"
chunk_size = 1000  # Adjust this value based on your needs

split_csv(input_file, output_dir, chunk_size)