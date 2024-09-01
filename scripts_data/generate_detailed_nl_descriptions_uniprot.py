import os
import pandas as pd
import requests
import logging
from tqdm import tqdm
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_pdb_description(pdb_id):
    if not isinstance(pdb_id, str):
        logger.error(f"Invalid PDB ID: {pdb_id}, type: {type(pdb_id)}")
        return f"Error: Invalid PDB ID format"

    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant information
        struct_title = data.get('struct', {}).get('title', '')
        struct_keywords = data.get('struct_keywords', [])
        keywords = ', '.join([kw.get('text', '') for kw in struct_keywords if isinstance(kw, dict)])
        
        # Get entity information
        entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
        entity_response = requests.get(entity_url)
        entity_response.raise_for_status()
        entity_data = entity_response.json()
        
        entity_name = entity_data.get('rcsb_polymer_entity', {}).get('pdbx_description', '')
        entity_type = entity_data.get('entity_poly', {}).get('type', '')
        
        description = f"Title: {struct_title}\n"
        description += f"Keywords: {keywords}\n"
        description += f"Entity Name: {entity_name}\n"
        description += f"Entity Type: {entity_type}\n"
        
        return description.strip()
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data for PDB ID {pdb_id}: {str(e)}")
        return f"Error fetching description for PDB ID {pdb_id}"

def process_csv(file_path, batch_size=10):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Processing file: {file_path}")
        logger.info(f"Number of entries: {len(df)}")
        logger.info(f"Columns: {df.columns}")
        
        if 'PDB_ID' not in df.columns:
            logger.error("'PDB_ID' column not found in the CSV file.")
            return
        
        output_file = os.path.splitext(file_path)[0] + '_with_descriptions.csv'
        
        # Create a new column for descriptions
        df['Description'] = ''
        
        # Process in batches
        for i in tqdm(range(0, len(df), batch_size)):
            end = min(i + batch_size, len(df))
            
            # Use .loc to set values
            for idx in range(i, end):
                pdb_id = df.loc[idx, 'PDB_ID']
                logger.info(f"Processing PDB ID: {pdb_id}, type: {type(pdb_id)}")
                description = get_pdb_description(pdb_id)
                df.loc[idx, 'Description'] = description
                logger.info(f"PDB ID: {pdb_id}, Description: {description[:100]}...")  # Log first 100 chars of description
            
            # Save progress after each batch
            if i == 0:
                df.iloc[:end].to_csv(output_file, index=False, mode='w')
            else:
                df.iloc[i:end].to_csv(output_file, index=False, mode='a', header=False)
            
            # Sleep to respect API rate limits
            time.sleep(1)
        
        logger.info(f"Processed data saved to: {output_file}")
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        logger.error(f"DataFrame info: {df.info()}")
        logger.error(f"First few rows of DataFrame: {df.head()}")

def main():
    input_directory = r'C:\Users\wes\vectordb_data_good\split_pdb_data_with_img'  # Update this to your input directory
    
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            process_csv(file_path)

if __name__ == "__main__":
    main()