import requests
import pandas as pd
import time
from ratelimit import limits, sleep_and_retry

# Define rate limits
CALLS_PER_SECOND = 1
CALLS_PER_MINUTE = 60

print("Starting the data fetching process...")

@sleep_and_retry
@limits(calls=CALLS_PER_SECOND, period=1)
@limits(calls=CALLS_PER_MINUTE, period=60)
def rate_limited_request(url):
    print(f"Making a rate-limited request to: {url}")
    response = requests.get(url)
    if response.status_code == 429:  # Too Many Requests
        retry_after = int(response.headers.get('Retry-After', 60))
        print(f"Rate limit exceeded. Waiting for {retry_after} seconds before retrying...")
        time.sleep(retry_after)
        return rate_limited_request(url)  # Retry the request
    return response

def fetch_uniprot_data(uniprot_id):
    print(f"Fetching UniProt data for ID: {uniprot_id}")
    base_url = "https://rest.uniprot.org/uniprotkb/"
    url = f"{base_url}{uniprot_id}"
    response = rate_limited_request(url)
    
    if response.status_code == 200:
        data = response.json()
        sequence = data.get('sequence', {}).get('value', '')
        function = data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
        gene_name = data.get('genes', [{}])[0].get('geneName', {}).get('value', '')
        
        pdb_ids = [xref['id'] for xref in data.get('uniProtKBCrossReferences', []) if xref['database'] == 'PDB']
        
        print(f"Successfully fetched data for UniProt ID: {uniprot_id}")
        return {
            'UniProt_ID': uniprot_id,
            'Sequence': sequence,
            'Function': function,
            'Gene_Name': gene_name,
            'PDB_IDs': ','.join(pdb_ids),
            'Exists': True
        }
    else:
        print(f"Failed to fetch data for UniProt ID: {uniprot_id}")
        return {
            'UniProt_ID': uniprot_id,
            'Sequence': '',
            'Function': '',
            'Gene_Name': '',
            'PDB_IDs': '',
            'Exists': False
        }

def fetch_pdb_data(pdb_id):
    print(f"Fetching PDB data for ID: {pdb_id}")
    url = f"https://files.rcsb.org/view/{pdb_id}.pdb"
    response = rate_limited_request(url)
    
    if response.status_code == 200:
        pdb_content = response.text
        resolution = next((line.split()[3] for line in pdb_content.split('\n') if line.startswith('REMARK   2 RESOLUTION.')), 'N/A')
        
        print(f"Successfully fetched PDB data for ID: {pdb_id}")
        return {
            'PDB_ID': pdb_id,
            'Resolution': resolution,
            'PDB_Content': pdb_content
        }
    else:
        print(f"Failed to fetch PDB data for ID: {pdb_id}")
        return {
            'PDB_ID': pdb_id,
            'Resolution': 'N/A',
            'PDB_Content': ''
        }

def save_dataframes(uniprot_results, pdb_results, original_df):
    uniprot_df = pd.DataFrame(uniprot_results)
    pdb_df = pd.DataFrame(pdb_results)
    
    final_df = pd.merge(original_df, uniprot_df, left_on='Uniprot ID', right_on='UniProt_ID', how='left')
    
    final_df.to_csv('updated_aging_proteins_uniprot.csv', index=False)
    pdb_df.to_csv('aging_proteins_pdb.csv', index=False)
    
    print("Intermediate results saved.")

# Load your dataset
print("Loading the aging proteins dataset...")
df = pd.read_csv('aging_human.csv', sep=';')
print(f"Loaded {len(df)} proteins from the dataset.")

# Process each UniProt ID
uniprot_results = []
pdb_results = []

print("Starting to process UniProt IDs...")
for index, uniprot_id in enumerate(df['Uniprot ID'], 1):
    uniprot_data = fetch_uniprot_data(uniprot_id)
    uniprot_results.append(uniprot_data)
    
    if uniprot_data['PDB_IDs']:
        print(f"Found PDB IDs for UniProt ID {uniprot_id}: {uniprot_data['PDB_IDs']}")
        for pdb_id in uniprot_data['PDB_IDs'].split(','):
            pdb_data = fetch_pdb_data(pdb_id)
            pdb_results.append(pdb_data)
    
    # Save intermediate results every 5 entries
    if index % 5 == 0:
        save_dataframes(uniprot_results, pdb_results, df)

print("Finished processing all UniProt IDs.")

# Final save
save_dataframes(uniprot_results, pdb_results, df)

print("Datasets updated and saved as 'updated_aging_proteins_uniprot.csv' and 'aging_proteins_pdb.csv'")
print("Data fetching process completed successfully.")
