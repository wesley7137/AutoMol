import os
import subprocess

def create_foldseek_db(pdb_dir, db_name):
    """
    Create a Foldseek database from PDB files.
    
    :param pdb_dir: Directory containing PDB files
    :param db_name: Name for the Foldseek database
    """
    cmd = f"foldseek createdb {pdb_dir} {db_name}"
    subprocess.run(cmd, shell=True, check=True)
    print(f"Foldseek database '{db_name}' created successfully.")

def foldseek_search(query_pdb, db_name, output_file):
    """
    Perform a Foldseek search.
    
    :param query_pdb: Path to the query PDB file
    :param db_name: Name of the Foldseek database to search against
    :param output_file: Name of the output file for search results
    """
    cmd = f"foldseek search {query_pdb} {db_name} alignmentDB {output_file} -a"
    subprocess.run(cmd, shell=True, check=True)
    print(f"Foldseek search completed. Results saved to {output_file}")

def parse_foldseek_results(result_file):
    """
    Parse Foldseek search results.
    
    :param result_file: Path to the Foldseek search results file
    :return: List of dictionaries containing parsed results
    """
    results = []
    with open(result_file, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            results.append({
                'query': fields[0],
                'target': fields[1],
                'fident': float(fields[2]),
                'alnlen': int(fields[3]),
                'mismatch': int(fields[4]),
                'gapopen': int(fields[5]),
                'qstart': int(fields[6]),
                'qend': int(fields[7]),
                'tstart': int(fields[8]),
                'tend': int(fields[9]),
                'evalue': float(fields[10]),
                'bits': float(fields[11])
            })
    return results

# Main execution
if __name__ == "__main__":
    # Paths and names
    pdb_dir = "./pdb_files"  # Directory containing PDB files for the database
    db_name = "my_foldseek_db"
    query_pdb = "./generated_structure.pdb"  # Your generated structure
    output_file = "search_results.m8"

    # Create Foldseek database
    create_foldseek_db(pdb_dir, db_name)

    # Perform Foldseek search
    foldseek_search(query_pdb, db_name, output_file)

    # Parse and analyze results
    results = parse_foldseek_results(output_file)

    # Print top 5 results
    print("\nTop 5 structural matches:")
    for i, result in enumerate(sorted(results, key=lambda x: x['fident'], reverse=True)[:5]):
        print(f"{i+1}. Target: {result['target']}, Similarity: {result['fident']:.2f}, E-value: {result['evalue']}")
