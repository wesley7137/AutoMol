import xml.etree.ElementTree as ET
import pandas as pd
import tqdm
import os
from typing import List, Dict, Any
from datetime import datetime
def parse_uniprot_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    for entry in tqdm.tqdm(root.findall('{http://uniprot.org/uniprot}entry'), desc="Parsing XML"):
        entry_data = {
            'accession': entry.find('{http://uniprot.org/uniprot}accession').text,
            'name': entry.find('{http://uniprot.org/uniprot}name').text,
            'sequence': entry.find('{http://uniprot.org/uniprot}sequence').text.replace('\n', ''),
            'length': entry.find('{http://uniprot.org/uniprot}sequence').attrib['length'],
            'function': '',
            'protein_name': '',
            'gene_names': [],
            'organism': '',
            'subcellular_location': '',
            'go_terms': [],
            'keywords': [],
            'disease': '',
            'pathway': ''
        }
        
        # Extract protein name
        protein = entry.find('{http://uniprot.org/uniprot}protein')
        if protein is not None:
            recommended_name = protein.find('{http://uniprot.org/uniprot}recommendedName')
            if recommended_name is not None:
                full_name = recommended_name.find('{http://uniprot.org/uniprot}fullName')
                if full_name is not None:
                    entry_data['protein_name'] = full_name.text

        # Extract gene names
        gene = entry.find('{http://uniprot.org/uniprot}gene')
        if gene is not None:
            for name in gene.findall('{http://uniprot.org/uniprot}name'):
                entry_data['gene_names'].append(name.text)

        # Extract organism
        organism = entry.find('{http://uniprot.org/uniprot}organism')
        if organism is not None:
            scientific_name = organism.find('{http://uniprot.org/uniprot}name[@type="scientific"]')
            if scientific_name is not None:
                entry_data['organism'] = scientific_name.text

        # Extract comments (function, subcellular location, disease, pathway)
        for comment in entry.findall('{http://uniprot.org/uniprot}comment'):
            comment_type = comment.get('type')
            if comment_type == 'function':
                function_text = comment.find('{http://uniprot.org/uniprot}text')
                if function_text is not None:
                    entry_data['function'] = function_text.text
            elif comment_type == 'subcellular location':
                location_text = comment.find('{http://uniprot.org/uniprot}text')
                if location_text is not None:
                    entry_data['subcellular_location'] = location_text.text
            elif comment_type == 'disease':
                disease_text = comment.find('{http://uniprot.org/uniprot}text')
                if disease_text is not None:
                    entry_data['disease'] = disease_text.text
            elif comment_type == 'pathway':
                pathway_text = comment.find('{http://uniprot.org/uniprot}text')
                if pathway_text is not None:
                    entry_data['pathway'] = pathway_text.text

        # Extract GO terms
        for dbReference in entry.findall('{http://uniprot.org/uniprot}dbReference[@type="GO"]'):
            entry_data['go_terms'].append(dbReference.get('id'))

        # Extract keywords
        for keyword in entry.findall('{http://uniprot.org/uniprot}keyword'):
            entry_data['keywords'].append(keyword.text)

        data.append(entry_data)

    return pd.DataFrame(data)


def main():
    input_file = 'uniprot_sprot_human.xml'
    output_file = 'protein_data.csv'

    print(f"Parsing {input_file}...")
    new_df = parse_uniprot_xml(input_file)

    print(f"Appending to {output_file}...")
    try:
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        combined_df.drop_duplicates(subset='accession', keep='last', inplace=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Appended {len(new_df)} entries. Total entries: {len(combined_df)}")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_output_file = f'protein_data_{timestamp}.csv'
        
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except:
                print(f"Couldn't read {output_file}. Creating new file with only new data.")
                combined_df = new_df
        else:
            combined_df = new_df

        combined_df.drop_duplicates(subset='accession', keep='last', inplace=True)
        combined_df.to_csv(new_output_file, index=False)
        print(f"Permission denied for {output_file}. Created new file: {new_output_file}")
        print(f"Appended {len(new_df)} entries. Total entries: {len(combined_df)}")

    print("Conversion complete!")

if __name__ == "__main__":
    main()