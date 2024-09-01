import requests
import pandas as pd
import torch
import logging
import requests
from langchain_ollama import ChatOllama
import json
import csv
from langchain_openai import ChatOpenAI
import sys
import os
from openai import OpenAI
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Initialize the OpenAI client
client = OpenAI(
    api_key="lm-studio",
    base_url="http://localhost:1234/v1"
)
# Function to set maximum CSV field size
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




def fetch_uniprot_id_from_pdb(pdb_id):
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        uniprot_ids = []
        for polymer in data['rcsb_entry_container_identifiers']['polymer_entity_ids']:
            entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{polymer}"
            entity_response = requests.get(entity_url)
            if entity_response.status_code == 200:
                entity_data = entity_response.json()
                if 'reference_sequence_identifiers' in entity_data['rcsb_polymer_entity_container_identifiers']:
                    for reference in entity_data['rcsb_polymer_entity_container_identifiers']['reference_sequence_identifiers']:
                        if reference['database_name'] == 'UniProt':
                            uniprot_ids.append(reference['database_accession'])
                else:
                    logger.warning(f"No 'reference_sequence_identifiers' found for PDB ID {pdb_id}, polymer {polymer}")
        return uniprot_ids
    else:
        logger.error(f"Failed to retrieve data for PDB ID {pdb_id}")
        return None
    
    
def fetch_uniprot_data(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.xml"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve data for UniProt ID {uniprot_id}")
        return None
    
import xml.etree.ElementTree as ET

def extract_uniprot_info(xml_data):
    root = ET.fromstring(xml_data)
    namespace = {'ns': 'http://uniprot.org/uniprot'}

    protein_name = root.find('.//ns:protein/ns:recommendedName/ns:fullName', namespace).text
    gene_name = root.find('.//ns:gene/ns:name[@type="primary"]', namespace).text
    organism = root.find('.//ns:organism/ns:name[@type="scientific"]', namespace).text
    function = root.find('.//ns:comment[@type="function"]/ns:text', namespace).text

    return {
        'protein_name': protein_name,
        'gene_name': gene_name,
        'organism': organism,
        'function': function
    }
    

def generate_advanced_description(uniprot_info, pdb_id):
    try:
        system = "You are an expert biochemist creating detailed, scientifically accurate descriptions of proteins for upper-level undergraduate students."
        prompt = f"""Generate a comprehensive, college-level description of the protein with PDB ID {pdb_id}.
        Use the following information and context to create your description:

        Context from UniProt: {uniprot_info}
        
            Structure your description as follows:

            1. Introduction:
            - Full name and abbreviations of the protein
            - Organism(s) of origin and cellular localization
            - Primary function and biological significance

            2. Structural Characteristics:
            - Overall structural classification and dimensions
            - Detailed secondary and tertiary structure elements
            - Active sites, binding pockets, or other functionally important regions
            - Notable structural features visible in 3D representations

            3. Biochemical Function:
            - Detailed explanation of the protein's molecular mechanism
            - Substrates, products, and cofactors involved in its function
            - Regulatory mechanisms (if applicable)

            4. Physiological Role:
            - The protein's role in cellular or organismal processes
            - Its place in relevant biochemical pathways
            - Impact of its function on broader biological systems

            5. Clinical Significance:
            - Associated diseases or disorders
            - Current diagnostic or therapeutic applications

            6. Uniqueness of the protein:
            - Explain why this protein is rare or unique
            - Provide specific examples of its uniqueness
            - What separates this protein from others

            7. Current Research and Future Applications:
            - Ongoing areas of research involving this protein
            - Potential future applications in medicine, biotechnology, or other fields
            - For each application, provide specific details on the biological basis and potential methodologies

            8. Structural Insights:
            - Describe 2-3 key structural features that would be prominent in 3D visualizations
            - Explain how these features relate to the protein's function or applications

            Guidelines:
            - Use precise scientific terminology, providing brief explanations where necessary
            - Include relevant numerical data (e.g., molecular weight, key residue numbers)
            - Emphasize structure-function relationships
            - When discussing applications, provide specific biological mechanisms and methodologies
            - Maintain scientific accuracy while ensuring understandability for undergraduate students
            - If certain information is unavailable, acknowledge this rather than speculate

            This description will accompany 2D and 3D visualizations of the protein in a dataset for training an AI model. Ensure your description complements and enhances understanding of these visual representations.

            Example Description:

            Protein Description for PDB ID: 7XYZ

            1. Introduction:
            Superoxide Dismutase 1 (SOD1) is a crucial antioxidant enzyme found in the cytoplasm of virtually all eukaryotic cells. This 32 kDa homodimeric protein plays a vital role in cellular defense against oxidative stress by catalyzing the dismutation of superoxide radicals (O2•-) into molecular oxygen and hydrogen peroxide.

            2. Structural Characteristics:
            SOD1 is a β-barrel protein composed of two identical subunits, each containing 153 amino acids. Each monomer adopts a Greek key β-barrel fold, consisting of eight antiparallel β-strands arranged in a flattened cylinder. The active site of each subunit contains one copper and one zinc ion, crucial for its catalytic function. A prominent feature in 3D models is the electrostatic loop (residues 121-142), which guides superoxide to the active site.

            3. Biochemical Function:
            SOD1 catalyzes the dismutation of superoxide through a two-step process:
            Cu2+-SOD + O2•- → Cu+-SOD + O2
            Cu+-SOD + O2•- + 2H+ → Cu2+-SOD + H2O2
            The copper ion undergoes alternate reduction and oxidation, while the zinc ion plays a structural role. The reaction occurs at a diffusion-limited rate, making SOD1 one of the most efficient enzymes known.

            4. Physiological Role:
            SOD1 is a primary antioxidant defense, protecting cells from the damaging effects of superoxide radicals produced as byproducts of cellular respiration. It works in concert with catalase and glutathione peroxidase to neutralize reactive oxygen species, maintaining cellular redox homeostasis and preventing oxidative damage to proteins, lipids, and DNA.

            5. Clinical Significance:
            Mutations in the SOD1 gene are associated with familial amyotrophic lateral sclerosis (fALS), a progressive neurodegenerative disorder. These mutations often lead to protein misfolding and aggregation, contributing to motor neuron death. SOD1 is also implicated in other neurodegenerative diseases and has been a focus in cancer research due to its role in cellular redox balance.

            6. Uniqueness of the protein:
            Catalytic efficiency: SOD1 is one of the fastest enzymes known, with a catalytic rate (kcat/Km) approaching the diffusion limit, making it uniquely efficient in neutralizing superoxide radicals.
            Structural stability: Unlike many enzymes, SOD1 maintains its activity across a wide pH range (pH 5-9.5) and remains stable at temperatures up to 80°C, a rare characteristic for a human enzyme.
            Dual metal cofactor system: SOD1 uniquely utilizes both copper and zinc ions in its active site, with copper involved in the catalytic cycle and zinc providing structural stability.
            Evolutionary conservation: SOD1's structure and function are highly conserved across species, from bacteria to humans, indicating its fundamental importance in cellular metabolism.
            Multifunctional roles: Beyond its primary antioxidant function, SOD1 has been implicated in zinc and copper buffering, and potentially in cellular signaling, showcasing its versatility.

            7. Current Research and Future Applications:
            a) ALS Therapeutics: Research is ongoing to develop SOD1 stabilizers or antisense oligonucleotides to reduce mutant SOD1 expression. For example, tofersen, an antisense oligonucleotide targeting SOD1 mRNA, is in clinical trials. It works by binding to SOD1 mRNA, promoting its degradation and reducing the production of mutant SOD1 protein.

            b) Antioxidant Therapy: SOD1 mimetics are being investigated as potential treatments for conditions associated with oxidative stress. These small molecule compounds, such as M40403, mimic SOD1's catalytic activity. They could be used in treating ischemia-reperfusion injury by catalyzing the dismutation of superoxide radicals produced during reperfusion, thereby reducing tissue damage.

            c) Biomarker Development: The presence of misfolded SOD1 in cerebrospinal fluid is being explored as a potential biomarker for ALS. This could involve developing antibodies specific to misfolded SOD1 conformations for use in ELISA or immunohistochemistry assays, potentially enabling earlier diagnosis or monitoring of disease progression.

            d) Nanocatalysts: The high efficiency of SOD1 is inspiring the development of biomimetic nanocatalysts. By incorporating key structural features of SOD1's active site into nanoparticles, researchers aim to create highly efficient catalysts for industrial processes or environmental remediation. For instance, copper-zinc nanoparticles with a structure mimicking SOD1's active site could be used for breaking down pollutants in wastewater treatment.

            7. Structural Insights:
            In 3D visualizations of SOD1, three key features stand out:
            1. The β-barrel structure of each subunit, visible as a compact, cylindrical arrangement of β-strands.
            2. The copper and zinc ions in the active site, appearing as distinct, colored spheres within each subunit.
            3. The electrostatic loop, protruding from the main body of the protein, which guides superoxide to the active site.

            These structural elements collectively contribute to SOD1's remarkable catalytic efficiency and are crucial to its function as a cellular antioxidant defense mechanism.        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        advanced_description = client.chat.completions.create(
            model="lmstudio-community/mathstral-7B-v0.1-GGUF/mathstral-7B-v0.1-Q8_0.gguf",
            messages=messages,
            temperature=0.3,
            max_tokens=2048
        )

        return advanced_description.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating {advanced_description} description: {e}")
        return f"{advanced_description.capitalize()} description generation failed."

def generate_simple_description(uniprot_info, pdb_id):
    try:
        system = "You are a knowledgeable biochemistry professor creating informative protein descriptions for first-year college students. Your task is to balance scientific accuracy with clarity, introducing students to key concepts in molecular biology."

        prompt = f"""Generate a comprehensive description of the protein with PDB ID {pdb_id}.
        Use the following information and context to create your description:

        {uniprot_info}

        Structure your description as follows:

        1. Introduction:
        - Full name and common abbreviations of the protein
        - Where it's found (organism and cellular location)
        - Its main function in simple terms

        2. Protein Structure:
        - Overall shape and size of the protein
        - Basic structural elements (e.g., alpha helices, beta sheets)
        - Any important regions or sites (e.g., active sites, binding pockets)
        - Describe 1-2 key features that would be visible in a 3D model

        3. How It Works:
        - Explain the protein's function in biological terms
        - Describe its role in relevant cellular processes
        - Mention any key molecules it interacts with

        4. Importance in Health and Disease:
        - Discuss the protein's role in normal bodily functions
        - Explain its involvement in any diseases or disorders


        5. Uniqueness of the protein:
        - Explain why this protein is rare or unique
        - Provide specific examples of its uniqueness
        - What separates this protein from others
        

        6. Current Research and Future Possibilities:
        - Describe how this protein is currently used in research or medicine
        - Explain 2-3 potential future applications, providing specific details on why and how it could be used

        Guidelines:
        - Use clear language suitable for first-year college students
        - Explain scientific terms when you introduce them
        - Focus on the most important concepts without overwhelming detail
        - Provide specific examples to illustrate key points
        - When discussing applications, give concrete examples with basic explanations of the underlying biology
        - If certain information isn't available, it's okay to say so

        Remember, this description will be paired with 2D and 3D visualizations of the protein. Make sure your description helps students understand what they might see in these images.

        Example Description:

        Protein Description for PDB ID: 8XYZ

        1. Introduction:
        Superoxide Dismutase 1 (SOD1) is an important enzyme found in nearly all living cells, from bacteria to humans. In human cells, it's mainly located in the cytoplasm, which is the gel-like substance filling the cell. SOD1's main job is to protect the cell from damage caused by harmful molecules called superoxide radicals.

        2. Protein Structure:
        SOD1 is a relatively small protein, made up of two identical parts (called subunits) that work together. Each subunit is about 153 amino acids long. The overall shape of SOD1 resembles a flattened barrel, which scientists call a β-barrel structure. This shape is formed by flat sheets of protein (beta sheets) arranged in a circular pattern.

        A key feature you'd see in a 3D model of SOD1 is its active site. This is where the main action happens, and it contains two important metal ions: one copper and one zinc. These metal ions appear as small, colored spheres in 3D visualizations and are crucial for SOD1's function.

        3. How It Works:
        SOD1's job is to break down superoxide radicals, which are harmful byproducts of cellular processes. It does this through a chemical reaction called dismutation, which converts superoxide into ordinary oxygen and hydrogen peroxide. This process happens extremely quickly – SOD1 is one of the fastest working enzymes we know!

        The copper ion in the active site is the main player in this reaction. It alternates between two forms (Cu2+ and Cu+) as it breaks down the superoxide. The zinc ion, on the other hand, helps maintain the protein's structure.

        4. Importance in Health and Disease:
        By breaking down superoxide radicals, SOD1 helps protect cells from a type of damage called oxidative stress. This is important because oxidative stress is linked to aging and many diseases.
        
        

        When SOD1 doesn't work correctly, it can lead to health problems. The most well-known of these is a type of amyotrophic lateral sclerosis (ALS), also known as Lou Gehrig's disease. Some forms of ALS are caused by mutations in the SOD1 gene, which lead to a misfolded protein that can harm motor neurons.

        5. Uniqueness of the protein:
        Super-fast reaction: SOD1 is one of the fastest working enzymes we know. It can break down superoxide radicals almost as quickly as they bump into the enzyme!
        Tough and adaptable: Unlike many proteins that are picky about their environment, SOD1 can work in a wide range of conditions. It stays active whether the environment is slightly acidic or basic, and it can withstand higher temperatures than many human proteins.
        Two-metal team: SOD1 is special because it uses two different metal ions (copper and zinc) to do its job. The copper helps break down superoxide, while the zinc helps hold the protein's shape.
        Found everywhere: SOD1 is so important that it's found in almost every living thing, from tiny bacteria to complex human cells. Its job is so crucial that it has changed very little over millions of years of evolution.
        Multi-tasking protein: While SOD1's main job is to protect cells from damage, scientists are discovering that it might do other important things too, like helping cells manage their metal levels and possibly even sending signals within the cell.

        6. Current Research and Future Possibilities:
        Currently, SOD1 is a major focus in ALS research. Scientists are working on several approaches to address SOD1-related ALS:

        a) Gene Therapy: Researchers are developing methods to reduce the production of mutant SOD1 in patients with ALS. One promising approach uses special molecules called antisense oligonucleotides. These molecules can bind to the genetic instructions for making SOD1 and signal the cell to destroy them, thereby reducing the amount of harmful mutant SOD1 produced.

        b) Protein Stabilizers: Another avenue of research involves creating drugs that can stabilize the SOD1 protein, preventing it from misfolding. This approach targets the root cause of SOD1-related ALS at the protein level. Scientists are using their knowledge of SOD1's structure to design molecules that can bind to the protein and keep it in its correct shape.

        c) Biomarker Development: Researchers are also exploring whether SOD1 could be used as a biomarker for ALS. This means looking for abnormal forms of SOD1 in a patient's cerebrospinal fluid (the liquid surrounding the brain and spinal cord) to diagnose ALS earlier or track its progression. This could involve developing specific antibodies that recognize misfolded SOD1, which could then be used in diagnostic tests.

        In addition to its role in ALS research, SOD1's powerful antioxidant properties make it interesting for other applications. For example, scientists are studying whether SOD1 or drugs that mimic its activity could be used to protect cells from damage in various conditions, from heart attacks to radiation exposure.

        Understanding SOD1 not only helps us learn about diseases like ALS but also provides insights into how cells protect themselves from damage. As research progresses, this small but mighty protein may play a big role in developing new treatments for a variety of conditions."""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        simple_description = client.chat.completions.create(
            model="lmstudio-community/mathstral-7B-v0.1-GGUF/mathstral-7B-v0.1-Q8_0.gguf",
            messages=messages,
            temperature=0.3,
            max_tokens=2048
        )

        return simple_description.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating {simple_description} description: {e}")
        return f"{simple_description.capitalize()} description generation failed."




def process_csv_file(csv_file_path, output_file_path, skipped_file_path):
    logger.info(f"Starting to process CSV file: {csv_file_path}")
    
    # Read existing entries
    existing_entries = {}
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as existing_file:
            existing_entries = json.load(existing_file)
    
    logger.info(f"Found {len(existing_entries)} existing entries")

    # Process new entries
    new_entries = {}
    skipped_entries = []  # List to store skipped entries
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for i, row in enumerate(csv_reader, 1):
            pdb_id = row['PDB_ID']
            
            # Skip already processed entries
            if pdb_id in existing_entries:
                logger.info(f"Skipping already processed entry: {pdb_id}")
                continue
            
            try:
                logger.info(f"Processing entry {i}: PDB ID {pdb_id}")
                
                # Fetch UniProt IDs
                uniprot_ids = fetch_uniprot_id_from_pdb(pdb_id)
                if not uniprot_ids:
                    logger.warning(f"Skipped: {pdb_id} - No UniProt IDs found")
                    skipped_entries.append({"pdb_id": pdb_id, "reason": "No UniProt IDs found"})
                    continue

                # Process the first UniProt ID
                uniprot_id = uniprot_ids[0]
                uniprot_data = fetch_uniprot_data(uniprot_id)
                if not uniprot_data:
                    logger.warning(f"Skipped: {pdb_id} - Failed to fetch UniProt data")
                    skipped_entries.append({"pdb_id": pdb_id, "reason": "Failed to fetch UniProt data"})
                    continue

                uniprot_info = extract_uniprot_info(uniprot_data)             
                
                logger.info(f"Generating advanced description for PDB ID: {pdb_id}")
                advanced_description = generate_advanced_description(uniprot_info, pdb_id)
                logger.info("Advanced description generated successfully")

                logger.info(f"Generating simple description for PDB ID: {pdb_id}")
                simple_description = generate_simple_description(uniprot_info, pdb_id)
                logger.info("Simple description generated successfully")
                
                new_entries[pdb_id] = {
                    "uniprot_id": uniprot_id,  # Add UniProt ID here
                    "advanced_description": advanced_description,
                    "simple_description": simple_description
                }
                logger.info(f"Processed: {pdb_id}")

                # Save progress every 5 entries
                if len(new_entries) % 5 == 0:
                    save_json(output_file_path, {**existing_entries, **new_entries})
                    logger.info(f"Saved progress: {len(new_entries)} new entries")

            except Exception as e:
                logger.error(f"Error processing entry {i}: {str(e)}", exc_info=True)
                skipped_entries.append({"pdb_id": pdb_id, "reason": str(e)})

    # Save final results
    save_json(output_file_path, {**existing_entries, **new_entries})
    
    # Save skipped entries
    save_json(skipped_file_path, skipped_entries)
    
    logger.info(f"Processing complete. Total new entries processed: {len(new_entries)}")
    logger.info(f"Results saved to: {output_file_path}")
    logger.info(f"Skipped entries saved to: {skipped_file_path}")

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=2, ensure_ascii=False)

# Example usage
csv_file_path = r'C:\Users\wes\vectordb_data_good\split_pdb_data_with_img\pdb_dataset_with_images_chunk_4.csv'
output_file_path = r'C:\Users\wes\vectordb_data_good\split_pdb_data_with_img\pdb_dataset_with_images_chunk_4_detailed_dual_descriptions.json'
skipped_file_path = r'C:\Users\wes\vectordb_data_good\split_pdb_data_with_img\pdb_dataset_with_images_chunk_4_skipped_entries.json'
process_csv_file(csv_file_path, output_file_path, skipped_file_path)