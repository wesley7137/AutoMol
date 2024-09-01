import os
import pandas as pd
import logging
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_ollama import ChatOllama
from query_db import OmicsRAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize RAG pipeline
vector_store_path = (r"C:\Users\wes\vectordb_data_good\omics_vector_store")
rag_pipeline = OmicsRAGPipeline(vector_store_path)
rag_pipeline.load_vector_store()
rag_pipeline.create_rag_pipeline()

# Initialize ChatOllama for description generation
llm = ChatOllama(
    model="mathstral:7b-v0.1-q6_K",
    temperature=0.3,
    max_tokens=4098,
    verbose=True,
    stream=True,
)

def get_context(row):
    # Use the RAG pipeline to get context
    query = f"Provide detailed information about the protein with PDB ID {row['PDB_ID']} including its structure, function, and any notable features."
    result = rag_pipeline.query(query)
    return result['result']  # This should contain the retrieved context

def generate_description(row, context):
    try:
        system = "You are a helpful assistant that generates detailed descriptions of proteins. You have extensive knowledge about protein structures, functions, and features."
        prompt = f"""Generate a detailed description of the protein with PDB ID {row['PDB_ID']}.
        Use the following information and context to create a comprehensive description:
        
        
        Include information about:
        1. The protein's structure (e.g., alpha helices, beta sheets, domains)
        2. Its function and role in biological processes
        3. Any notable features or characteristics
        4. Its interactions with other molecules or proteins
        5. Any known mutations or variants and their effects
        6. Its relevance in research or medical applications

        Context from scientific literature: {context[:4000]} This description will be used in a multi-modal dataset for training a model to generate detailed natural language descriptions of proteins. It should be the same degree of quality as the example below:
        
        <example_description>
        
        Protein Description for PDB ID: 4XYZ

        The protein with PDB ID 4XYZ, also known as Thermostable Beta-Glucosidase A (TBG-A), is a crucial enzyme in the cellulase complex of the thermophilic bacterium Thermotoga maritima. This protein was crystallized at a resolution of 1.8 Angstroms, providing a detailed view of its structure and active site.

        Structure:
        TBG-A is a globular protein with a (β/α)8 TIM barrel fold, characteristic of the glycoside hydrolase family 1. The structure consists of eight parallel β-strands forming the central barrel, surrounded by eight α-helices. The active site is located at the C-terminal end of the β-barrel, forming a deep pocket that accommodates the substrate.

        Key structural features include:
        1. A highly conserved glutamate residue (Glu166) acting as the catalytic nucleophile
        2. Another glutamate (Glu355) functioning as the acid/base catalyst
        3. A network of hydrogen bonds stabilizing the active site geometry

        Function:
        TBG-A catalyzes the hydrolysis of β-1,4-glycosidic bonds in cellobiose and other short chain oligosaccharides. Its primary role is in the final step of cellulose degradation, converting cellobiose to glucose. The enzyme shows remarkable thermostability, retaining its activity at temperatures up to 80°C, which is attributed to its compact structure and increased number of salt bridges compared to mesophilic homologs.

        Notable features:
        1. Thermostability: TBG-A maintains its structural integrity and catalytic activity at high temperatures, making it valuable for industrial applications in biofuel production.
        2. Broad substrate specificity: While primarily acting on cellobiose, TBG-A can also hydrolyze other β-glucosides, including some plant-derived flavonoids.
        3. Metal ion coordination: A calcium ion is coordinated near the active site, contributing to the protein's structural stability.

        Interactions:
        TBG-A functions as a monomer but has been observed to form dimers in solution at high concentrations. It interacts with its substrates through an extensive hydrogen bonding network and hydrophobic interactions in the active site pocket. The enzyme also shows synergistic activity with other cellulases in the degradation of complex cellulosic biomass.

        Mutations and Variants:
        Several mutant variants of TBG-A have been studied:
        1. E166Q: This mutation in the catalytic nucleophile results in a completely inactive enzyme, confirming the critical role of Glu166 in catalysis.
        2. W168F: A mutation in the substrate binding pocket that reduces the enzyme's affinity for larger substrates while maintaining activity on smaller ones.
        3. N220F: This mutation in a loop region near the active site increases the enzyme's thermostability by introducing additional hydrophobic interactions.

        Relevance in Research and Applications:
        TBG-A has gained significant attention in biotechnology due to its thermostability and efficient catalytic activity. Key areas of research and application include:
        1. Biofuel production: TBG-A is being explored for use in the conversion of cellulosic biomass to fermentable sugars.
        2. Food industry: The enzyme's ability to hydrolyze certain flavonoid glycosides is being investigated for enhancing the bioavailability of these compounds in functional foods.
        3. Structural biology: As a model thermostable protein, TBG-A has contributed to our understanding of the molecular basis of protein thermostability.
        4. Protein engineering: Efforts are ongoing to further enhance TBG-A's catalytic efficiency and thermostability through rational design and directed evolution approaches.

        In conclusion, TBG-A (PDB ID: 4XYZ) represents a significant enzyme in both basic research and applied biotechnology, offering insights into protein structure-function relationships and holding promise for various industrial applications.

        </example_description>
        """
        # Limit context to 2000 characters to avoid overwhelming the model

        
        input_message = system + prompt
        completion = llm.invoke(input_message)
        response = completion.content
        logger.info(f"Generated description for PDB ID {row['PDB_ID']}")
        return response
    except Exception as e:
        logger.error(f"Error generating description for PDB ID {row['PDB_ID']}: {e}")
        return "Description generation failed."

def process_batch(batch):
    batch['Context'] = batch.apply(get_context, axis=1)
    batch['Description'] = batch.apply(lambda row: generate_description(row, row['Context']), axis=1)
    return batch

def process_csv(file_path, batch_size=10, save_interval=50):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Processing file: {file_path}")
        logger.info(f"Number of entries: {len(df)}")
        
        output_file = os.path.splitext(file_path)[0] + '_with_descriptions.csv'
        temp_file = os.path.splitext(file_path)[0] + '_temp.json'
        
        # Load progress if temp file exists
        if os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                progress = json.load(f)
            start_index = progress['last_processed_index'] + 1
            df.loc[:start_index-1, ['Context', 'Description']] = progress['processed_data']
        else:
            start_index = 0
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in tqdm(range(start_index, len(df), batch_size)):
                batch = df.iloc[i:i+batch_size].copy()
                futures.append(executor.submit(process_batch, batch))
                
                if (i + 1) % save_interval == 0 or i + batch_size >= len(df):
                    for future in as_completed(futures):
                        result = future.result()
                        df.loc[result.index, ['Context', 'Description']] = result[['Context', 'Description']]
                    
                    # Save progress
                    progress = {
                        'last_processed_index': i + batch_size - 1,
                        'processed_data': df.loc[:i+batch_size-1, ['Context', 'Description']].to_dict()
                    }
                    with open(temp_file, 'w') as f:
                        json.dump(progress, f)
                    
                    logger.info(f"Saved progress at index {i + batch_size}")
                    futures = []
        
        df.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to: {output_file}")
        
        # Remove temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")

def main():
    input_directory = './split_pdb_data_with_img'  # Update this to your input directory
    
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            process_csv(file_path)

if __name__ == "__main__":
    main()