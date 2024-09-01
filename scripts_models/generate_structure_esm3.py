from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig, ESMProteinError

# Will instruct you how to get an API key from huggingface hub, make one with "Read" permission.
#login()
def generate_structure_esm3():
    # This will download the model weights and instantiate the model on your machine.
    model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda") # or "cpu"

    # Generate a completion for a partial Carbonic Anhydrase (2vvb)
    prompt = "______________NGTQICAQTYAYQNGNAYANPYNPANLNLSIDPTKVENPKKLPPN_________________"
    protein = ESMProtein(sequence=prompt)

    try:
        # Generate the sequence
        result = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))
        if isinstance(result, ESMProteinError):
            print(f"Error during sequence generation: {result}")
            print(f"Error attributes: {dir(result)}")
            exit(1)
        protein = result
        generated_sequence = protein.sequence.strip()
        print(f"Generated sequence: {generated_sequence}")
        print(f"Generated sequence length: {len(generated_sequence)}")

        # Trim the sequence to 78 residues (the expected structure length)
        trimmed_sequence = generated_sequence[:78]
        print(f"Trimmed sequence length: {len(trimmed_sequence)}")

        # Create a new ESMProtein instance with the trimmed sequence
        protein = ESMProtein(sequence=trimmed_sequence)

        # Generate the structure for the trimmed sequence
        result = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
        if isinstance(result, ESMProteinError):
            print(f"Error during structure generation: {result}")
            print(f"Error attributes: {dir(result)}")
            exit(1)
        protein = result
        print(f"Structure shape: {protein.coordinates.shape}")

        protein.to_pdb("./generation.pdb")

        # Then we can do a round trip design by inverse folding the sequence and recomputing the structure
        protein.sequence = None
        protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8))
        protein.coordinates = None
        protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
        protein.to_pdb("./round_tripped.pdb")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error attributes: {dir(e)}")