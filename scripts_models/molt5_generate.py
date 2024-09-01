import logging
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)
class MolT5Translator:
    def __init__(self,  max_length=512):
        self.max_length = max_length
        
    def generate_molecule(self, input_text):
        logger.info("Generating molecule...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("laituan245/t5-v1_1-small-caption2smiles-ft-from-pretrained-c4")
            model = AutoModelForSeq2SeqLM.from_pretrained("laituan245/t5-v1_1-small-caption2smiles-ft-from-pretrained-c4")
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            outputs = model.generate(input_ids, num_beams=5, max_length=self.max_length)
            molecule = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated molecule: {molecule}")
            if molecule:
                self.save_molecule_to_file(molecule)
            return molecule
        except Exception as e:
            logger.error(f"Error generating molecule: {str(e)}")
            return None

    def generate_caption(self, smiles):
        logger.info("Generating caption...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("laituan245/t5-v1_1-small-smiles2caption-ft-from-pretrained-c4")
            model = AutoModelForSeq2SeqLM.from_pretrained("laituan245/t5-v1_1-small-smiles2caption-ft-from-pretrained-c4")
            input_ids = tokenizer(smiles, return_tensors="pt").input_ids
            outputs = model.generate(input_ids, num_beams=5, max_length=self.max_length)
            caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated caption: {caption}")
            if caption:
                self.save_molecule_to_file(caption)
            return caption
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return None


    def save_molecule_to_file(self, molecule):
        output_dir = "generated_molecules"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"molecule_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, "w") as f:
                f.write(molecule)
            logger.info(f"Molecule saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving molecule to file: {str(e)}")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    translator = MolT5Translator()
    smiles = "CC(=O)NC1=CC=C(C=C1)C(=O)NC(=O)N[C@@H]2[C@H]([C@@H]([C@H](O2)CO)O)O)O"
    caption = translator.generate_caption(smiles)
    if caption:
        translator.save_molecule_to_file(caption)

if __name__ == "__main__":
    main()