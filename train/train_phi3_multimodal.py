import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import pandas as pd
import random
import wandb
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image
import torch.optim as optim

torch.manual_seed(3)

# Initialize Weights & Biases for experiment tracking
run = wandb.init(project="protein-structure-phi3", entity="your_wandb_entity")

class ProteinStructureDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Randomly choose an interaction type, including the new 'generate_molecule'
        interaction_type = random.choice(['analyze', 'chat_simple', 'chat_advanced', 'generate_molecule'])
        
        # Define the JSON prompt structures based on the interaction type
        if interaction_type == 'analyze':
            input_data = {
                "<|system|>": "analyze <|end|>",
                "<|user|>\n": f"<|image_1|>You are an AI assistant specialized in protein structures. Provide a detailed analysis of the protein structure and its features when requested. Analyze this protein structure.<|end|>",
                "<|assistant|>\n": f"{row['Advanced_Description']}<|end|>"
            }
        elif interaction_type == 'chat_simple':
            input_data = {
                "<|system|>": "chat_simple <|end|>",
                "<|user|>\n": f"<|image_1|>You are an AI assistant specialized in protein structures. Provide simple and non-technical information about proteins when asked. Assist the user in whatever tasks they need. You are friendly and helpful. Tell me about this protein.<|end|>",
                "<|assistant|>\n": f"This image shows the structure of protein {row['UniProt_ID']}. It is involved in {row['Function']} and has PDB ID {row['PDB_ID']}. {row['Simple_Description']}<|end|>"
            }
        elif interaction_type == 'chat_advanced':
            input_data = {
                "<|system|>": "chat_advanced <|end|>",
                "<|user|>\n": f"<|image_1|>You are an AI assistant specialized in protein structures. Provide detailed and technical information about proteins when asked. Assist the user with expert-level details. Tell me about this protein.<|end|>",
                "<|assistant|>\n": f"This image shows the structure of protein {row['UniProt_ID']}. It belongs to the {row['Function']} family and its PDB ID is {row['PDB_ID']}. {row['Advanced_Description']}<|end|>"
            }
        elif interaction_type == 'generate_molecule':
            # Generate a natural language prompt for molecule generation
            user_prompt = "Generate a molecule or protein that would increase the functionality and mitophagy of the mitochondria within the human cell."
            input_data = {
                "<|system|>": "generate_molecule <|end|>",
                "<|user|>\n": f"<|image_1|>You are an AI assistant specialized in molecular generation. When asked, generate a molecule based on the given protein structure and description. {user_prompt}<|end|>",
                "<|assistant|>\n": f"Based on the input, generating a molecule that targets mitochondrial functionality and enhances mitophagy. The protein sequence for this molecule is: {row['Sequence']}. It is designed to {row['Advanced_Description']}<|end|>"
            }
        
        # Convert JSON to text format suitable for the tokenizer
        text = json.dumps(input_data)

        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length)
        
        try:
            image = Image.open(row['Image_Path']).convert("RGB")
            image = np.array(image)
        except (FileNotFoundError, IOError):
            return None

        encodings['pixel_values'] = image
        encodings['protein_id'] = row['UniProt_ID']
        
        return {key: torch.tensor(val) for key, val in encodings.items()}




# Load dataset
dataset_path = 'pdb_dataset_with_images_and_descriptions.csv'
df = pd.read_csv(dataset_path)

# Initialize processor and tokenizer
model_id = "microsoft/Phi-3-vision-128k-instruct"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
tokenizer = processor.tokenizer

# Split dataset
train_size = int(0.9 * len(df))
val_size = len(df) - train_size
train_indices, val_indices = random_split(range(len(df)), [train_size, val_size])
train_df = df.iloc[train_indices.indices]
val_df = df.iloc[val_indices.indices]

# Create datasets and dataloaders
train_dataset = ProteinStructureDataset(train_df, tokenizer, max_length=512)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataset = ProteinStructureDataset(val_df, tokenizer, max_length=512)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize model
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize optimizer
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training parameters
num_epochs = 1
eval_interval = 150
save_dir = './saved_models'
step = 0
accumulation_steps = 64

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

best_val_loss = float('inf')
best_model_path = None

# Select random samples for logging
num_log_samples = 10
log_indices = random.sample(range(len(val_dataset)), num_log_samples)

def evaluate(model, val_loader, device, tokenizer, step, log_indices, max_samples=None):
    model.eval()
    total_loss = 0
    table = wandb.Table(columns=["Image", "Ground Truth Text", "Predicted Text"])

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_samples and i >= max_samples:
                break
            if batch is None:
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = input_ids.clone().detach()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            if i in log_indices:
                predictions = torch.argmax(outputs.logits, dim=-1)
                gt_text = tokenizer.decode(labels[0], skip_special_tokens=True)
                pred_text = tokenizer.decode(predictions[0], skip_special_tokens=True)
                
                pil_img = to_pil_image(resize(torch.from_numpy(pixel_values.cpu().squeeze().numpy()).permute(2, 0, 1), (336, 336))).convert("RGB")
                table.add_data(wandb.Image(pil_img), gt_text, pred_text)

    wandb.log({"Evaluation Results step {}".format(step): table, "Step": step})
    avg_loss = total_loss / (i + 1)
    model.train()
    return avg_loss

# Training loop
model.train()
for epoch in range(num_epochs):
    total_train_loss = 0
    batch_count = 0

    for batch in train_loader:
        step += 1
        if batch is None:
            continue

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = input_ids.clone().detach()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss
        total_loss = loss
        total_loss.backward()

        if (step % accumulation_steps) == 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= accumulation_steps
            optimizer.step()
            optimizer.zero_grad()

        total_train_loss += total_loss.item()
        batch_count += 1

        wandb.log({"Batch Loss": total_loss.item(), "Step": step})
        print(f"Epoch: {epoch}, Step: {step}, Batch Loss: {total_loss.item()}")

        if step % eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, tokenizer=tokenizer, log_indices=log_indices, step=step)
            wandb.log({"Validation Loss": val_loss, "Step": step})
            print(f"Step: {step}, Validation Loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(save_dir, f"best_model")
                model.save_pretrained(best_model_path, safe_serialization=False)
                tokenizer.save_pretrained(best_model_path)

    avg_train_loss = total_train_loss / batch_count
    wandb.log({"Epoch": epoch, "Average Training Loss": avg_train_loss})
    print(f"Epoch: {epoch}, Average Training Loss: {avg_train_loss}")

# Log the best model to Weights & Biases
if best_model_path:
    run.log_model(
        path=best_model_path,
        name="phi3-v-protein-structure",
        aliases=["best"],
    )

# Finish the Weights & Biases run
wandb.finish()
