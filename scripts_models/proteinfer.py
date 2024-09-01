import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class ProteInfer(nn.Module):
    def __init__(self, num_labels):
        super(ProteInfer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        return self.classifier(pooled_output)

# Load the model (assuming you have a trained model)
num_labels = 1000  # Number of GO terms or EC numbers
model = ProteInfer(num_labels)
model.load_state_dict(torch.load('proteinfer_model.pth'))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_function(sequence):
    inputs = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
    
    predictions = torch.sigmoid(logits)
    return predictions.squeeze().tolist()

# Example usage
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

function_predictions = predict_function(sequence)
print(f"Function predictions: {function_predictions}")

# You would typically have a mapping of indices to GO terms or EC numbers
# to interpret these predictions
