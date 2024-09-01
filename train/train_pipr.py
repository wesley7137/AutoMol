import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class PIPR(nn.Module):
    def __init__(self):
        super(PIPR, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768 * 2, 1)
        
    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        output1 = self.bert(input_ids1, attention_mask=attention_mask1)[1]
        output2 = self.bert(input_ids2, attention_mask=attention_mask2)[1]
        combined = torch.cat((output1, output2), dim=1)
        return torch.sigmoid(self.classifier(combined))

# Load the model (assuming you have a trained model)
model = PIPR()
model.load_state_dict(torch.load('pipr_model.pth'))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_interaction(seq1, seq2):
    inputs1 = tokenizer(seq1, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs2 = tokenizer(seq2, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        prediction = model(inputs1['input_ids'], inputs1['attention_mask'],
                           inputs2['input_ids'], inputs2['attention_mask'])
    
    return prediction.item()

# Example usage
seq1 = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
seq2 = "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"

interaction_probability = predict_interaction(seq1, seq2)
print(f"Interaction probability: {interaction_probability}")
