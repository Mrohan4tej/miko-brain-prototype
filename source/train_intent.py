import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torch.cuda.amp import autocast, GradScaler # KEY: Optimization Library

# 1. Custom Dataset
class IntentDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = self.tokenizer(item['text'], padding='max_length', max_length=32, truncation=True, return_tensors="pt")
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

# 2. Custom Model Architecture (Not just AutoModel)
class MikoIntentModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MikoIntentModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Optimization: Freeze lower layers to speed up training
        for param in self.bert.transformer.layer[:2].parameters():
            param.requires_grad = False
            
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes) # Custom Head

    def forward(self, input_ids, mask):
        output = self.bert(input_ids=input_ids, attention_mask=mask)
        cls_rep = output.last_hidden_state[:, 0, :] # Take [CLS] token
        x = self.dropout(cls_rep)
        return self.classifier(x)

# 3. The Training Loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = IntentDataset('data/intent_data.json')
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = MikoIntentModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler() # KEY: For Mixed Precision

    model.train()
    for epoch in range(3): # Quick training
        total_loss = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # KEY: Mixed Precision Context Manager
            # Runs operations in float16 where safe, float32 where needed
            with autocast(): 
                outputs = model(input_ids, mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)

            # Scale loss to prevent underflow in float16
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader)}")

    # Save
    torch.save(model.state_dict(), "models/intent_model.pt")
    print("Model saved to models/intent_model.pt")

if __name__ == "__main__":
    train()