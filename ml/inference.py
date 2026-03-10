import torch
from transformers import BertTokenizer, BertForSequenceClassification

LABEL_MAP = {'Positive': 1, 'Negative': 0}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
MAX_LENGTH = 128

class SentimentModel:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)

        label_id = predicted_class.item()
        label = ID2LABEL[label_id]

        return {
            "label": label,
            "confidence": confidence.item()
        }