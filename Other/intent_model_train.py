import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset


# Loading the dataset
df = pd.read_csv("intent_dataset.csv")

# Encode Lables
labels = {label : idx for idx, label in enumerate(df["intent"].unique())}
df["label"] = df["intent"].map(labels)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["question"].tolist(), df["label"].tolist(), test_size = 0.2, random_state = 42
)

# Tokenizer 
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization 
def tokenize_function(texts) : 
    return tokenizer (texts, padding = True, truncation = True, max_length = 128, return_tensor="pt")

train_encodings = tokenize_function(train_texts)
val_encondings = tokenize_function(val_texts)

# Custom dataset class
class IntentDataset (Dataset) :
    def __init__ (self, encodings, labels):
        self.encodings = encodings 
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx) :
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = IntentDataset(train_encodings, train_labels)
val_dataset = IntentDataset(val_encondings, val_labels)

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels = len(labels))

# training arguments
training_args = TrainingArguments(
    output_dir ="./results",
    eval_strategy = "epoch",
    save_strategy= "epoch", 
    per_device_train_batch_size = 8,
    per_device_eval_batch_size= 8,
    num_train_epochs= 3,
    weight_decay= 0.01,
)


# Trainer
trainer = Trainer(
    model =model, 
    args = training_args ,
    train_dataset= train_dataset,
    eval_dataset= val_dataset,
)

# Train Model
trainer.train()

# Save Model
model.save_pretrained("intent_classifier_model")
tokenizer.save_pretrained("intent_classifier_model")

print("Model training complete! The model has been saved.")

# Evaluate the model
results = trainer.evaluate()

# Print the evaluation results
print(f"Validation Results: {results}")

# Function to predict intent for a new question
def predict_intent(question):
    encoding = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**encoding)
    logits = outputs.logits
    print(logits)
    predicted_label = torch.argmax(logits, dim=-1).item()
    intent = list(labels.keys())[predicted_label]
    return intent

# Test prediction
new_question = "Who won the F1 championship in 2015?"
predicted_intent = predict_intent(new_question)
print(f"Predicted Intent: {predicted_intent}")
