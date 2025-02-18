# %%
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import EarlyStoppingCallback


# %%
# Loading the dataset
df = pd.read_csv("intent_dataset.csv")

# Encode Lables
labels = {label : idx for idx, label in enumerate(df["intent"].unique())}
df["label"] = df["intent"].map(labels)



# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["question"].tolist(), df["label"].tolist(), test_size = 0.2, random_state = 42
)
print(f"Training examples: {len(train_texts)}")
print(f"Validation examples: {len(val_texts)}")
#print(df.count(df['intent'] == "get_teams_in_year"))

intent_rep = df['intent'].value_counts()
print(intent_rep)


# %%
# Tokenizer 
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization 
def tokenize_function(texts) : 
    return tokenizer (texts, padding = True, truncation = True, max_length = 128, return_tensor="pt")

train_encodings = tokenize_function(train_texts)
val_encondings = tokenize_function(val_texts)

# %%
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

# %%
# Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels = len(labels))


# %%
# training arguments
training_args = TrainingArguments(
    output_dir ="./results",
    eval_strategy = "epoch",
    save_strategy= "epoch", 
    per_device_train_batch_size = 4,
    per_device_eval_batch_size= 4,
    num_train_epochs= 20,
    learning_rate= 5e-5,
    weight_decay= 0.02,
    warmup_steps=100,
    load_best_model_at_end=True,
)


# Trainer
trainer = Trainer(
    model =model, 
    args = training_args ,
    train_dataset= train_dataset,
    eval_dataset= val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
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

# %%
# Function to predict intent for a new question
def predict_intent(question):
    encoding = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**encoding)
    logits = outputs.logits
    print(logits)
    predicted_label = torch.argmax(logits, dim=-1).item()
    intent = list(labels.keys())[predicted_label]
    print(labels.keys())
    return intent

# Test prediction
new_question = "Who is the  wi most wins ?"
predicted_intent = predict_intent(new_question)
print(f"Predicted Intent: {predicted_intent}")


# %%
# Generate predictions for the validation set
predictions = trainer.predict(val_dataset)

# Get the predicted labels
predicted_labels = np.argmax(predictions.predictions, axis=-1)

# Get the true labels
true_labels = predictions.label_ids

# Map numerical labels to intent names
y_true = [list(labels.keys())[label] for label in true_labels] # True intents
y_pred = [list(labels.keys())[label] for label in predicted_labels] # Predicted intents

# %%
# Metrics
# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Precision, Recall, F1-Score (Per-Class Metrics)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')  # Use 'micro' or 'macro' for other types of averaging
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report (includes precision, recall, F1-score per class)
print('Classification Report:')
print(classification_report(y_true, y_pred))



