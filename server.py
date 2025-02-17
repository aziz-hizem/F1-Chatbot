from chatbot_F1 import get_championship_winner, get_teams_in_year, get_driver_wins, get_driver_with_most_wins
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
import os
import pandas as pd 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from spellchecker import SpellChecker


# function to correct typos
#def correct_text_SymSpell(text):
    #suggestions = sym_spell.lookup(text, Verbosity.CLOSEST, max_edit_distance=2)
    #return suggestions[0].term if suggestions else text

spell = SpellChecker()
#  fuunction to correct typos
def correct_text(text):
    words = text.split()
    corrected_words = [spell.correction(word) if spell.correction(word) is not None else word for word in words]
    return ' '.join(corrected_words)

app = FastAPI()

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from React frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods, including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

# Explicitly handle OPTIONS requests for the specific route
@app.options("/predict_intent/")
async def handle_options():
    return JSONResponse(status_code=200, content={})

# Define query structure
class Query(BaseModel):
    question: str

# Dataset folder path (general)
current_dir = os.path.dirname (os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "F1 Datasets")

# List of CSV files in the dataset
csv_files = [
    "results.csv", "seasons.csv", "sprint_results.csv", "status.csv", "constructor_standings.csv", 
    "constructors.csv", "driver_standings.csv", "drivers.csv", "lap_times.csv", "pit_stops.csv", 
    "qualifying.csv", "races.csv", "circuits.csv", "constructor_results.csv"
]

def load_and_explore () :
    dataframes = {}
    for file in csv_files : 
        file_path = os.path.join(dataset_path, file)
        df = pd.read_csv(file_path)
        dataframes[file] = df # Storing DataFrame for later use
    return dataframes

# Load Datasets 
datasets = load_and_explore()

# Load trained model and tokenizer
model_path = "intent_classifier_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Mapping labels back to intents
labels = [
    'get_teams_in_year',
    'get_driver_wins',
    'get_championship_winner',
    'get_driver_with_most_wins',
    'get_constructors_championship',
    'get_race_winners_in_year',
    'get_championship_runner_up',
    'get_drivers_with_multiple_titles',
    'get_driver_with_most_poles',
    'get_car_in_season']
#10 Labels for now

# Function to predict intent
def predict_intent(question: str):
    encoding = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**encoding)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=-1).item()
    return labels[predicted_label]


# Extract year from question (basic regex, can improve later)
def extract_year(question: str):
    match = re.search(r"\b(19\d{2}|20\d{2})\b", question)
    return int(match.group()) if match else None

# API endpoint for processing user queries
@app.post("/predict_intent/")
async def get_intent(query: Query):
    corrected_question = correct_text(query.question)
    intent = predict_intent(corrected_question)
    year = extract_year(corrected_question)
    response = "Sorry, I couldn't find an answer."

    if intent == "get_championship_winner" and year:
        response = get_championship_winner(year, datasets['driver_standings.csv'], datasets['races.csv'], datasets['drivers.csv'])
    elif intent == "get_teams_in_year" and year:
        response = get_teams_in_year(year, datasets['results.csv'], datasets['races.csv'], datasets['constructors.csv'])
    elif intent == "get_driver_wins":
        driver_name = corrected_question.replace("How many wins does", "").replace("have?", "").strip()
        response = get_driver_wins(driver_name, datasets['results.csv'], datasets['drivers.csv'])
    elif intent == "get_driver_with_most_wins":
        response = get_driver_with_most_wins(datasets['results.csv'], datasets['drivers.csv'])
    else :
        response = str(intent)

    print(corrected_question)
    print(intent)
    return {"intent": intent, "response": response, "corrected_question": corrected_question  }# Show corrected text

# Run FastAPI with: uvicorn server:app --reload
