# %%
from chatbot_F1 import get_championship_winner, get_teams_in_year, get_driver_wins, get_driver_with_most_wins
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
import os
import pandas as pd 
from spellchecker import SpellChecker

# %%
f1_words = [
    'Formula', '1', 'F1', 'constructor', 'championship', 'pole', 'qualifying', 'race', 'driver', 'pitstop',
    'lap', 'sector', 'tyre', 'pit', 'team', 'engine', 'circuit', 'track', 'chassis', 'brake', 'pitlane', 'grid',
    'points', 'podium', 'winner', 'fastest', 'lap-time', 'team-mate', 'strategy', 'pit-crew', 'mechanic', 'team-principal',
    'season', 'constructor-standings', 'driver-standings', 'constructor-championship', 'world-champion', 'driver-champion',
    'red-bull', 'mercedes', 'ferrari', 'mclaren', 'renault', 'alpine', 'aston-martin', 'williams', 'haas', 'alphatauri',
    'alfa-romeo', 'jaguar', 'benetton', 'tyrrell', 'minardi', 'lotus', 'brabham', 'march', 'matra', 'sauber', 'brundle',
    'schumacher', 'hamilton', 'verstappen', 'leclerc', 'sainz', 'ricciardo', 'gasly', 'norris', 'albon', 'schumacher',
    'rosberg', 'raikkonen', 'vettel', 'webber', 'massa', 'hakkinen', 'hill', 'button', 'fittipaldi', 'prost', 'lauda',
    'senna', 'gilles-villeneuve', 'suzuka', 'monza', 'silverstone', 'spa', 'monaco', 'austria', 'hungary', 'belgium', 
    'france', 'germany', 'brazil', 'canada', 'italy', 'singapore', 'china', 'azerbaijan', 'russia', 'turkey', 'mexico',
    'saudi-arabia', 'emilia-romagna', 'portugal', 'dutch', 'bahrain', 'japan', 'united-states', 'abu-dhabi', 'australian',
    'french', 'spain', 'world-title', 'rookie', 'podium-finish', 'team-order',
    'drs', 'ferrari-boost', 'pit-window', 'fuel-load', 'tire-degradation', 'cold-tires', 'hot-tires', 'braking-zone',
    'track-limits', 'safety-car', 'virtual-safety-car', 'yellow-flag', 'red-flag', 'blue-flag', 'green-flag', 'racecraft',
    'lap-record', 'qualifying-position', 'front-row', 'backmarker', 'grid-position', 'overtake', 'undercut', 'overcut',
    'laps-completed', 'engine-mode', 'fuel-consumption', 'energy-recovery', 'brake-balance', 'team-radio', 'race-strategy',
    'pit-stop-strategy', 'tire-compound', 'wet-weather', 'dry-weather', 'mixed-conditions', 'intermediates', 'full-wet',
    'dry-tyres', 'hard-tyres', 'medium-tyres', 'soft-tyres', 'ultra-soft', 'super-soft', 'race-control', 'pit-in', 'pit-out',
    'track-position', 'push-lap', 'cool-down-lap', 'team-messages', 'fast-lap', 'short-lap', 'clutch', 'wheel-spin', 'driving-style',
    'aero-package', 'suspension', 'brake-ducts', 'front-wing', 'rear-wing', 'diffuser', 'airflow', 'engine-mapping', 'exhausts',
    'traction-control', 'ABS', 'carbon-fiber', 'weight-distribution', 'g-forces', 'chassis-tuning', 'pit-crew', 'driver-swap',
    'safety-cell', 'carbon-composite', 'drivers-briefing', 'pre-season-testing', 'race-weekend', 'practice-session',
    'free-practice', 'qualifying-session', 'pit-stop-strategy', 'race-schedule', 'constructor-standings', 'driver-standings',
    'grid-girls', 'fan-festival', 'motorsport', 'simulator', 'team-radio', 'virtual-race', 'esports',
    'pit-wall', 'racetrack', 'broadcast', 'team-rival', 'driver-fitness', 'track-circuit', 'track-record', 'double-points',
    'sprint-race', 'driving-test', 'safety-standard', 'yellow-flag', 'pit-exit', 'brake-checks', 'race-win', 'championship-lead',
    'tires', 'podium-position', 'race-lap', 'championship-points', 'fast-lap', 'qualification', 'track-record'
]


# %%
spell = SpellChecker()

spell.word_frequency.load_words(f1_words)

def correct_text(text):
    words = text.split()
    corrected_words = [spell.correction(word) if spell.correction(word) is not None else word for word in words]
    return ' '.join(corrected_words)

question = "Woh wno hte costructor chmpoinshp en 2016 ?"
corrected_question = correct_text(question)
print(corrected_question)

# %%
# Dataset folder path (general)
current_dir = os.getcwd() 
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

# %%
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

# %%
# Define query structure
class Query(BaseModel):
    question: str
    
# Function to predict intent
def predict_intent(question: str):
    encoding = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**encoding)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)  # Convert logits to probabilities
    # Get the predicted intent and confidence
    confidence, predicted_class = torch.max(probabilities, dim=-1)
    predicted_intent = str(predicted_class.item())  # Convert predicted class to string
    #predicted_label = torch.argmax(logits, dim=-1).item()
    print(logits)
    print(f"Confidence :  {confidence}")
    #return labels[predicted_intent]
    return {"intent": labels[int(predicted_intent)], "confidence": confidence.item()}


# %%
# Extract year from question (basic regex, can improve later)
def extract_year(question: str):
    match = re.search(r"\b(19\d{2}|20\d{2})\b", question)
    return int(match.group()) if match else None

# %%
def get_intent(question):
    corrected_question = question  # for now (correct_text(question))
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

    return {"intent": intent, "response": response, "corrected_question": corrected_question  }# Show corrected text

# %%
question1 = "who won the 2023 driver championship ?"
question2 = "who were the teams competing in the 2015 season ?"
question3 = "which car did mercedes use in 2020 ?"
question4 = "Who won the constructors championship in 2016 ?"
question = "Woh wno hte costructor chmpoinshp en 2016 ?"
question_corr = correct_text(question)
print(question_corr)
get_intent(question)


