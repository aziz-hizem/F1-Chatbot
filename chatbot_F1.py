import os
import pandas as pd

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
        #print(f"\n==={file} ===")
        #print(df.head()) 
        #print("Columns:", df.columns.tolist())
    return dataframes

# Load Datasets 
datasets = load_and_explore()


def get_championship_winner (year, driver_standings, races, drivers):

    if year < 1950 :
        return("There were no F1 races before 1950.")
    elif year > 2024 :
        return(f"The {year} F1 season has not been held yet !")

    # RacesIds for given year
    races_ids = races[races['year'] == year]['raceId']

    # driver standings for the races in the given year
    standings = driver_standings[driver_standings['raceId'].isin(races_ids)]

    # find driver with most points 
    year_points = standings.groupby('driverId')['points'].max()
    champion_id = year_points.idxmax()
    champion_points = year_points.max()

    # Get the driver's name
    champion_name = drivers[drivers['driverId'] == champion_id].iloc[0]['forename'] + " " + drivers[drivers['driverId'] == champion_id].iloc[0]['surname']

    # Driver name and points of the year
    champion = [champion_name, champion_points]
    

    return(f"{champion[0]} won the {year} driver championship with {int(champion[1]) if int(champion[1]) == champion[1] else champion[1]} points.")

def get_teams_in_year (year, results, races, constructors):

    # !!! This works starting from 1958 because The Constructors Championship was not awarded until then !!!

    # Get races ids for given year
    races_ids = races[races['year'] == year]['raceId']

    # Filter results for the races that year
    results_year = results[results['raceId'].isin(races_ids)]

    # Get the constructorIds for the teams in given year
    teams_ids = results_year['constructorId'].unique()

      # Get the teams names
    teams_names = constructors[constructors['constructorId'].isin(teams_ids)]

    teams = teams_names['name']

    return(teams)

def get_driver_wins(driver, results, drivers):
   
    # Find the driverId for the race results where position = 1 (win)
    winning_drivers = results[results['positionOrder'] == 1]['driverId']

    # Count the number of wins for each driver
    wins_count = winning_drivers.value_counts()

    # Extract driver's name
    name_parts = driver.split()
    driver_forename = name_parts[0]
    driver_surname = " ".join(name_parts[1:])

    driver_id = drivers[ (drivers['forename'] == driver_forename) & (drivers['surname'] == driver_surname)]['driverId']

    # Get the number of wins for the driver
    if not driver_id.empty:  # Check if driver_id exists
        return wins_count.get(driver_id.iloc[0], 0)  # Return the number of wins or 0 if not found
    else:
        return "Driver Not Found."

def get_driver_with_most_wins(results, drivers):
   
    # Find the driverId for the race results where position = 1 (win)
    winning_drivers = results[results['positionOrder'] == 1]['driverId']

    # Count the number of wins for each driver and gets sorted directly unless we set sort = False
    wins_count = winning_drivers.value_counts()

    # Find driver's name and wins
    driver_id = wins_count.index[0]
    driver_wins = wins_count.values[0]

    driver_name = drivers[drivers['driverId'] == driver_id].iloc[0]['forename'] + " " + drivers[drivers['driverId'] == driver_id].iloc[0]['surname']

    driver = [driver_name, driver_wins]
    return (f"{driver[0]} is the driver with most wins having {driver[1]} victories since his debut in F1 ! Impressive Right ?")

def get_constructors_championship (year ) :
    return("Super MAX")
    



