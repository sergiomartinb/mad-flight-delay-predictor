# get_weather_data.py
import pandas as pd
import requests
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv("WEATHER_API_KEY")
INPUT_FILE = "flights_data_raw.csv"
OUTPUT_FILE = "weather_data_raw.csv"
LOCATION = "LEMD"  # ICAO code for Madrid-Barajas Airport
BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"

def get_previously_fetched_dates(output_file):
    """Checks the existing output CSV and returns a list of dates already downloaded."""
    if not os.path.exists(output_file):
        print("No existing weather file found. Starting fresh download.")
        return []
    
    try:
        df_weather = pd.read_csv(output_file)
        if 'date' in df_weather.columns:
            fetched_dates = df_weather['date'].unique()
            print(f"Found existing weather file. Already have {len(fetched_dates)} days of data.")
            return fetched_dates
        else:
            print("Weather file found, but 'date' column is missing. Starting fresh.")
            return []
    except pd.errors.EmptyDataError:
        print("Weather file is empty. Starting fresh download.")
        return []
    except Exception as e:
        print(f"Error reading existing weather file: {e}. Starting fresh.")
        return []

def get_unique_dates(input_file):
    """Reads the flight data CSV and returns a list of unique dates (YYYY-MM-DD)."""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return []

    df_flights = pd.read_csv(input_file)
    df_flights['datetime_utc'] = pd.to_datetime(df_flights['dep_scheduled_time'], errors='coerce')
    df_flights = df_flights.dropna(subset=['datetime_utc'])
    unique_dates = df_flights['datetime_utc'].dt.strftime('%Y-%m-%d').unique()
    
    print(f"Found {len(unique_dates)} total unique dates in flight data.")
    return unique_dates

def fetch_weather_data(dates_to_fetch, api_key):
    """Loops through dates, calls the API, and flattens hourly data."""
    all_hourly_data = []

    for i, date_str in enumerate(dates_to_fetch):
        # Stop condition: 24 records/day. Stop before hitting 1000 records.
        # We stop at 40 days to be safe.
        if i >= 41: 
            print("\nReached safe daily limit (approx 1000 records).")
            break
            
        print(f"({i+1}/{len(dates_to_fetch)}) Fetching weather for: {date_str}...")
        url = f"{BASE_URL}{LOCATION}/{date_str}/{date_str}"
        params = {
            'unitGroup': 'metric',
            'key': api_key,
            'include': 'hours',
            'contentType': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"  -> Error {response.status_code} for date {date_str}: {response.text}")
                # If we hit the limit, stop this run
                if "limit" in response.text.lower():
                    print("API limit reached for today. Stopping.")
                    break
                continue
                
            data = response.json()
            daily_data = data['days'][0]
            
            # Flatten hourly data
            for hour_data in daily_data['hours']:
                flat_record = {
                    'date': date_str,
                    'datetime': f"{date_str}T{hour_data['datetime']}",
                    'temp': hour_data.get('temp'),
                    'precip': hour_data.get('precip'),
                    'windspeed': hour_data.get('windspeed'),
                    'winddir': hour_data.get('winddir'),
                    'visibility': hour_data.get('visibility'),
                    'cloudcover': hour_data.get('cloudcover'),
                    'conditions': hour_data.get('conditions')
                }
                all_hourly_data.append(flat_record)
            
            time.sleep(0.5)

        except requests.RequestException as e:
            print(f"  -> Network Error for date {date_str}: {e}")
            break # Stop on network error
            
    return all_hourly_data

# Main execution
if __name__ == "__main__":
    
    # Fetch weather data
    all_required_dates = get_unique_dates(INPUT_FILE)

    if all_required_dates.size > 0:
        
        # Get dates we already have
        fetched_dates = get_previously_fetched_dates(OUTPUT_FILE)
        
        # Determine which dates are left to fetch
        dates_to_fetch = [d for d in all_required_dates if d not in set(fetched_dates)]
        
        if not dates_to_fetch:
            print("\n--- All weather data has already been downloaded! ---")
            print(f"File '{OUTPUT_FILE}' is complete.")
            exit()
            
        print(f"Total dates required: {len(all_required_dates)}")
        print(f"Already fetched: {len(fetched_dates)}")
        print(f"Dates remaining to fetch this run: {len(dates_to_fetch)}")

        # Fetch the new data
        hourly_weather = fetch_weather_data(dates_to_fetch, API_KEY)
        
        if hourly_weather:
            # Save the results to the CSV
            df_weather = pd.DataFrame(hourly_weather)
            
            # Check if file exists to decide whether to write header
            file_exists = os.path.exists(OUTPUT_FILE)
            
            print(f"\nAppending {len(hourly_weather)} new hourly records to '{OUTPUT_FILE}'...")
            df_weather.to_csv(
                OUTPUT_FILE, 
                mode='a',  
                header=(not file_exists), 
                index=False, 
                encoding='utf-8-sig'
            )
            
            print(f"Success! Data appended to '{OUTPUT_FILE}'.")
            
        else:
            print("No new weather data was fetched in this run.")
    
    else:
        print("No unique dates were found in the input file. Cannot fetch weather.")