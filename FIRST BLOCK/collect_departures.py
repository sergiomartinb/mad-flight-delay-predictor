# collect_departures.py
import json
import os
import time
from datetime import datetime, timedelta
from aviationedge_client import fetch_all_delays  
from config import AVIATIONEDGE_API_KEY 

# Set variables
AIRPORT = "MAD"
TYPE = "departure"
YEARS_OF_DATA = 1    # How many years back we want to go
DAYS_PER_WINDOW = 10  # Grab 10 days from each month
OUTPUT_FOLDER = "stored_data" # Folder to save the JSON files

def generate_time_windows(years_back=3, window_days=7):
    """Generates date ranges (start, end) for each month."""
    windows = []
    current_date = datetime.now()
    
    # We start from last month
    first_day_current_month = current_date.replace(day=1)
    last_day_last_month = first_day_current_month - timedelta(days=1)
    
    iter_date = last_day_last_month
    
    for _ in range(12 * years_back):
        # Take the first X days of the month
        date_end = iter_date.replace(day=window_days)
        date_start = iter_date.replace(day=1)
        
        windows.append({
            "start": date_start.strftime("%Y-%m-%d"),
            "end": date_end.strftime("%Y-%m-%d")
        })
        
        # Go back to the previous month
        last_day_prev_month = date_start - timedelta(days=1)
        iter_date = last_day_prev_month

    return windows

# Main execution
if __name__ == "__main__":
    
    if not AVIATIONEDGE_API_KEY:
        print("ERROR: Could not load AVIATIONEDGE_API_KEY from config.py")
        exit()
    else:
        print(f"AviationEdge Key loaded (...{AVIATIONEDGE_API_KEY[-4:]})")

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Folder '{OUTPUT_FOLDER}' created.")

    # Generate time windows
    windows = generate_time_windows(YEARS_OF_DATA, DAYS_PER_WINDOW)
    print(f"{len(windows)} time windows generated (e.g., {windows[0]['start']} to {windows[0]['end']}).")

    for i, window in enumerate(windows):
        
        output_file = f"{OUTPUT_FOLDER}/data_{AIRPORT}_{window['start']}_to_{window['end']}.json"
        
        # If we've already downloaded it, skip
        if os.path.exists(output_file):
            print(f"({i+1}/{len(windows)}) SKIPPED: {output_file} already exists")
            continue

        print(f"({i+1}/{len(windows)}) Downloading: {window['start']} to {window['end']}...")
        
        params = {
            "code": AIRPORT,
            "type": TYPE,
            "date_from": window["start"],
            "date_to": window["end"]
        }
        
        try:
            # Fetch data in the date range
            flights = fetch_all_delays(params)
            
            if flights:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(flights, f, ensure_ascii=False, indent=2)
                print(f"    -> Success! Saved {len(flights)} flights to {output_file}")
            else:
                print("    -> Call succeeded, but no flights were returned for this range.")
            
            time.sleep(1.5) # Wait 1.5 seconds between calls

        except Exception as e:
            print(f"    -> ERROR on this window: {e}")
            time.sleep(5) 
            
    print(f"All files are saved in the '{OUTPUT_FOLDER}' folder.")