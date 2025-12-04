# collect_arrivals.py
import json
import os
import time
from datetime import datetime, timedelta
from aviationedge_client import fetch_all_delays 
from config import AVIATIONEDGE_API_KEY 

# Set variables
AIRPORT = "MAD"
TYPE = "arrival"     
YEARS_OF_DATA = 1     
DAYS_PER_WINDOW = 10  
OUTPUT_FOLDER = "stored_data_arrivals" 

def generate_time_windows(years_back=1, window_days=10):
    """Generates date ranges (start, end) for each month."""
    windows = []
    current_date = datetime.now()
    
    # Start from last month
    first_day_current_month = current_date.replace(day=1)
    last_day_last_month = first_day_current_month - timedelta(days=1)
    
    iter_date = last_day_last_month
    
    for _ in range(12 * years_back):
        date_end = iter_date.replace(day=window_days)
        date_start = iter_date.replace(day=1)
        
        windows.append({
            "start": date_start.strftime("%Y-%m-%d"),
            "end": date_end.strftime("%Y-%m-%d")
        })
        
        last_day_prev_month = date_start - timedelta(days=1)
        iter_date = last_day_prev_month

    return windows

# Main execution
if __name__ == "__main__":
    
    if not AVIATIONEDGE_API_KEY:
        exit()

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Folder '{OUTPUT_FOLDER}' created.")

    windows = generate_time_windows(YEARS_OF_DATA, DAYS_PER_WINDOW)
    print(f"{len(windows)} time windows generated for ARRIVALS.")

    for i, window in enumerate(windows):
        output_file = f"{OUTPUT_FOLDER}/arrival_data_{AIRPORT}_{window['start']}_to_{window['end']}.json"
        
        if os.path.exists(output_file):
            print(f"({i+1}/{len(windows)}) SKIPPED: {output_file} already exists")
            continue

        print(f"({i+1}/{len(windows)}) Downloading Arrivals: {window['start']} to {window['end']}...")
        
        params = {
            "code": AIRPORT,
            "type": TYPE, # 'arrival'
            "date_from": window["start"],
            "date_to": window["end"]
        }
        
        try:
            # Fetch arrivals data in the date range
            flights = fetch_all_delays(params)
            
            if flights:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(flights, f, ensure_ascii=False, indent=2)
                print(f"    -> Saved {len(flights)} arrivals.")
            else:
                print("    -> No arrivals returned for this range.")
            
            time.sleep(1.5) 

        except Exception as e:
            print(f"    -> ERROR: {e}")
            time.sleep(5)