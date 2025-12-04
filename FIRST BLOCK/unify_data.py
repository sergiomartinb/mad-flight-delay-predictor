# unify_data.py
import pandas as pd
import json
import glob
import os
import argparse

# Default configurations
CONFIGS = {
    "departures": {
        "input_folder": "stored_data",
        "output_file": "flights_data_raw.csv",
    },
    "arrivals": {
        "input_folder": "stored_data_arrivals", 
        "output_file": "arrivals_data_raw.csv",
    }
}

def flatten_flight_data(all_flights_list):
    """
    Extracts nested JSON data into a flat list of dictionaries.
    """
    flattened_list = []
    
    for flight in all_flights_list:
        flat_record = {
            # Top-level fields
            'status': flight.get('status'),
            'type': flight.get('type'),
            
            # Departure fields
            'dep_iata': flight.get('departure', {}).get('iataCode'),
            'dep_icao': flight.get('departure', {}).get('icaoCode'),
            'dep_terminal': flight.get('departure', {}).get('terminal'),
            'dep_gate': flight.get('departure', {}).get('gate'),
            'dep_delay': flight.get('departure', {}).get('delay'), 
            'dep_scheduled_time': flight.get('departure', {}).get('scheduledTime'),
            'dep_actual_time': flight.get('departure', {}).get('actualTime'),
            'dep_estimated_time': flight.get('departure', {}).get('estimatedTime'),
            'dep_actual_runway': flight.get('departure', {}).get('actualRunway'), 
            'dep_estimated_runway': flight.get('departure', {}).get('estimatedRunway'), 
            
            # Arrival fields
            'arr_iata': flight.get('arrival', {}).get('iataCode'),
            'arr_icao': flight.get('arrival', {}).get('icaoCode'),
            'arr_terminal': flight.get('arrival', {}).get('terminal'),
            'arr_gate': flight.get('arrival', {}).get('gate'),
            'arr_baggage_claim': flight.get('arrival', {}).get('baggage'), 
            'arr_delay': flight.get('arrival', {}).get('delay'), 
            'arr_scheduled_time': flight.get('arrival', {}).get('scheduledTime'),
            'arr_actual_time': flight.get('arrival', {}).get('actualTime'),
            'arr_estimated_time': flight.get('arrival', {}).get('estimatedTime'), 
            
            # Airline fields
            'airline_name': flight.get('airline', {}).get('name'),
            'airline_iata': flight.get('airline', {}).get('iataCode'),
            'airline_icao': flight.get('airline', {}).get('icaoCode'),
            
            # Flight fields
            'flight_number': flight.get('flight', {}).get('number'),
            'flight_iata': flight.get('flight', {}).get('iata'),
            'flight_icao': flight.get('flight', {}).get('icaoNumber'),
            
            # Aircraft fields
            'aircraft_reg': flight.get('aircraft', {}).get('registrationNumber'),
            'aircraft_model': flight.get('aircraft', {}).get('typeName'),
        }
        flattened_list.append(flat_record)
        
    return flattened_list

def unify_flights(json_folder, csv_file, verbose=True):
    """
    Reads all .json files from a folder, flattens them, and saves to CSV.
    """
    search_pattern = os.path.join(json_folder, "*.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        print(f"Error: No .json files found in folder '{json_folder}'.")
        return None

    print(f"Found {len(json_files)} JSON files in '{json_folder}'.")

    master_flight_list = []
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    master_flight_list.extend(file_data)
        except Exception as e:
            if verbose:
                print(f"Error reading {file_path}: {e}")

    if not master_flight_list:
        print("No flights were loaded. The DataFrame is empty.")
        return None

    print(f"Loaded {len(master_flight_list)} flight records. Flattening...")
    
    flattened_data = flatten_flight_data(master_flight_list)
    df_raw = pd.DataFrame(flattened_data)
    
    if verbose:
        cols_to_show = ['dep_scheduled_time', 'dep_delay', 'flight_number', 'arr_iata']
        cols_to_show = [col for col in cols_to_show if col in df_raw.columns]
        print(df_raw[cols_to_show].head())
        print(f"\nShape: {df_raw.shape}")
    
    df_raw.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"Saved to '{csv_file}'")
    
    return df_raw

def main():
    parser = argparse.ArgumentParser(description="Unify flight JSON data into CSV")
    parser.add_argument(
        "mode", 
        nargs="?",
        choices=["departures", "arrivals", "both"],
        default="both",
        help="Which data to process: departures, arrivals, or both (default: both)"
    )
    parser.add_argument("--input", "-i", help="Custom input folder")
    parser.add_argument("--output", "-o", help="Custom output file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    if args.mode == "both":
        modes = ["departures", "arrivals"]
    else:
        modes = [args.mode]
    
    for mode in modes:
        print(f"\n{'='*50}")
        print(f"Processing: {mode.upper()}")
        print('='*50)
        
        config = CONFIGS[mode]
        input_folder = args.input or config["input_folder"]
        output_file = args.output or config["output_file"]
        
        unify_flights(input_folder, output_file, verbose=not args.quiet)

# Main execution
if __name__ == "__main__":
    main()
