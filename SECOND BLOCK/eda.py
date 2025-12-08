import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
from pathlib import Path

# Avoid warnings
warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
FIRST_BLOCK = SCRIPT_DIR.parent / "FIRST BLOCK"

# Add FIRST BLOCK to path so we can import unify_data
if str(FIRST_BLOCK) not in sys.path:
    sys.path.insert(0, str(FIRST_BLOCK))

def ensure_raw_data_exists():
    """
    Check if raw CSV files exist, if not run unify_data.py to generate them.
    """
    data_folder = FIRST_BLOCK / "data"
    flights_csv = data_folder / "flights_data_raw.csv"
    arrivals_csv = data_folder / "arrivals_complete_data_raw.csv"
    
    missing = []
    if not flights_csv.exists():
        missing.append("flights_data_raw.csv")
    if not arrivals_csv.exists():
        missing.append("arrivals_complete_data_raw.csv")
    
    if missing:
        print(f"Missing files: {missing}")
        print("Attempting to generate from JSON data using unify_data.py...")
        
        try:
            # Change to FIRST BLOCK directory for unify_data to find the folders
            original_dir = os.getcwd()
            os.chdir(FIRST_BLOCK)
            
            from unify_data import unify_flights, CONFIGS
            
            if "flights_data_raw.csv" in missing:
                config = CONFIGS["departures"]
                unify_flights(config["input_folder"], config["output_file"])
            
            if "arrivals_data_raw.csv" in missing:
                config = CONFIGS["arrivals"]
                unify_flights(config["input_folder"], config["output_file"])
            
            os.chdir(original_dir)
            print("Raw data files generated successfully.")
            
        except Exception as e:
            os.chdir(original_dir)
            print(f"Could not auto-generate data: {e}")
            print("Please run unify_data.py manually in FIRST BLOCK/scripts/")


def load_and_clean_data(filepath):
    """
    Loads data, handles codeshare duplicates, and parses dates.
    """
    df = pd.read_csv(filepath)
    original_count = len(df)

    # If aircraft_reg is missing (rare), we fall back to Time + Destination
    df['aircraft_reg'] = df['aircraft_reg'].fillna('UNKNOWN_REG')

    # We want to keep the 'Operating Carrier' if possible.
    df.sort_values(by=['dep_scheduled_time', 'aircraft_reg', 'airline_name'], inplace=True)

    # Define the subset of columns that make up a "Duplicate Physical Flight"
    subset_cols = ['dep_scheduled_time', 'aircraft_reg', 'arr_iata']

    # For this EDA, keeping the first entry is sufficient to remove the statistical bias.
    df_unique = df.drop_duplicates(subset=subset_cols, keep='first')

    # Restore NaN for aircraft_reg if we created the placeholder
    df_unique['aircraft_reg'] = df_unique['aircraft_reg'].replace('UNKNOWN_REG', np.nan)

    print(f"Rows loaded: {original_count}")
    print(f"Rows after removing codeshare duplicates: {len(df_unique)}")
    print(f"Duplicate/Codeshare rows removed: {original_count - len(df_unique)}")

    # Date Normalization
    date_cols = ['dep_scheduled_time', 'dep_actual_time']
    for col in date_cols:
        # Coerce errors 
        df_unique[col] = pd.to_datetime(df_unique[col], errors='coerce')
        # Normalize
        df_unique[col] = df_unique[col].dt.floor('min')

    # Feature Engineering for EDA
    df_unique['dep_hour'] = df_unique['dep_scheduled_time'].dt.hour
    df_unique['dep_day_of_week'] = df_unique['dep_scheduled_time'].dt.day_name()
    df_unique['dep_month'] = df_unique['dep_scheduled_time'].dt.month_name()

    # Drop rows where target (dep_delay) is null (cancelled flights or data errors)
    df_clean = df_unique.dropna(subset=['dep_delay'])

    print(f"Rows available for Delay Analysis (non-null target): {len(df_clean)}")

    return df_clean

def analyze_target_variable(df):
    """
    Detailed analysis of 'dep_delay'.
    """
    print("\n--- 2. TARGET VARIABLE ANALYSIS (dep_delay) ---")

    # Basic Statistics
    stats = df['dep_delay'].describe()
    skewness = df['dep_delay'].skew()
    print(stats)
    print(f"Skewness: {skewness:.2f} (High skew indicates outliers)")

    # If we see an unrealistic number of delays > 24 hours (1440 mins), it's a flag
    extreme_delays = df[df['dep_delay'] > 1000]
    if not extreme_delays.empty:
        print(f"WARNING: Found {len(extreme_delays)} flights with > 1000 min delay. Verify units.")

    # Delay >= 25 minutes is "Late"
    df['is_delayed_25'] = df['dep_delay'] >= 25
    class_balance = df['is_delayed_25'].value_counts(normalize=True) * 100
    print("\nClass Balance (Threshold >= 25 mins):")
    print(class_balance)

    # Visualization
    plt.figure(figsize=(16, 5))

    # Standard Distribution 
    plt.subplot(1, 3, 1)
    upper_limit = df['dep_delay'].quantile(0.95)
    sns.histplot(df['dep_delay'], bins=50, kde=True)
    plt.xlim(0, max(upper_limit, 60))
    plt.title('Linear Distribution (Clipped at 95%)')
    plt.xlabel('Delay (Minutes)')

    # Log-Transformed Distribution
    plt.subplot(1, 3, 2)
    log_delay = np.log1p(df[df['dep_delay'] > 0]['dep_delay'])
    sns.histplot(log_delay, bins=50, kde=True, color='orange')
    plt.title('Log-Transformed Distribution')
    plt.xlabel('Log(Delay + 1)')

    # Boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(x=df['dep_delay'], color='lightblue')
    plt.title('Boxplot (Outliers Visible)')
    plt.xlabel('Delay (Minutes)')

    plt.tight_layout()
    plt.show()

def analyze_correlations_and_categories(df):
    """
    Analyzes relationships between delay and other features.
    Prepares insights for Graph Theory (Routes).
    """

    # Delay by Time of Day
    plt.figure(figsize=(12, 5))
    hourly_delay = df.groupby('dep_hour')['dep_delay'].mean()
    sns.barplot(x=hourly_delay.index, y=hourly_delay.values, palette="viridis")
    plt.title('Average Delay by Hour of Day')
    plt.ylabel('Avg Delay (min)')
    plt.xlabel('Hour (UTC)')
    plt.show()

    # Delay by Airline
    top_airlines = df['airline_iata'].value_counts().nlargest(15).index
    df_top_airlines = df[df['airline_iata'].isin(top_airlines)]

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='airline_iata', y='dep_delay', data=df_top_airlines, showfliers=False)
    plt.title('Delay Distribution by Top 15 Airlines (Outliers Hidden)')
    plt.xticks(rotation=45)
    plt.show()

    # Route Importance (MAD -> Destination)
    # We calculate "Edge Weights" based on Flight Volume and Avg Delay
    route_stats = df.groupby('arr_iata').agg(
        flight_count=('dep_delay', 'count'),
        avg_delay=('dep_delay', 'mean')
    ).sort_values(by='flight_count', ascending=False).head(15)

    print("\nTop 15 Destinations (Graph Nodes) by Volume & Avg Delay:")
    print(route_stats)

    # Visualizing the "Star Graph" connections
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=route_stats, x='flight_count', y='avg_delay', size='flight_count', sizes=(100, 1000))
    for line in range(0, route_stats.shape[0]):
        plt.text(route_stats.flight_count.iloc[line]+0.2,
                 route_stats.avg_delay.iloc[line],
                 route_stats.index[line],
                 horizontalalignment='left', size='medium', color='black')
    plt.title('Destination Analysis: Volume vs. Delay Risk')
    plt.xlabel('Number of Flights (Edge Weight)')
    plt.ylabel('Average Delay (Minutes)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def analyze_temporal_patterns(df):
    """
    Analyzes temporal patterns using Heatmaps.
    REPLACES: Basic bar charts with 2D Heatmaps.
    """

    # Order days correctly
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                  'Saturday', 'Sunday']

    # Day vs Hour, values = Mean Delay
    heatmap_data = df.groupby(['dep_day_of_week', 'dep_hour'])['dep_delay'].mean().reset_index()
    heatmap_data['dep_day_of_week'] = pd.Categorical(heatmap_data['dep_day_of_week'],
                                                     categories=days_order,
                                                     ordered=True)

    heatmap_pivot = heatmap_data.pivot(index='dep_day_of_week',
                                       columns='dep_hour',
                                       values='dep_delay')

    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_pivot, cmap="coolwarm", annot=False, cbar_kws={'label': 'Avg Delay (min)'})
    plt.title('Heatmap: Average Delay by Day and Hour')
    plt.xlabel('Hour of Day (UTC)')
    plt.ylabel('Day of Week')
    plt.show()

def missing_values_analysis(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found in relevant columns.")
    else:
        print(missing)
        print("\nInsight: 'arr_delay' missingness in this file is expected if the flight")
        print("hasn't landed or data wasn't merged yet. 'dep_delay' missingness")
        print("usually implies cancelled flights.")

# Weather analysis
def load_and_clean_weather(filepath):
    """
    Loads weather data, normalizes dates, and checks for consistency.
    """
    try:
        df_weather = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None

    # Date Normalization
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'], errors='coerce')
    df_weather['datetime'] = df_weather['datetime'].dt.floor('min')

    # Check for duplicates 
    if df_weather.duplicated(subset=['datetime']).any():
        print("Warning: Duplicate weather entries found. Keeping first occurrence.")
        df_weather.drop_duplicates(subset=['datetime'], keep='first', inplace=True)

    print(f"Weather records loaded: {len(df_weather)}")
    return df_weather

def analyze_weather_variables(df_weather):
    """
    EDA specifically for the weather dataset before merging.
    """
    
    # General Statistics
    print(df_weather[['temp', 'windspeed', 'visibility', 'precip']].describe())

    # Visualizations
    plt.figure(figsize=(14, 5))

    # Temperature Distribution
    plt.subplot(1, 3, 1)
    sns.histplot(df_weather['temp'], bins=30, kde=True, color='salmon')
    plt.title('Temperature Distribution (°C)')

    # Windspeed Distribution
    plt.subplot(1, 3, 2)
    sns.histplot(df_weather['windspeed'], bins=30, kde=True, color='skyblue')
    plt.title('Wind Speed Distribution (km/h)')

    # Conditions Count
    plt.subplot(1, 3, 3)
    # Take top 10 conditions to avoid clutter
    top_conditions = df_weather['conditions'].value_counts().nlargest(10).index
    sns.countplot(y='conditions', data=df_weather[df_weather['conditions'].isin(top_conditions)],
                  order=top_conditions, palette='viridis')
    plt.title('Top 10 Weather Conditions')

    plt.tight_layout()
    plt.show()

def merge_flight_weather_data(df_flights, df_weather):
    """
    Joins flight data with weather data.
    We must join by flooring the flight time to the nearest hour.
    """

    # Create temporary keys for joining
    df_flights['join_key_hour'] = df_flights['dep_scheduled_time'].dt.floor('H')
    df_weather['join_key_hour'] = df_weather['datetime'].dt.floor('H')

    # Perform Left Join
    df_merged = pd.merge(
        df_flights,
        df_weather,
        on='join_key_hour',
        how='left',
        suffixes=('', '_weather')
    )

    # Drop the temporary join key and redundant datetime column from weather
    df_merged.drop(columns=['join_key_hour', 'datetime'], inplace=True)

    # Check for missing weather data after join
    missing_weather = df_merged['temp'].isnull().sum()
    if missing_weather > 0:
        print(f"Warning: {missing_weather} flights could not be matched with weather data.")
        df_merged['conditions'] = df_merged['conditions'].fillna('Unknown')
        df_merged['precip'] = df_merged['precip'].fillna(0)
        df_merged['windspeed'] = df_merged['windspeed'].fillna(0)
        df_merged['visibility'] = df_merged['visibility'].fillna(df_merged['visibility'].mean())

    print(f"Merged dataset shape: {df_merged.shape}")
    return df_merged

def analyze_weather_impact(df):
    """
    Analyzes the correlation between weather features and flight delays.
    """
    print("\n--- 5. WEATHER IMPACT ON DELAYS ---")

    # Correlation Matrix
    weather_cols = ['temp', 'precip', 'windspeed', 'visibility', 'cloudcover', 'dep_delay']
    # Ensure columns exist
    valid_cols = [c for c in weather_cols if c in df.columns]

    corr_matrix = df[valid_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation: Weather vs Departure Delay')
    plt.show()

    # Conditions vs Delay
    # We group rare conditions into "Other" to make the plot readable
    top_conds = df['conditions'].value_counts().nlargest(8).index
    df['cond_grouped'] = df['conditions'].apply(lambda x: x if x in top_conds else 'Other')

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cond_grouped', y='dep_delay', data=df, showfliers=False, palette="Set2")
    plt.title('Impact of Weather Conditions on Delay (Outliers Hidden)')
    plt.xticks(rotation=45)
    plt.xlabel('Weather Condition')
    plt.ylabel('Departure Delay (min)')
    plt.show()

    # Wind Speed vs Delay (Scatter)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='windspeed', y='dep_delay', data=df, alpha=0.3)
    plt.title('Wind Speed vs. Departure Delay')
    plt.xlabel('Wind Speed (km/h)')
    plt.ylabel('Delay (min)')
    plt.show()


def load_and_clean_arrivals(filepath):
    """
    Loads arrival data, handles codeshare duplicates, and prepares for merging.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None

    original_count = len(df)

    # Similar to departures, we must identify the unique physical landing.
    # Key: Scheduled Arrival Time + Tail Number
    
    # We can't have nulls for the time though
    df = df.dropna(subset=['arr_actual_time'])

    # Handle Codeshares
    # The unique physical event is a landing at a specific gate at a specific time
    df.sort_values(by=['arr_scheduled_time', 'airline_iata'],
                   inplace=True)

    # Deduplicate
    subset_cols = ['arr_scheduled_time', 'dep_iata']
    df_unique = df.drop_duplicates(subset=subset_cols, keep='first')

    print(f"Arrivals loaded: {original_count}")
    print(f"Rows with valid actual time: {len(df)}")
    print(f"Unique physical arrivals: {len(df_unique)}")

    # Date Normalization
    date_cols = ['arr_scheduled_time', 'arr_actual_time']
    for col in date_cols:
        df_unique[col] = pd.to_datetime(df_unique[col], errors='coerce')
        df_unique[col] = df_unique[col].dt.floor('min')

    # Rename columns to avoid collision during merge
    # We prefix with prev_ because these represent the previous flight
    df_unique = df_unique.rename(columns={
        'arr_delay': 'prev_arr_delay',
        'arr_actual_time': 'prev_arr_actual_time',
        'arr_gate': 'gate_id',
        'dep_iata': 'prev_origin'
    })

    # Keep only relevant columns for the merge
    cols_to_keep = ['gate_id', 'prev_arr_actual_time', 'prev_arr_delay',
                    'prev_origin']

    return df_unique[cols_to_keep]

def merge_arrivals_departures(df_flights, df_arrivals):
    """
    Merges departures with the previous arrival AT THE SAME GATE.
    Uses pd.merge_asof with a short tolerance to capture gate turnover.
    """

    # Prepare Departure Data for Gate Join
    if 'dep_gate' in df_flights.columns:
        df_flights['gate_id'] = df_flights['dep_gate']
    else:
        print("CRITICAL WARNING: No 'dep_gate' or 'gate' column in flights data.")
        print("Cannot perform Gate-based merge. Returning original data.")
        return df_flights

    # Pre-Merge Cleaning
    # RIGHT SIDE (Arrivals): We MUST drop NaNs. An arrival at an unknown gate
    # cannot block a known gate.
    df_arrivals_clean = df_arrivals.dropna(subset=['gate_id']).sort_values(by='prev_arr_actual_time')

    # LEFT SIDE (Departures): We cannot drop rows (user requirement), but merge_asof
    # crashes with NaNs.
    # STRATEGY: Split -> Merge Valid -> Concat Invalid back
    mask_valid_gate = df_flights['gate_id'].notna()

    df_flights_valid = df_flights[mask_valid_gate].copy().sort_values(by='dep_scheduled_time')
    df_flights_nan = df_flights[~mask_valid_gate].copy() # Keep these safe to add back later

    print(f"Proceeding to merge {len(df_flights_valid)} records with valid gates...")

    # MERGE LOGIC:
    # We look for an arrival at 'gate_id' that happened BEFORE the
    # 'dep_scheduled_time'.
    # TOLERANCE: Reduced to 90 minutes.
    # Logic: If a plane arrived >90 mins ago, the gate likely cleared or
    # it's a different rotation.
    df_merged_valid = pd.merge_asof(
        df_flights_valid,
        df_arrivals_clean,
        left_on='dep_scheduled_time',
        right_on='prev_arr_actual_time',
        by='gate_id', # Match the Gate
        direction='backward',
        tolerance=pd.Timedelta(minutes=90)
    )

    # Recombine Data
    # Add back the flights that had no gate info (they will have NaNs for prev_arr columns)
    df_final = pd.concat([df_merged_valid, df_flights_nan], axis=0)

    # Restore original sort order (by scheduled time)
    df_final = df_final.sort_values(by='dep_scheduled_time')

    # 4. Feature Engineering: Gate Turnaround Time
    df_final['gate_turnaround_min'] = (
        df_final['dep_scheduled_time'] - df_final['prev_arr_actual_time']
    ).dt.total_seconds() / 60

    matches = df_final['prev_arr_delay'].notnull().sum()
    print(f"Successfully linked {matches} flights to a previous gate arrival.")
    print(f"Unlinked flights: {len(df_final) - matches}")

    return df_final

def analyze_rotational_delays(df):
    """
    Analyzes the impact of the previous flight's delay on the current flight,
    based on gate occupancy.
    """

    # Filter only rows where we successfully linked an arrival
    df_linked = df.dropna(subset=['prev_arr_delay'])

    if df_linked.empty:
        print("Not enough data linked to perform rotational analysis.")
        return

    # Correlation Statistics
    corr = df_linked['prev_arr_delay'].corr(df_linked['dep_delay'])
    print(f"Correlation (Prev Gate Arrival Delay vs Curr Dep Delay): {corr:.4f}")

    # Visualization: Scatter Plot with Regression Line
    plt.figure(figsize=(14, 6))

    # Scatter: Delay Propagation
    plt.subplot(1, 2, 1)
    sns.regplot(x='prev_arr_delay', y='dep_delay', data=df_linked,
                scatter_kws={'alpha':0.3, 's': 10}, line_kws={'color':'red'})
    plt.title(f'Gate Conflict: Delay Propagation (r={corr:.2f})')
    plt.xlabel('Previous Arrival Delay at Gate (min)')
    plt.ylabel('Current Departure Delay (min)')
    plt.grid(True, alpha=0.3)

    # Boxplot: Turnaround Time vs Delay
    # If the plane arrives very close to departure time (low turnaround),
    # delay should be higher.
    df_linked['turnaround_cat'] = pd.cut(
        df_linked['gate_turnaround_min'],
        bins=[-float('inf'), 30, 60, 90, float('inf')],
        labels=['Critical (<30m)', 'Tight (30-60m)', 'Standard (60-90m)', 'Long (>90m)']
    )

    plt.subplot(1, 2, 2)
    sns.boxplot(x='turnaround_cat', y='dep_delay', data=df_linked, showfliers=False)
    plt.title('Impact of Gate Turnaround Time on Delay')
    plt.xlabel('Time between Arrival and Scheduled Departure')
    plt.ylabel('Departure Delay (min)')

    plt.tight_layout()
    plt.show()

def add_graph_features(df):
    """
    Adds Graph Theory metrics based on the Star Graph topology (MAD -> Dest).
    Since we cannot cluster (no edges between destinations), we assign
    weights to the edges connecting MAD to the destination.
    """

    # We calculate two weights:
    # 1. Volume: How 'thick' is the edge? (Traffic)
    # 2. Risk: How 'heavy' is the edge? (Average Delay)
    route_stats = df.groupby('arr_iata').agg(
        raw_volume=('dep_delay', 'count'),
        raw_risk=('dep_delay', 'mean')
    ).reset_index()

    # Normalize Volume (Min-Max Scaling) to create a 0-1 edge weight
    # This represents the 'Centrality' of the destination relative to MAD
    min_vol = route_stats['raw_volume'].min()
    max_vol = route_stats['raw_volume'].max()

    route_stats['route_volume_weight'] = (
        (route_stats['raw_volume'] - min_vol) / (max_vol - min_vol)
    )

    # Rename risk column for clarity in the main dataframe
    route_stats.rename(columns={'raw_risk': 'route_risk_score'}, inplace=True)

    # Merge these weights back into the main dataframe
    # We use a Left Join to ensure we don't lose any flight records
    df_weighted = pd.merge(
        df,
        route_stats[['arr_iata', 'route_volume_weight', 'route_risk_score']],
        on='arr_iata',
        how='left'
    )

    print("Added 'route_volume_weight' and 'route_risk_score'.")
    return df_weighted

def add_occupancy_features(df_main, df_arrivals):
    """
    Calculates Airport Occupancy (Congestion) by counting unique flights
    per hour.
    """

    # Calculate Departure Occupancy (Congestion at Takeoff)
    # We use Scheduled Time because that determines the demand on the runway
    df_main['temp_hour_key'] = df_main['dep_scheduled_time'].dt.floor('H')

    dep_counts = df_main.groupby('temp_hour_key').size().reset_index(
        name='hourly_dep_count'
    )

    # Calculate Arrival Occupancy (Congestion at Landing)
    # We use the Arrivals dataframe. Even though we merged it earlier for
    # rotational delays, we use the full list here to get total landing flux.
    df_arrivals['temp_hour_key'] = df_arrivals['prev_arr_actual_time'].dt.floor('H')

    arr_counts = df_arrivals.groupby('temp_hour_key').size().reset_index(
        name='hourly_arr_count'
    )

    # We map the counts to the specific hour of the flight in df_main
    df_occupancy = pd.merge(
        df_main,
        dep_counts,
        on='temp_hour_key',
        how='left'
    )

    df_occupancy = pd.merge(
        df_occupancy,
        arr_counts,
        on='temp_hour_key',
        how='left'
    )

    # Fill NaNs with 0 (implies no flights found for that hour, which is rare
    # but possible in early morning)
    df_occupancy['hourly_dep_count'] = df_occupancy['hourly_dep_count'].fillna(0)
    df_occupancy['hourly_arr_count'] = df_occupancy['hourly_arr_count'].fillna(0)

    # Total Load Feature
    df_occupancy['total_airport_load'] = (
        df_occupancy['hourly_dep_count'] + df_occupancy['hourly_arr_count']
    )

    # Cleanup temporary key
    df_occupancy.drop(columns=['temp_hour_key'], inplace=True)

    print("Added 'hourly_dep_count', 'hourly_arr_count', and 'total_airport_load'.")
    return df_occupancy

# Main execution
if __name__ == "__main__":
    # Ensure raw data CSVs exist (generate if needed)
    ensure_raw_data_exists()
    
    # Paths to data files in FIRST BLOCK/data
    flights_path = FIRST_BLOCK / "data" / 'flights_data_raw.csv'
    weather_path = FIRST_BLOCK / "data" / 'weather_data_raw.csv'
    arrivals_path = FIRST_BLOCK / "data" / 'arrivals_complete_data_raw.csv'

    # Load & Clean Flights
    df_flights = load_and_clean_data(flights_path)

    # Load & Clean Weather
    df_weather = load_and_clean_weather(weather_path)

    # Load & Clean Arrivals
    df_arrivals = load_and_clean_arrivals(arrivals_path)

    # Merge Datasets
    if df_weather is not None:
        # Analyze Weather in Isolation
        analyze_weather_variables(df_weather)

        # Merge Datasets
        df_main = merge_flight_weather_data(df_flights, df_weather)
    else:
        print("Skipping weather analysis due to load error.")
        df_main = df_flights

    # Merge Arrivals
    if df_arrivals is not None:
        df_main = merge_arrivals_departures(df_main, df_arrivals)
    else:
        print("Skipping arrivals merge due to load error.")

    # Missing Values
    missing_values_analysis(df_main)

    # Target Analysis
    analyze_target_variable(df_main)

    # Feature Relationships
    analyze_correlations_and_categories(df_main)

    # Temporal Analysis
    analyze_temporal_patterns(df_main)

    # Weather Impact Analysis
    if 'temp' in df_main.columns:
        analyze_weather_impact(df_main)

    # Rotational/Gate Analysis
    if 'prev_arr_delay' in df_main.columns:
        analyze_rotational_delays(df_main)

    # Graph Theory Features (Edge Weights)
    df_main = add_graph_features(df_main)

    # Occupancy Features (Hourly Counts)
    if df_arrivals is not None:
        df_main = add_occupancy_features(df_main, df_arrivals)
    else:
        print("Skipping occupancy features: Arrivals data missing.")

    # Add target variable (delay >= 25 min) as last column
    df_main['is_delayed_25'] = df_main['dep_delay'] >= 25

    print(f"\nFinal Dataset Columns: {df_main.columns.tolist()}")
    df_main.to_csv("filtered_flights_data.csv", index=False)
