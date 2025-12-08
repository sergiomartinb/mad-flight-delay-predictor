import pickle
from pathlib import Path
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier
import plotly.express as px
import sys
from dotenv import load_dotenv
import requests
import os
from airport_data import IATA_LABELS, IATA_COORDS, MAD_COORDS

# Make sure we load .env from the same folder as config.py or from the project root just above it
HERE = Path(__file__).resolve().parent
env_path = HERE / ".env"
if not env_path.exists():
    env_path = HERE.parent / ".env"

load_dotenv(dotenv_path=env_path)

AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
AVIATIONEDGE_API_KEY = os.getenv("AVIATIONEDGE_API_KEY")

AVIATIONSTACK_BASE_URL = "http://api.aviationstack.com/v1"
AVIATIONEDGE_BASE_URL = "https://aviation-edge.com/v2/public"

if not AVIATIONSTACK_API_KEY:
    print("Warning: Missing AVIATIONSTACK_API_KEY in .env")
if not AVIATIONEDGE_API_KEY:
    print("Warning: Missing AVIATIONEDGE_API_KEY in .env")

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

FIRST_BLOCK = BASE_DIR / "FIRST BLOCK"
FIRST_BLOCK_DATA = FIRST_BLOCK / "data"
SECOND_BLOCK = BASE_DIR / "SECOND BLOCK"
THIRD_BLOCK = BASE_DIR / "THIRD BLOCK" / "model"

if str(FIRST_BLOCK) not in sys.path:
    sys.path.append(str(FIRST_BLOCK))

from aviationstack_client import fetch_all_flights
from aviationedge_client import fetch_future_schedules

# Set page config
st.set_page_config(
    page_title="MAD Flight Delay Dashboard",
    layout="wide",
)

# Custom CSS for navy blue and white theme with airport styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Sidebar - airport departure board style */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1a2744 100%);
        border-right: 3px solid #f0a500;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a2744 !important;
    }
    
    /* Main title with plane icon styling */
    h1 {
        border-bottom: 3px solid #f0a500;
        padding-bottom: 10px;
    }
    
    /* Tabs - runway/terminal gate style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #1a2744;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e8edf5;
        color: #1a2744;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a2744 !important;
        color: #ffffff !important;
        border-top: 3px solid #f0a500;
    }
    
    /* Buttons - boarding pass style */
    .stButton > button {
        background-color: #1a2744;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        border-left: 4px solid #f0a500;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #2d3e5f;
        color: #ffffff;
        border-left: 4px solid #ffcc00;
    }
    
    /* Metrics - flight info display style */
    [data-testid="stMetricValue"] {
        color: #1a2744;
        font-family: 'Courier New', monospace;
    }
    [data-testid="stMetricLabel"] {
        color: #666666;
        text-transform: uppercase;
        font-size: 0.75em;
        letter-spacing: 1px;
    }
    
    /* Selectbox and inputs */
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border: 1px solid #1a2744;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        border-radius: 4px;
        border-left: 4px solid #f0a500;
    }
    
    /* DataFrames - timetable style */
    .stDataFrame {
        border: 1px solid #1a2744;
    }
</style>
""", unsafe_allow_html=True)

st.title("MAD Flight Delay Dashboard")
st.caption("Adolfo Suárez Madrid-Barajas (MAD) - delays, weather and predictions")

# Load data functions
@st.cache_data
def load_filtered():
    df = pd.read_csv(SECOND_BLOCK / "filtered_flights_data.csv")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_raw_flights():
    return pd.read_csv(FIRST_BLOCK_DATA / "flights_data_raw.csv")


@st.cache_data
def load_raw_arrivals():
    return pd.read_csv(FIRST_BLOCK_DATA / "arrivals_complete_data_raw.csv")


@st.cache_data
def load_weather():
    df = pd.read_csv(FIRST_BLOCK_DATA / "weather_data_raw.csv")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

# Load model and add predictions
@st.cache_resource
def load_model():
    model = pickle.load(open(THIRD_BLOCK / "catboost_flight_delay_model.pkl", "rb"))
    model_features = pickle.load(open(THIRD_BLOCK / "model_features.pkl", "rb"))
    categorical_features = pickle.load(open(THIRD_BLOCK / "categorical_features.pkl", "rb"))
    return model, model_features, categorical_features

@st.cache_data
def add_model_predictions(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column 'model_delay_prob' with the CatBoost model's predicted
    probability of delay ≥ 25 minutes for each flight.
    Assumes the model in THIRD_BLOCK was trained on the same schema,
    with target 'is_delayed_25' meaning delay ≥ 25 minutes.
    """
    model, features, cat_features = load_model()

    X = df_filtered[features].copy()

    # Clean categorical features
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].fillna("Unknown").astype(str)
    X[[c for c in cat_features if c in X.columns]] = \
        X[[c for c in cat_features if c in X.columns]].astype(str)

    probs = model.predict_proba(X)[:, 1]  # probability of delay ≥ 25 min

    df = df_filtered.copy()
    df["model_delay_prob"] = probs
    return df

# Fetch current weather at MAD using Visual Crossing API
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
WEATHER_LOCATION = "LEMD"   # Madrid-Barajas

def _map_cond_grouped(text: str) -> str:
    """Map Visual Crossing conditions to cond_grouped categories."""
    t = (text or "").lower()
    if any(k in t for k in ["thunder", "storm"]):
        return "storm"
    if any(k in t for k in ["rain", "drizzle", "shower"]):
        return "rain"
    if any(k in t for k in ["snow", "sleet", "ice"]):
        return "snow"
    if any(k in t for k in ["fog", "mist", "haze"]):
        return "fog"
    if "cloud" in t:
        return "cloudy"
    return "clear"

@st.cache_data(ttl=3600)   # cache for 1 hour
def fetch_current_weather_mad():
    """
    Call Visual Crossing to get current weather at MAD.
    Returns (weather_dict, error_msg). If error_msg is not None, the call failed.
    """
    if not WEATHER_API_KEY:
        return None, "WEATHER_API_KEY is missing in .env"

    url = WEATHER_BASE_URL + WEATHER_LOCATION
    params = {
        "unitGroup": "metric",
        "include": "current",
        "key": WEATHER_API_KEY,
        "contentType": "json",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return None, f"Error calling Visual Crossing API: {e}"

    data = resp.json()
    cur = data.get("currentConditions")
    if not cur:
        return None, "Weather API response has no 'currentConditions' field."

    cond_text = cur.get("conditions") or ""
    mapped_cond = _map_cond_grouped(cond_text)

    weather = {
        "temp": cur.get("temp"),
        "precip": cur.get("precip"),
        "windspeed": cur.get("windspeed"),
        "visibility": cur.get("visibility"),
        "cloudcover": cur.get("cloudcover"),
        "cond_grouped": mapped_cond,
        "raw_conditions": cond_text,
        "datetime": cur.get("datetime"),
    }
    return weather, None

# Fetch live flight data from AviationStack API
from datetime import datetime, timedelta

@st.cache_data(ttl=300, show_spinner="Fetching flight schedule...")
def _fetch_future_flights_raw(target_date: str) -> list:
    """Fetch raw future flight data from API (cached)."""
    if not AVIATIONEDGE_API_KEY:
        return []
    try:
        return fetch_future_schedules("MAD", target_date, "departure")
    except Exception:
        return []

def fetch_future_flight_from_api(airline_name: str, flight_num_digits: str, target_date: str, df_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch future flight schedule from Aviation Edge API and prepare for model prediction.
    """
    flights = _fetch_future_flights_raw(target_date)
    
    if not flights:
        return pd.DataFrame()

    flat_records = []
    for rec in flights:
        dep = (rec.get("departure") or {})
        arr = (rec.get("arrival") or {})
        airline_data = (rec.get("airline") or {})
        flight_data = (rec.get("flight") or {})

        flat_records.append({
            "airline_name": airline_data.get("name"),
            "airline_iata": airline_data.get("iataCode"),
            "airline_icao": airline_data.get("icaoCode"),
            "flight_iata": flight_data.get("iataNumber"),
            "flight_icao": flight_data.get("icaoNumber"),
            "flight_number": flight_data.get("number"),
            "dep_iata": dep.get("iataCode", "").upper(),
            "dep_terminal": dep.get("terminal"),
            "dep_gate": dep.get("gate"),
            "dep_scheduled_time": f"{target_date} {dep.get('scheduledTime', '00:00')}",
            "arr_iata": arr.get("iataCode", "").upper(),
            "arr_terminal": arr.get("terminal"),
            "weekday": rec.get("weekday"),
        })

    df = pd.DataFrame(flat_records)
    if df.empty:
        return df

    df["flight_number_str"] = df["flight_number"].astype(str).str.upper()
    digits = flight_num_digits.strip().upper()

    mask_num = df["flight_number_str"].str.endswith(digits)
    
    airline_lower = airline_name.lower()
    # Match airline by: API name contains our name, our name contains API name, or IATA code matches
    api_names = df["airline_name"].astype(str).str.lower()
    mask_airline = (
        api_names.str.contains(airline_lower.split()[0], na=False) |  # First word match
        api_names.apply(lambda x: x in airline_lower) |  # API name is substring of our name
        (df["airline_iata"].astype(str).str.lower() == airline_lower[:2])
    )

    df_match = df[mask_num & mask_airline].copy()

    if df_match.empty:
        return pd.DataFrame()

    if "dep_scheduled_time" in df_match.columns:
        dt = pd.to_datetime(df_match["dep_scheduled_time"], errors="coerce")
        df_match["date"] = dt.dt.date
        df_match["dep_hour"] = dt.dt.hour
        df_match["dep_day_of_week"] = dt.dt.day_name()
        df_match["dep_month"] = dt.dt.month_name()

    df_match["dep_delay"] = 0
    df_match["is_delayed_25"] = False

    weather, weather_err = fetch_current_weather_mad()
    if weather and not weather_err:
        df_match["temp"] = weather.get("temp")
        df_match["precip"] = weather.get("precip") or 0
        df_match["windspeed"] = weather.get("windspeed")
        df_match["visibility"] = weather.get("visibility")
        df_match["cloudcover"] = weather.get("cloudcover")
        df_match["winddir"] = 0
        df_match["cond_grouped"] = weather.get("cond_grouped", "Unknown")
    else:
        df_match["temp"] = df_filtered["temp"].median() if "temp" in df_filtered.columns else 15.0
        df_match["precip"] = 0
        df_match["windspeed"] = df_filtered["windspeed"].median() if "windspeed" in df_filtered.columns else 10.0
        df_match["visibility"] = df_filtered["visibility"].median() if "visibility" in df_filtered.columns else 10.0
        df_match["cloudcover"] = df_filtered["cloudcover"].median() if "cloudcover" in df_filtered.columns else 50.0
        df_match["winddir"] = 0
        df_match["cond_grouped"] = "Partially cloudy"

    # Create lowercase version of arr_iata for case-insensitive lookup
    df_filtered_arr_lower = df_filtered["arr_iata"].astype(str).str.lower()
    
    for idx, row in df_match.iterrows():
        arr_code = str(row.get("arr_iata", "")).lower()
        if arr_code and arr_code in df_filtered_arr_lower.values:
            route_data = df_filtered[df_filtered_arr_lower == arr_code]
            df_match.loc[idx, "route_volume_weight"] = route_data["route_volume_weight"].iloc[0] if "route_volume_weight" in route_data.columns else 0.5
            df_match.loc[idx, "route_risk_score"] = route_data["route_risk_score"].iloc[0] if "route_risk_score" in route_data.columns else df_filtered["dep_delay"].mean()
        else:
            df_match.loc[idx, "route_volume_weight"] = 0.5
            df_match.loc[idx, "route_risk_score"] = df_filtered["dep_delay"].mean() if "dep_delay" in df_filtered.columns else 15.0

    current_hour = df_match["dep_hour"].iloc[0] if "dep_hour" in df_match.columns and len(df_match) > 0 else 12
    if "dep_hour" in df_filtered.columns:
        hourly_stats = df_filtered[df_filtered["dep_hour"] == current_hour]
        df_match["hourly_arr_count"] = hourly_stats["hourly_arr_count"].median() if "hourly_arr_count" in hourly_stats.columns and len(hourly_stats) > 0 else 30
        df_match["total_airport_load"] = hourly_stats["total_airport_load"].median() if "total_airport_load" in hourly_stats.columns and len(hourly_stats) > 0 else 60
    else:
        df_match["hourly_arr_count"] = 30
        df_match["total_airport_load"] = 60

    df_match["dep_terminal"] = df_match["dep_terminal"].fillna("Unknown")
    df_match["dep_gate"] = df_match["dep_gate"].fillna("Unknown")
    if "arr_terminal" not in df_match.columns:
        df_match["arr_terminal"] = "Unknown"
    else:
        df_match["arr_terminal"] = df_match["arr_terminal"].fillna("Unknown")
    
    df_match["gate_id"] = df_match["dep_gate"]

    return df_match

@st.cache_data(ttl=300)
def fetch_live_flight_from_api(airline_name: str, flight_num_digits: str, df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Fetch live flight data from API and prepare it for model prediction."""
    if not AVIATIONSTACK_API_KEY:
        st.info("Live API lookups are disabled because AVIATIONSTACK_API_KEY is missing in .env.")
        return pd.DataFrame()
    
    params = {
        "dep_iata": "MAD",
        "access_key": AVIATIONSTACK_API_KEY
    }
    flights = fetch_all_flights(params, max_pages=5) 
    if not flights:
        return pd.DataFrame()

    flat_records = []
    for rec in flights:
        dep = (rec.get("departure") or {})
        arr = (rec.get("arrival") or {})
        airline = (rec.get("airline") or {})
        flight = (rec.get("flight") or {})
        aircraft = (rec.get("aircraft") or {})

        flat_records.append(
            {
                "flight_status": rec.get("flight_status"),
                "airline_name": airline.get("name"),
                "airline_iata": airline.get("iata"),
                "airline_icao": airline.get("icao"),
                "flight_iata": flight.get("iata"),
                "flight_icao": flight.get("icao"),
                "flight_number": flight.get("number"),
                "dep_iata": dep.get("iata"),
                "dep_terminal": dep.get("terminal"),
                "dep_gate": dep.get("gate"),
                "dep_scheduled_time": dep.get("scheduled"),
                "dep_actual_time": dep.get("actual"),
                "dep_delay": dep.get("delay"),
                "arr_iata": arr.get("iata"),
                "arr_terminal": arr.get("terminal"),
                "arr_scheduled_time": arr.get("scheduled"),
                "arr_actual_time": arr.get("actual"),
                "arr_delay": arr.get("delay"),
                "aircraft_reg": aircraft.get("registration"),
            }
        )

    df = pd.DataFrame(flat_records)
    if df.empty:
        return df

    if "dep_iata" in df.columns:
        df = df[df["dep_iata"].astype(str).str.upper() == "MAD"]

    if df.empty:
        return df

    df["flight_number_str"] = df["flight_number"].astype(str).str.upper()
    digits = flight_num_digits.strip().upper()

    mask_num = df["flight_number_str"].str.endswith(digits)
    mask_airline = df["airline_name"].astype(str).str.lower() == airline_name.lower()

    df_match = df[mask_num & mask_airline].copy()

    if df_match.empty:
        return pd.DataFrame()

    if "dep_scheduled_time" in df_match.columns:
        dt = pd.to_datetime(df_match["dep_scheduled_time"], errors="coerce")
        df_match["date"] = dt.dt.date
        df_match["dep_hour"] = dt.dt.hour
        df_match["dep_day_of_week"] = dt.dt.day_name()
        df_match["dep_month"] = dt.dt.month_name()

    if "dep_delay" in df_match.columns:
        delay_val = pd.to_numeric(df_match["dep_delay"], errors="coerce").fillna(0)
        df_match["is_delayed_25"] = delay_val >= 25
    else:
        df_match["is_delayed_25"] = False
        df_match["dep_delay"] = 0

    weather, weather_err = fetch_current_weather_mad()
    if weather and not weather_err:
        df_match["temp"] = weather.get("temp")
        df_match["precip"] = weather.get("precip") or 0
        df_match["windspeed"] = weather.get("windspeed")
        df_match["visibility"] = weather.get("visibility")
        df_match["cloudcover"] = weather.get("cloudcover")
        df_match["winddir"] = 0
        df_match["cond_grouped"] = weather.get("cond_grouped", "Unknown")
    else:
        df_match["temp"] = df_filtered["temp"].median() if "temp" in df_filtered.columns else 15.0
        df_match["precip"] = 0
        df_match["windspeed"] = df_filtered["windspeed"].median() if "windspeed" in df_filtered.columns else 10.0
        df_match["visibility"] = df_filtered["visibility"].median() if "visibility" in df_filtered.columns else 10.0
        df_match["cloudcover"] = df_filtered["cloudcover"].median() if "cloudcover" in df_filtered.columns else 50.0
        df_match["winddir"] = 0
        df_match["cond_grouped"] = "Partially cloudy"

    # Create lowercase version of arr_iata for case-insensitive lookup
    df_filtered_arr_lower = df_filtered["arr_iata"].astype(str).str.lower()
    
    for idx, row in df_match.iterrows():
        arr_code = str(row.get("arr_iata", "")).lower()
        if arr_code and arr_code in df_filtered_arr_lower.values:
            route_data = df_filtered[df_filtered_arr_lower == arr_code]
            df_match.loc[idx, "route_volume_weight"] = route_data["route_volume_weight"].iloc[0] if "route_volume_weight" in route_data.columns else 0.5
            df_match.loc[idx, "route_risk_score"] = route_data["route_risk_score"].iloc[0] if "route_risk_score" in route_data.columns else df_filtered["dep_delay"].mean()
        else:
            df_match.loc[idx, "route_volume_weight"] = 0.5
            df_match.loc[idx, "route_risk_score"] = df_filtered["dep_delay"].mean() if "dep_delay" in df_filtered.columns else 15.0

    current_hour = df_match["dep_hour"].iloc[0] if "dep_hour" in df_match.columns else 12
    if "dep_hour" in df_filtered.columns:
        hourly_stats = df_filtered[df_filtered["dep_hour"] == current_hour]
        df_match["hourly_arr_count"] = hourly_stats["hourly_arr_count"].median() if "hourly_arr_count" in hourly_stats.columns and len(hourly_stats) > 0 else 30
        df_match["total_airport_load"] = hourly_stats["total_airport_load"].median() if "total_airport_load" in hourly_stats.columns and len(hourly_stats) > 0 else 60
    else:
        df_match["hourly_arr_count"] = 30
        df_match["total_airport_load"] = 60

    df_match["dep_terminal"] = df_match["dep_terminal"].fillna("Unknown")
    df_match["dep_gate"] = df_match["dep_gate"].fillna("Unknown")
    if "arr_terminal" not in df_match.columns:
        df_match["arr_terminal"] = "Unknown"
    else:
        df_match["arr_terminal"] = df_match["arr_terminal"].fillna("Unknown")
    
    df_match["gate_id"] = df_match["dep_gate"]

    return df_match

# Dashboard overview page
def page_overview(df_filtered, df_raw, df_arrivals, df_weather):
    st.subheader("Overview")

    total_flights = len(df_filtered)
    delay_rate = df_filtered["is_delayed_25"].mean()
    avg_dep_delay = df_filtered["dep_delay"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total flights in dataset", f"{total_flights:,}")
    c2.metric("Delay rate ≥ 25 min", f"{delay_rate*100:.1f}%")
    c3.metric("Average dep. delay (min)", f"{avg_dep_delay:.1f}")

    st.divider()

    tab_summary, tab_delays, tab_weather = st.tabs(
        ["Summary", "Delays", "Weather"]
    )

    with tab_summary:
        st.markdown("### What this dashboard shows")
        st.write(
            """
            This dashboard tracks flight operations at **Adolfo Suárez Madrid-Barajas (MAD)** and
            analyzes **departure delays ≥ 25 minutes** together with **local weather conditions**.

            - The **Delays** tab shows daily patterns, delay by hour, airlines with the
              highest delay rates, and the most problematic routes from MAD.  
            - The **Weather** tab shows how temperature, visibility and cloud cover evolve
              over time.  
            - The **Predict Delay** page (in the sidebar) lets you check the delay risk
              for a specific flight number and even simulate different weather scenarios.
            """
        )
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: #666666; font-size: 0.85em;'>"
            "© 2025 Javier Fernández, Sergio Martín, Amaranta Canova, Víctor Rodrigo. All rights reserved."
            "</p>",
            unsafe_allow_html=True
        )

    with tab_delays:
        st.markdown("###  Daily delay rate (historical vs model)")

        if "date" in df_filtered.columns and "model_delay_prob" in df_filtered.columns:
            daily = (
                df_filtered
                .assign(Date=df_filtered["date"].dt.date)
                .groupby("Date")
                .agg(
                    historical_rate=("is_delayed_25", "mean"),
                    model_rate=("model_delay_prob", "mean"),
                )
                .reset_index()
            )

            daily_long = daily.melt(
                id_vars="Date",
                value_vars=["historical_rate", "model_rate"],
                var_name="Series",
                value_name="Value",
            )
            daily_long["Value"] = daily_long["Value"].astype(float)
            daily_long["Series"] = daily_long["Series"].map(
                {
                    "historical_rate": "Historical delay rate (≥ 25 min)",
                    "model_rate": "Model predicted delay probability",
                }
            )

            fig_delay = px.line(
                daily_long,
                x="Date",
                y="Value",
                color="Series",
                markers=True,
                labels={"Date": "Date", "Value": "Delay probability"},
                template="plotly_dark",
            )
            fig_delay.update_layout(height=350)
            st.plotly_chart(fig_delay, use_container_width=True)
            
            st.caption(
                "**Note:** Data was collected in batches of approximately 2 weeks per month, "
                "which explains the gaps and periodic patterns visible in the chart. "
            )
        else:
            st.info("Required columns `date` or `model_delay_prob` are missing.")

        st.markdown("###  Delay rate by departure hour")

        if "dep_hour" in df_filtered.columns:
            hourly = (
                df_filtered.groupby("dep_hour")["is_delayed_25"]
                .mean()
                .rename("Delay rate")
                .reset_index()
            )
            fig_hour = px.bar(
                hourly,
                x="dep_hour",
                y="Delay rate",
                labels={"dep_hour": "Hour of day"},
                template="plotly_white",
            )
            fig_hour.update_layout(height=350)
            st.plotly_chart(fig_hour, use_container_width=True)
            
            st.caption(
                "Delays tend to build up throughout the day due to earlier disruptions in the schedule"
                " affecting later flights."
            )

        st.markdown("---")

        st.markdown("###  Departure delay distribution by month")

        if "dep_month" in df_filtered.columns and "dep_delay" in df_filtered.columns:
            df_box = df_filtered[df_filtered["dep_delay"].between(-10, 90)].copy()
            
            month_order = ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            months_present = [m for m in month_order if m in df_box["dep_month"].unique()]
            
            fig_box = px.box(
                df_box,
                x="dep_month",
                y="dep_delay",
                category_orders={"dep_month": months_present},
                labels={
                    "dep_month": "Month",
                    "dep_delay": "Departure Delay (min)",
                },
                template="plotly_white",
                color="dep_month",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_box.update_traces(boxpoints=False, hoverinfo="skip")
            fig_box.update_layout(
                height=420,
                showlegend=False,
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            st.caption(
                "Boxplot showing the spread of departure delays by month. "
                "Outliers are hidden to better visualize the distribution differences."
            )
        else:
            st.info("Required columns for month boxplot are missing.")

        st.markdown("---")

        st.markdown("###  Airlines with the highest delay rate")

        if "airline_name" in df_filtered.columns:
            airline_stats = (
                df_filtered.groupby("airline_name")
                .agg(
                    flights=("is_delayed_25", "size"),
                    delay_rate=("is_delayed_25", "mean"),
                    avg_dep_delay=("dep_delay", "mean"),
                )
                .reset_index()
            )

            min_flights = st.slider(
                "Minimum number of flights per airline to include",
                min_value=20,
                max_value=int(airline_stats["flights"].max()),
                value=100,
                step=10,
            )

            airline_filtered = airline_stats[airline_stats["flights"] >= min_flights]
            top_airlines = (
                airline_filtered.sort_values("delay_rate", ascending=False)
                .head(15)
            )

            fig_airline = px.bar(
                top_airlines,
                x="airline_name",
                y="delay_rate",
                hover_data=["flights", "avg_dep_delay"],
                labels={
                    "airline_name": "Airline",
                    "delay_rate": "Delay rate (≥ 25 min)",
                    "avg_dep_delay": "Avg dep. delay (min)",
                },
                template="plotly_white",
            )
            fig_airline.update_layout(
                xaxis_tickangle=-45,
                height=400,
                margin=dict(l=10, r=10, t=30, b=150),
            )
            st.plotly_chart(fig_airline, use_container_width=True)

            st.caption(
                "Bars are sorted by delay rate. Hover to see how many flights each airline has "
                "and its average departure delay in minutes."
            )
        else:
            st.info("No `airline_name` column available to compute airline delay statistics.")

        st.markdown("---")

        st.markdown("###  Routes from MAD with the highest delay")

        if "arr_iata" in df_filtered.columns:
            route_stats = (
                df_filtered.groupby("arr_iata")
                .agg(
                    flights=("is_delayed_25", "size"),
                    delay_rate=("is_delayed_25", "mean"),
                    avg_dep_delay=("dep_delay", "mean"),
                )
                .reset_index()
                .dropna(subset=["arr_iata"])
            )

            col_r1, col_r2, col_r3 = st.columns([2, 1, 2])
            with col_r1:
                min_route_flights = st.slider(
                    "Minimum number of flights per route to include",
                    min_value=20,
                    max_value=int(route_stats["flights"].max()),
                    value=80,
                    step=10,
                )

            with col_r2:
                top_n_routes = st.selectbox(
                    "Number of routes to show",
                    options=[5, 10, 15, 20],
                    index=2,
                )

            with col_r3:
                min_delay_rate = st.slider(
                    "Minimum delay rate to show on map (%)",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=5,
                    help="Filter routes on the map by delay rate threshold",
                )

            route_filtered = route_stats[route_stats["flights"] >= min_route_flights]
            top_routes = (
                route_filtered.sort_values("delay_rate", ascending=False)
                .head(top_n_routes)
            )

            top_routes = top_routes.copy()
            top_routes["arr_iata"] = top_routes["arr_iata"].str.upper()
            top_routes["Destination"] = top_routes["arr_iata"].apply(
                lambda code: IATA_LABELS.get(code, f"{code}")
            )
            table_df = top_routes.copy()
            table_df["Delay rate (%)"] = (table_df["delay_rate"] * 100).round(1)
            table_df["Avg dep. delay (min)"] = table_df["avg_dep_delay"].round(1)

            table_df = table_df[
                ["Destination", "flights", "Delay rate (%)", "Avg dep. delay (min)"]
            ].rename(
                columns={
                    "flights": "Flights",
                }
            )

            st.markdown("Top delayed routes (MAD → destination):")
            st.dataframe(
                table_df,
                use_container_width=True,
            )

            MAD_LAT, MAD_LON = MAD_COORDS
            map_rows = []
            for _, row in top_routes.iterrows():
                code = str(row["arr_iata"]).upper()
                if code in IATA_COORDS:
                    lat, lon = IATA_COORDS[code]
                    map_rows.append(
                        {
                            "Destination": IATA_LABELS.get(code, code),
                            "code": code,
                            "lat": lat,
                            "lon": lon,
                            "delay_rate": row["delay_rate"],
                            "flights": row["flights"],
                            "avg_dep_delay": row["avg_dep_delay"],
                        }
                    )

            if map_rows:
                import plotly.graph_objects as go
                
                map_df = pd.DataFrame(map_rows)
                min_delay_threshold = min_delay_rate / 100.0
                map_df_filtered = map_df[map_df["delay_rate"] >= min_delay_threshold]
                
                fig_map = go.Figure()
                
                for _, dest in map_df_filtered.iterrows():
                    line_color = f"rgba(255, {int(255 * (1 - dest['delay_rate']))}, {int(255 * (1 - dest['delay_rate']))}, 0.5)"
                    
                    fig_map.add_trace(go.Scattergeo(
                        lon=[MAD_LON, dest["lon"]],
                        lat=[MAD_LAT, dest["lat"]],
                        mode="lines",
                        line=dict(
                            width=1.5,  # Thinner fixed line width
                            color=line_color,
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    ))
                
                fig_map.add_trace(go.Scattergeo(
                    lon=[MAD_LON],
                    lat=[MAD_LAT],
                    mode="markers",
                    marker=dict(size=12, color="#1E90FF", symbol="star", line=dict(width=1, color="white")),
                    name="Madrid (MAD)",
                    hovertemplate="<b>Madrid-Barajas (MAD)</b><br>Origin Airport<extra></extra>",
                ))
                
                if not map_df_filtered.empty:
                    fig_map.add_trace(go.Scattergeo(
                        lon=map_df_filtered["lon"],
                        lat=map_df_filtered["lat"],
                        mode="markers",
                        marker=dict(
                            size=map_df_filtered["flights"] / map_df_filtered["flights"].max() * 18 + 8,  # Scale marker size
                            color=map_df_filtered["delay_rate"],
                            colorscale="Reds",
                            cmin=0,
                            cmax=1,
                            colorbar=dict(title="Delay Rate"),
                            line=dict(width=1, color="white"),
                        ),
                        text=map_df_filtered["Destination"],
                        customdata=map_df_filtered[["flights", "avg_dep_delay", "delay_rate"]].values,
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "Flights: %{customdata[0]}<br>"
                            "Delay rate: %{customdata[2]:.1%}<br>"
                            "Avg delay: %{customdata[1]:.1f} min<extra></extra>"
                        ),
                        name="Destinations",
                    ))
                
                fig_map.update_geos(
                    projection_type="natural earth",
                    showland=True,
                    landcolor="rgb(243, 243, 243)",
                    showocean=True,
                    oceancolor="rgb(204, 229, 255)",
                    showcoastlines=True,
                    coastlinecolor="rgb(180, 180, 180)",
                    showframe=False,
                )
                
                routes_shown = len(map_df_filtered)
                routes_total = len(map_df)
                
                fig_map.update_layout(
                    height=550,
                    margin=dict(l=0, r=0, t=30, b=0),
                    title=dict(
                        text=f"Routes from Madrid (MAD) — showing {routes_shown} of {routes_total} routes with delay ≥{min_delay_rate}%",
                        x=0.5,
                        xanchor="center",
                        font=dict(size=14),
                    ),
                    showlegend=False,
                )
                
                st.plotly_chart(fig_map, use_container_width=True)
                
                st.caption(
                    "Map shows flight routes from Madrid-Barajas (MAD) to destinations. "
                    "Marker color indicates delay rate (darker red = higher delays), "
                    "marker size reflects flight volume. Use the delay rate slider to filter high-risk routes."
                )
            else:
                st.info(
                    "We don't have coordinates for these destinations in the built-in dictionary. "
                    "The table above still shows the most delayed routes."
                )
        else:
            st.info("No `arr_iata` column available to compute route delay statistics.")

    with tab_weather:
        st.markdown("### Weather trends at MAD")

        weather_vars = [
            var for var in ["temp", "visibility", "cloudcover"]
            if var in df_weather.columns
        ]

        selected = st.multiselect(
            "Select weather variables to plot",
            weather_vars,
            default=weather_vars[:1] if weather_vars else [],
        )

        if selected:
            df_w = df_weather.copy()
            df_w["date_only"] = df_w["date"].dt.date

            fig_weather = px.line(df_w, x="date_only", y=selected)
            fig_weather.update_layout(height=400)
            st.plotly_chart(fig_weather, use_container_width=True)
        else:
            st.info("Select at least one weather variable to display a chart.")

        st.markdown("---")
        st.markdown("###  Flight distribution by weather condition")

        if "cond_grouped" in df_filtered.columns:
            weather_counts = df_filtered["cond_grouped"].value_counts().reset_index()
            weather_counts.columns = ["Weather Condition", "Flights"]
            
            fig_pie = px.pie(
                weather_counts,
                values="Flights",
                names="Weather Condition",
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.caption(
                "Distribution of flights across different weather conditions at MAD."
            )
        else:
            st.info("No `cond_grouped` column available to show weather distribution.")


def page_explorer(df_filtered: pd.DataFrame):
    st.subheader("Flights Explorer")

    st.markdown(
        "Use the filters below to explore flights, historical delays (≥ 25 min) and "
        "the model's predicted delay risk."
    )

    # Ensure date column is datetime
    if "date" in df_filtered.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_filtered["date"]):
            df_filtered = df_filtered.copy()
            df_filtered["date"] = pd.to_datetime(df_filtered["date"])
    else:
        st.error("The dataset does not contain a `date` column, so the Explorer cannot be used.")
        return

    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)

        dates = sorted(df_filtered["date"].dt.date.unique())
        date_options = ["All dates"] + dates

        def _date_label(x):
            return "All dates" if x == "All dates" else str(x)

        selected_date = col1.selectbox("Date", options=date_options, format_func=_date_label)

        airline_list = sorted(df_filtered["airline_name"].dropna().unique().tolist()) \
            if "airline_name" in df_filtered.columns else []
        airline_options = ["All"] + airline_list
        selected_airline = col2.selectbox("Airline", airline_options)

        status_options = ["All", "Delayed ≥ 25 min", "Not delayed"]
        selected_status = col3.selectbox("Delay status", status_options)

    df = df_filtered.copy()

    if selected_date != "All dates":
        df = df[df["date"].dt.date == selected_date]

    if selected_airline != "All" and "airline_name" in df.columns:
        df = df[df["airline_name"] == selected_airline]

    if selected_status == "Delayed ≥ 25 min":
        df = df[df["is_delayed_25"] == True]   # delay ≥ 25 min
    elif selected_status == "Not delayed":
        df = df[df["is_delayed_25"] == False]

    st.write(f"Showing **{len(df):,}** flights for the selected filters.")

    if df.empty:
        st.info("No flights match these filters. Try changing the date, airline or delay status.")
        return

    tab_table, tab_hour, tab_airline = st.tabs(
        ["Flights table", "Delay rate by hour", "Delay rate by airline"]
    )

    with tab_table:
        st.markdown("### Flights for the selected filters")

        cols_to_show = [
            c
            for c in [
                "date",
                "airline_name",
                "flight_icao",
                "flight_iata",
                "flight_number",
                "dep_hour",
                "dep_day_of_week",
                "dep_terminal",
                "dep_gate",
                "arr_iata",
                "dep_delay",
                "arr_delay",
                "is_delayed_25",      # delay ≥ 25 min
                "model_delay_prob",   # model prediction
                "cond_grouped",
            ]
            if c in df.columns
        ]

        df_display = df.sort_values(
            [col for col in ["date", "dep_hour", "airline_name", "flight_icao"] if col in df.columns]
        ).copy()

        # Show model prob as %
        if "model_delay_prob" in df_display.columns:
            df_display["model_delay_prob"] = (df_display["model_delay_prob"] * 100).round(1)

        st.dataframe(
            df_display[cols_to_show],
            use_container_width=True,
            height=420,
        )

    with tab_hour:
        st.markdown("###  Historical vs predicted delay rate by departure hour")

        if "dep_hour" in df.columns and "model_delay_prob" in df.columns:
            hourly = (
                df.groupby("dep_hour")
                .agg(
                    historical_rate=("is_delayed_25", "mean"),
                    model_rate=("model_delay_prob", "mean"),
                    samples=("is_delayed_25", "size"),
                )
                .reset_index()
            )

            df_hour_long = hourly.melt(
                id_vars=["dep_hour", "samples"],
                value_vars=["historical_rate", "model_rate"],
                var_name="Series",
                value_name="Value",
            )
            df_hour_long["Value"] = df_hour_long["Value"].astype(float)
            df_hour_long["Series"] = df_hour_long["Series"].map(
                {
                    "historical_rate": "Historical delay rate (≥ 25 min)",
                    "model_rate": "Model predicted delay probability",
                }
            )

            fig = px.line(
                df_hour_long,
                x="dep_hour",
                y="Value",
                color="Series",
                labels={"dep_hour": "Departure hour", "Value": "Delay probability"},
            )
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Hover the lines to compare historical vs model risk by hour.")
        else:
            st.info("Required columns `dep_hour` or `model_delay_prob` are missing.")

    with tab_airline:
        st.markdown("###  Historical vs predicted delay rate by airline")

        if "airline_name" in df.columns and "model_delay_prob" in df.columns:
            air_stats = (
                df.groupby("airline_name")
                .agg(
                    historical_rate=("is_delayed_25", "mean"),
                    model_rate=("model_delay_prob", "mean"),
                    flights=("is_delayed_25", "size"),
                )
                .reset_index()
            )

            col_a1, col_a2, col_a3 = st.columns(3)
            min_flights = col_a1.slider(
                "Minimum flights per airline",
                min_value=20,
                max_value=int(air_stats["flights"].max()),
                value=150,
                step=10,
            )

            sort_label = col_a2.selectbox(
                "Sort airlines by",
                ["Historical delay rate (≥ 25 min)", "Model predicted delay probability"],
            )

            top_n = col_a3.slider(
                "Number of airlines to show",
                min_value=5,
                max_value=40,
                value=15,
            )

            sort_col = (
                "historical_rate"
                if "Historical" in sort_label
                else "model_rate"
            )

            air_filtered = air_stats[air_stats["flights"] >= min_flights].copy()
            air_top = (
                air_filtered.sort_values(sort_col, ascending=False)
                .head(top_n)
            )

            if air_top.empty:
                st.info("No airlines satisfy this filter. Try lowering the minimum flights.")
            else:
                df_air_long = air_top.melt(
                    id_vars=["airline_name", "flights"],
                    value_vars=["historical_rate", "model_rate"],
                    var_name="Series",
                    value_name="Value",
                )
                df_air_long["Value"] = df_air_long["Value"].astype(float)
                df_air_long["Series"] = df_air_long["Series"].map(
                    {
                        "historical_rate": "Historical delay rate (≥ 25 min)",
                        "model_rate": "Model predicted delay probability",
                    }
                )

                fig = px.bar(
                    df_air_long,
                    y="airline_name",
                    x="Value",
                    color="Series",
                    barmode="group",
                    labels={"airline_name": "Airline", "Value": "Delay probability"},
                    orientation="h",
                )
                fig.update_layout(
                    height=500,
                    margin=dict(l=10, r=10, t=30, b=10),
                    yaxis={"categoryorder": "total ascending"},  # most delayed at top
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Bars compare true delay rates vs model predictions for the filtered subset. "
                    "Use the sliders above to focus on airlines with enough data."
                )
        else:
            st.info("Required columns `airline_name` or `model_delay_prob` are missing.")



def page_predict(df_filtered: pd.DataFrame):
    st.subheader("Check if your flight is likely to be delayed")

    # model is still needed for flight-specific history + weather simulation
    model, features, cat_features = load_model()

    st.markdown(
        """
        **Option 1 - Specific flight**

        1. Select the **airline**  
        2. Enter the **flight number (digits only)** - for example, if the flight is `RY2629`,
           select *ryanair* and type `2629`  
        3. We look up all historical records of that flight and estimate the chance it is
           **delayed ≥ 25 minutes**.

        **Option 2 - Route level**

        Even if we don't have that exact flight number, you can scroll down to check the predicted
        delay risk for a **route (airline + destination)**.
        """
    )

    st.markdown("###  Option 1: Specific flight")
    st.caption("Only airlines with historical data are shown. The model's accuracy depends on having past records for comparison.")

    col1, col2, col3 = st.columns([2, 1, 1])

    airlines = sorted(df_filtered["airline_name"].dropna().unique().tolist())
    default_airline_idx = airlines.index("ryanair") if "ryanair" in airlines else 0
    selected_airline = col1.selectbox("Airline", airlines, index=default_airline_idx, key="flight_airline")

    flight_num_input = col2.text_input(
        "Flight number (numbers only)",
        "",
        placeholder="e.g. 2629",
        key="flight_number_input",
    ).strip()
    
    # Date picker for future flights
    # Aviation Edge Future Schedules API requires dates at least 7 days ahead
    today = datetime.now().date()
    min_future_date = today + timedelta(days=7)
    max_date = today + timedelta(days=365)
    selected_date = col3.date_input(
        "Flight date",
        value=min_future_date,
        min_value=min_future_date,
        max_value=max_date,
        key="flight_date",
        help="Schedule data available from 7 days ahead up to 1 year"
    )

    if not flight_num_input:
        st.info("Select an airline, type the flight number digits, and choose a date (7+ days ahead) to get a prediction.")
    else:
        # Filter data for that airline
        df_air = df_filtered[df_filtered["airline_name"] == selected_airline].copy()

        # Try to match by different id columns (digits at the end)
        candidate_cols = ["flight_number", "flight_icao", "flight_iata"]
        mask = pd.Series(False, index=df_air.index)

        for col in candidate_cols:
            if col in df_air.columns:
                col_str = df_air[col].astype(str).str.upper()
                mask = mask | col_str.str.endswith(flight_num_input)

        matches = df_air[mask].copy()
        
        # Format selected date
        date_str = selected_date.strftime("%Y-%m-%d")
        
        # Fetch future schedule for the selected date
        future_matches = fetch_future_flight_from_api(selected_airline, flight_num_input, date_str, df_filtered)

        if future_matches.empty and matches.empty:
            st.error(
                f"We couldn't find flight **{selected_airline} {flight_num_input}** "
                f"scheduled for **{selected_date.strftime('%d %b %Y')}**.\n\n"
                "This could mean:\n"
                "- The flight doesn't operate on this date\n"
                "- The flight number is incorrect\n\n"
                "You can still check the delay risk for the **route (airline + destination)** "
                "in the section below."
            )
        elif not future_matches.empty:
            # We found the flight in future schedules - make prediction
            st.success(
                f"Found flight **{selected_airline} {flight_num_input}** scheduled for "
                f"**{selected_date.strftime('%d %b %Y')}**"
            )

            if "dep_scheduled_time" in future_matches.columns:
                future_matches = future_matches.sort_values("dep_scheduled_time")

            flight_row = future_matches.iloc[-1:].copy()

            for col in features:
                if col not in flight_row.columns:
                    flight_row[col] = None

            X_flight = flight_row[features].copy()

            for c in cat_features:
                if c in X_flight.columns:
                    X_flight[c] = X_flight[c].fillna("Unknown").astype(str)

            X_flight[[c for c in cat_features if c in X_flight.columns]] = \
                X_flight[[c for c in cat_features if c in X_flight.columns]].astype(str)

            proba_flight = model.predict_proba(X_flight)[0, 1]
            proba_flight_pct = round(proba_flight * 100)

            if proba_flight >= 0.6:
                risk_label = "High"
                risk_emoji = "🔴"
                card_color = "#ffe5e5"
                border_color = "#ff4b4b"
            elif proba_flight >= 0.4:
                risk_label = "Medium"
                risk_emoji = "🟡"
                card_color = "#fff7d6"
                border_color = "#ffcc00"
            else:
                risk_label = "Low"
                risk_emoji = "🟢"
                card_color = "#e6ffe6"
                border_color = "#33aa33"

            # Get scheduled time and other info
            dest = flight_row["arr_iata"].iloc[0] if "arr_iata" in flight_row.columns else "N/A"

            st.markdown("---")

            card_html = f"""
            <div style="
                background-color:{card_color};
                border-left: 6px solid {border_color};
                padding: 1rem 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                color: #222222;
            ">
              <h3 style="margin-top:0; margin-bottom:0.5rem; color: #222222;">
                {risk_emoji} Delay risk: <b>{risk_label}</b> for flight
                <b>{selected_airline} {flight_num_input}</b>
              </h3>
              <p style="color: #222222;"><b>Model prediction:</b> {proba_flight_pct}% chance of delay ≥ 25 minutes</p>
              <p style="color: #222222;"><b>Date:</b> {selected_date.strftime('%d %b %Y')}</p>
              <p style="color: #222222;"><b>Destination:</b> {dest}</p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Show historical context if we have it
            if not matches.empty:
                hist_delay_rate = matches["is_delayed_25"].mean()
                avg_dep_delay = matches["dep_delay"].mean()
                h_pct = round(hist_delay_rate * 100)
                avg_delay_round = round(avg_dep_delay)
                
                st.markdown(f"""
                <div style="background-color:#f5f5f5; padding: 0.8rem; border-radius: 0.3rem; color: #333;">
                <b>Historical context:</b> This flight was delayed in <b>{h_pct}%</b> of past records 
                ({len(matches)} flights). Average delay when late: <b>{avg_delay_round} min</b>.
                </div>
                """, unsafe_allow_html=True)

            with st.expander("See flight data used for this prediction"):
                st.dataframe(flight_row, use_container_width=True)

        else:
            # We have historical data but no future schedule for this date
            # Still use future schedule API to get flight info, with historical model average as fallback
            st.success(
                f"Found **{len(matches)}** historical records for flight "
                f"**{selected_airline} {flight_num_input}**"
            )
            
            st.warning(
                f"Could not fetch schedule for **{selected_date.strftime('%d %b %Y')}** from the API. "
                "Using historical data for prediction."
            )
            
            delay_col = "is_delayed_25"
            hist_delay_rate = matches[delay_col].mean()
            avg_dep_delay = matches["dep_delay"].mean()

            X_all = matches[features].copy()

            for c in cat_features:
                if c in X_all.columns:
                    X_all[c] = X_all[c].fillna("Unknown").astype(str)

            X_all[[c for c in cat_features if c in X_all.columns]] = \
                X_all[[c for c in cat_features if c in X_all.columns]].astype(str)

            probs_all = model.predict_proba(X_all)[:, 1]
            matches["predicted_delay_prob"] = probs_all

            avg_model_prob = probs_all.mean()

            p = avg_model_prob
            h = hist_delay_rate

            if p >= 0.6 or h >= 0.75:
                risk_label = "High"
                risk_emoji = "🔴"
                card_color = "#ffe5e5"
                border_color = "#ff4b4b"
            elif p >= 0.4 or h >= 0.5:
                risk_label = "Medium"
                risk_emoji = "🟡"
                card_color = "#fff7d6"
                border_color = "#ffcc00"
            else:
                risk_label = "Low"
                risk_emoji = "🟢"
                card_color = "#e6ffe6"
                border_color = "#33aa33"

            p_pct = round(p * 100)
            h_pct = round(h * 100)
            avg_delay_round = round(avg_dep_delay)

            st.markdown("---")

            card_html = f"""
            <div style="
                background-color:{card_color};
                border-left: 6px solid {border_color};
                padding: 1rem 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                color: #222222;
            ">
              <h3 style="margin-top:0; margin-bottom:0.5rem; color: #222222;">
                {risk_emoji} Delay risk: <b>{risk_label}</b> for flight
                <b>{selected_airline} {flight_num_input}</b>
              </h3>
              <p style="color: #222222;"><b>Model prediction:</b> {p_pct}% chance of delay ≥ 25 minutes.</p>
              <p style="color: #555555; font-size: 0.9em;"><i>(Based on historical average - schedule not available)</i></p>
              <p style="color: #222222;"><b>Historical behaviour:</b> delayed in {h_pct}% of past flights ({len(matches)} records).</p>
              <p style="color: #222222;"><b>Average delay (if late):</b> {avg_delay_round} min.</p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("###  Simulate delay risk with your own weather")

            st.write(
                "Adjust the weather conditions below to see how the model's prediction "
                "changes for this flight."
            )

            # Use latest historical record as base
            if "date" in matches.columns:
                base_row = matches.sort_values("date").iloc[-1].copy()
            else:
                base_row = matches.iloc[-1].copy()

            def _default(col, fallback):
                return float(matches[col].median()) if col in matches.columns else fallback
            temp_default = _default("temp", 20.0)
            precip_default = _default("precip", 0.0)
            windspeed_default = _default("windspeed", 15.0)
            visibility_default = _default("visibility", 10.0)
            cloud_default = _default("cloudcover", 50.0)

            col_w1, col_w2, col_w3 = st.columns(3)

            temp_val = col_w1.slider("Temperature (°C)", -10.0, 45.0, float(temp_default), 0.5)
            precip_val = col_w2.slider("Precipitation (mm)", 0.0, 30.0, float(precip_default), 0.5)
            windspeed_val = col_w3.slider("Wind speed (km/h)", 0.0, 80.0, float(windspeed_default), 1.0)

            col_w4, col_w5 = st.columns(2)
            visibility_val = col_w4.slider("Visibility (km)", 0.0, 20.0, float(visibility_default), 0.5)
            cloudcover_val = col_w5.slider("Cloud cover (%)", 0.0, 100.0, float(cloud_default), 1.0)

            if "cond_grouped" in matches.columns:
                cond_options = sorted(matches["cond_grouped"].dropna().unique().tolist())
            else:
                cond_options = ["clear", "cloudy", "rain", "storm", "fog"]

            cond_val = st.selectbox("Weather condition", cond_options)

            if st.button("Predict with this weather"):
                sim_row = base_row.copy()

                if "temp" in sim_row.index:
                    sim_row["temp"] = temp_val
                if "precip" in sim_row.index:
                    sim_row["precip"] = precip_val
                if "windspeed" in sim_row.index:
                    sim_row["windspeed"] = windspeed_val
                if "visibility" in sim_row.index:
                    sim_row["visibility"] = visibility_val
                if "cloudcover" in sim_row.index:
                    sim_row["cloudcover"] = cloudcover_val
                if "cond_grouped" in sim_row.index:
                    sim_row["cond_grouped"] = cond_val

                X_sim = sim_row[features].to_frame().T.copy()

                for c in cat_features:
                    if c in X_sim.columns:
                        X_sim[c] = X_sim[c].fillna("Unknown").astype(str)
                X_sim[[c for c in cat_features if c in X_sim.columns]] = \
                    X_sim[[c for c in cat_features if c in X_sim.columns]].astype(str)

                proba_sim = model.predict_proba(X_sim)[0, 1]
                proba_sim_pct = round(proba_sim * 100)

                if proba_sim >= 0.6:
                    sim_label = "High"
                    sim_emoji = "🔴"
                elif proba_sim >= 0.4:
                    sim_label = "Medium"
                    sim_emoji = "🟡"
                else:
                    sim_label = "Low"
                    sim_emoji = "🟢"

                sim_html = f"""
                <div style="
                    background-color:#f0f4ff;
                    border-left: 6px solid #3366ff;
                    padding: 0.75rem 1.25rem;
                    border-radius: 0.5rem;
                    margin-top: 1rem;
                    color: #222222;
                ">
                  <h4 style="margin-top:0; margin-bottom:0.5rem; color: #222222;">
                    {sim_emoji} With these weather conditions, delay risk is <b>{sim_label}</b>
                  </h4>
                  <p style="margin:0; color: #222222;">
                    The model predicts a delay ≥ 25 minutes with probability <b>{proba_sim_pct}%</b>
                    for flight <b>{selected_airline} {flight_num_input}</b> under the weather
                    you entered above.
                  </p>
                </div>
                """
                st.markdown(sim_html, unsafe_allow_html=True)

            st.markdown("#### Or use current weather at MAD")

            if st.button("Use current weather at MAD", key="btn_weather_flight"):
                weather, err = fetch_current_weather_mad()
                if err:
                    st.error(err)
                else:
                    st.info(
                        f"Current weather at MAD: {weather['temp']}°C, "
                        f"{weather['raw_conditions']} - "
                        f"visibility {weather['visibility']} km, "
                        f"cloud cover {weather['cloudcover']}%."
                    )

                    # Build a simulation row based on the latest historical row (base_row from above)
                    sim_row = base_row.copy()

                    # Overwrite weather features if present in the model’s feature set
                    for col in ["temp", "precip", "windspeed", "visibility", "cloudcover", "cond_grouped"]:
                        if col in sim_row.index and weather.get(col) is not None:
                            sim_row[col] = weather[col]

                    X_sim_live = sim_row[features].to_frame().T.copy()

                    for c in cat_features:
                        if c in X_sim_live.columns:
                            X_sim_live[c] = X_sim_live[c].fillna("Unknown").astype(str)
                    X_sim_live[[c for c in cat_features if c in X_sim_live.columns]] = \
                        X_sim_live[[c for c in cat_features if c in X_sim_live.columns]].astype(str)

                    proba_live = model.predict_proba(X_sim_live)[0, 1]
                    proba_live_pct = round(proba_live * 100)

                    if proba_live >= 0.6:
                        sim_label = "High"
                        sim_emoji = "🔴"
                    elif proba_live >= 0.4:
                        sim_label = "Medium"
                        sim_emoji = "🟡"
                    else:
                        sim_label = "Low"
                        sim_emoji = "🟢"

                    sim_live_html = f"""
                    <div style="
                        background-color:#f0f4ff;
                        border-left: 6px solid #3366ff;
                        padding: 0.75rem 1.25rem;
                        border-radius: 0.5rem;
                        margin-top: 1rem;
                        color: #222222;
                    ">
                    <h4 style="margin-top:0; margin-bottom:0.5rem; color: #222222;">
                        {sim_emoji} With the <b>current weather</b> at MAD, delay risk is <b>{sim_label}</b>
                    </h4>
                    <p style="margin:0; color: #222222;">
                        The model predicts a delay ≥ 25 minutes with probability <b>{proba_live_pct}%</b>
                        for flight <b>{selected_airline} {flight_num_input}</b>.
                    </p>
                    </div>
                    """
                    st.markdown(sim_live_html, unsafe_allow_html=True)

            st.markdown("---")
            with st.expander(" See detailed analysis (history, hours, table)"):
                tab_history, tab_hours, tab_table = st.tabs(
                    ["History over time", "By departure hour", "All records"]
                )

                with tab_history:
                    st.markdown("#### Historical delays & model probability over time")
                    if "date" in matches.columns:
                        df_hist = (
                            matches.sort_values("date")[["date", "is_delayed_25", "predicted_delay_prob"]]
                            .rename(
                                columns={
                                    "is_delayed_25": "Historical delay (0/1)",
                                    "predicted_delay_prob": "Model probability",
                                }
                            )
                        )

                        df_long = df_hist.melt(
                            id_vars="date",
                            value_vars=["Historical delay (0/1)", "Model probability"],
                            var_name="Series",
                            value_name="Value",
                        )
                        df_long["Value"] = df_long["Value"].astype(float)

                        fig = px.line(
                            df_long,
                            x="date",
                            y="Value",
                            color="Series",
                            labels={"date": "Date", "Value": "Value"},
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No `date` column available to show history over time.")

                with tab_hours:
                    st.markdown("####  Delay risk by scheduled departure hour")
                    if "dep_hour" in matches.columns:
                        by_hour = (
                            matches.groupby("dep_hour")
                            .agg(
                                historical_rate=(delay_col, "mean"),
                                model_prob=("predicted_delay_prob", "mean"),
                                samples=(delay_col, "size"),
                            )
                            .reset_index()
                        )

                        df_hour_long = by_hour.melt(
                            id_vars=["dep_hour", "samples"],
                            value_vars=["historical_rate", "model_prob"],
                            var_name="Series",
                            value_name="Value",
                        )
                        df_hour_long["Value"] = df_hour_long["Value"].astype(float)

                        fig_hour = px.line(
                            df_hour_long,
                            x="dep_hour",
                            y="Value",
                            color="Series",
                            labels={
                                "dep_hour": "Departure hour",
                                "Value": "Delay probability",
                            },
                        )
                        fig_hour.update_layout(height=350)
                        st.plotly_chart(fig_hour, use_container_width=True)

                        st.caption("Number of historical flights per hour:")
                        st.dataframe(
                            by_hour[["dep_hour", "samples"]]
                            .rename(columns={"dep_hour": "Hour", "samples": "Flights"}),
                            use_container_width=True,
                            height=200,
                        )
                    else:
                        st.info("No `dep_hour` column available.")

                with tab_table:
                    st.markdown("####  All historical flights for this flight")

                    cols_to_show = [
                        c
                        for c in [
                            "date",
                            "airline_name",
                            "flight_icao",
                            "flight_iata",
                            "flight_number",
                            "dep_hour",
                            "dep_terminal",
                            "dep_gate",
                            "arr_iata",
                            "dep_delay",
                            "is_delayed_25",
                            "predicted_delay_prob",
                            "cond_grouped",
                        ]
                        if c in matches.columns
                    ]

                    if "date" in matches.columns:
                        matches_sorted = matches.sort_values("date", ascending=False)
                    else:
                        matches_sorted = matches

                    st.dataframe(
                        matches_sorted[cols_to_show],
                        use_container_width=True,
                        height=420,
                    )

    st.markdown("---")
    st.markdown("###  Option 2: Check delay risk for a route (airline + destination)")

    st.write(
        "Use this when you want a general idea of delay risk for a route, even if we don't "
        "have your exact flight number."
    )
    st.caption("Destinations shown are based on routes this airline has operated in our historical data.")

    colr1, colr2 = st.columns(2)

    airline_route = colr1.selectbox(
        "Airline (route)",
        airlines,
        index=default_airline_idx,
        key="route_airline",
    )

    dest_choices = (
        df_filtered[df_filtered["airline_name"] == airline_route]["arr_iata"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )
    dest_choices = sorted(dest_choices)

    if dest_choices:
        dest_route = colr2.selectbox(
            "Destination airport (IATA)",
            dest_choices,
            key="route_destination",
        )

        df_route = df_filtered[
            (df_filtered["airline_name"] == airline_route)
            & (df_filtered["arr_iata"].astype(str).str.upper() == dest_route)
        ].copy()

        if df_route.empty:
            st.info("We don't have data for this airline + destination combination yet.")
        else:
            if st.button("Check route delay risk", key="check_route_btn"):
                n_flights = len(df_route)
                hist_delay_route = df_route["is_delayed_25"].mean()
                avg_dep_delay_route = df_route["dep_delay"].mean()

                if "model_delay_prob" in df_route.columns:
                    avg_model_prob_route = df_route["model_delay_prob"].mean()
                else:
                    avg_model_prob_route = float("nan")

                p = avg_model_prob_route
                h = hist_delay_route

                if p >= 0.6 or h >= 0.75:
                    risk_label = "High"
                    risk_emoji = "🔴"
                    card_color = "#ffe5e5"
                    border_color = "#ff4b4b"
                elif p >= 0.4 or h >= 0.5:
                    risk_label = "Medium"
                    risk_emoji = "🟡"
                    card_color = "#fff7d6"
                    border_color = "#ffcc00"
                else:
                    risk_label = "Low"
                    risk_emoji = "🟢"
                    card_color = "#e6ffe6"
                    border_color = "#33aa33"

                p_pct = round(p * 100)
                h_pct = round(h * 100)
                avg_delay_round = round(avg_dep_delay_route)

                route_html = f"""
                <div style="
                    background-color:{card_color};
                    border-left: 6px solid {border_color};
                    padding: 1rem 1.5rem;
                    border-radius: 0.5rem;
                    margin-top: 0.75rem;
                    color: #222222;
                ">
                  <h3 style="margin-top:0; margin-bottom:0.5rem; color: #222222;">
                    {risk_emoji} Route delay risk: <b>{risk_label}</b> for
                    <b>{airline_route} → {dest_route}</b>
                  </h3>
                  <p style="margin:0.25rem 0; color: #222222;">
                    Based on <b>{n_flights}</b> flights in our dataset.
                  </p>
                  <p style="margin:0.25rem 0; color: #222222;">
                    <b>Model prediction:</b> average delay probability ≈ <b>{p_pct}%</b>.
                  </p>
                  <p style="margin:0.25rem 0; color: #222222;">
                    <b>Historical behaviour:</b> delay ≥ 25 minutes on <b>{h_pct}%</b> of trips,
                    with average departure delay ≈ <b>{avg_delay_round} minutes</b>.
                  </p>
                </div>
                """
                st.markdown(route_html, unsafe_allow_html=True)

            st.markdown("####  Simulate delay risk for this route with your own weather")
            st.write(
                "Adjust the weather conditions below to see how the model's prediction "
                f"changes for the route **{airline_route} → {dest_route}**."
            )

            if "date" in df_route.columns:
                base_row_route = df_route.sort_values("date").iloc[-1].copy()
            else:
                base_row_route = df_route.iloc[-1].copy()

            def _default_route(col, fallback):
                return float(df_route[col].median()) if col in df_route.columns else fallback

            temp_default_r = _default_route("temp", 20.0)
            precip_default_r = _default_route("precip", 0.0)
            windspeed_default_r = _default_route("windspeed", 15.0)
            visibility_default_r = _default_route("visibility", 10.0)
            cloud_default_r = _default_route("cloudcover", 50.0)

            col_rw1, col_rw2, col_rw3 = st.columns(3)

            temp_val_r = col_rw1.slider(
                "Temperature (°C)", -10.0, 45.0, float(temp_default_r), 0.5,
                key="route_temp",
            )
            precip_val_r = col_rw2.slider(
                "Precipitation (mm)", 0.0, 30.0, float(precip_default_r), 0.5,
                key="route_precip",
            )
            windspeed_val_r = col_rw3.slider(
                "Wind speed (km/h)", 0.0, 80.0, float(windspeed_default_r), 1.0,
                key="route_wind",
            )

            col_rw4, col_rw5 = st.columns(2)
            visibility_val_r = col_rw4.slider(
                "Visibility (km)", 0.0, 20.0, float(visibility_default_r), 0.5,
                key="route_vis",
            )
            cloudcover_val_r = col_rw5.slider(
                "Cloud cover (%)", 0.0, 100.0, float(cloud_default_r), 1.0,
                key="route_cloud",
            )

            if "cond_grouped" in df_route.columns:
                cond_options_r = sorted(df_route["cond_grouped"].dropna().unique().tolist())
            else:
                cond_options_r = ["clear", "cloudy", "rain", "storm", "fog"]

            cond_val_r = st.selectbox(
                "Weather condition for this route",
                cond_options_r,
                key="route_cond",
            )

            if st.button("Predict route delay with this weather", key="route_weather_button"):
                sim_row_r = base_row_route.copy()

                if "temp" in sim_row_r.index:
                    sim_row_r["temp"] = temp_val_r
                if "precip" in sim_row_r.index:
                    sim_row_r["precip"] = precip_val_r
                if "windspeed" in sim_row_r.index:
                    sim_row_r["windspeed"] = windspeed_val_r
                if "visibility" in sim_row_r.index:
                    sim_row_r["visibility"] = visibility_val_r
                if "cloudcover" in sim_row_r.index:
                    sim_row_r["cloudcover"] = cloudcover_val_r
                if "cond_grouped" in sim_row_r.index:
                    sim_row_r["cond_grouped"] = cond_val_r

                X_sim_r = sim_row_r[features].to_frame().T.copy()

                for c in cat_features:
                    if c in X_sim_r.columns:
                        X_sim_r[c] = X_sim_r[c].fillna("Unknown").astype(str)
                X_sim_r[[c for c in cat_features if c in X_sim_r.columns]] = \
                    X_sim_r[[c for c in cat_features if c in X_sim_r.columns]].astype(str)

                proba_sim_r = model.predict_proba(X_sim_r)[0, 1]
                proba_sim_r_pct = round(proba_sim_r * 100)

                if proba_sim_r >= 0.6:
                    sim_label_r = "High"
                    sim_emoji_r = "🔴"
                elif proba_sim_r >= 0.4:
                    sim_label_r = "Medium"
                    sim_emoji_r = "🟡"
                else:
                    sim_label_r = "Low"
                    sim_emoji_r = "🟢"

                sim_html_r = f"""
                <div style="
                    background-color:#f0f4ff;
                    border-left: 6px solid #3366ff;
                    padding: 0.75rem 1.25rem;
                    border-radius: 0.5rem;
                    margin-top: 1rem;
                    color: #222222;
                ">
                  <h4 style="margin-top:0; margin-bottom:0.5rem; color: #222222;">
                    {sim_emoji_r} With these weather conditions, route delay risk is <b>{sim_label_r}</b>
                  </h4>
                  <p style="margin:0; color: #222222;">
                    The model predicts a delay ≥ 25 minutes with probability
                    <b>{proba_sim_r_pct}%</b> for the route
                    <b>{airline_route} → {dest_route}</b> under the weather you entered above.
                  </p>
                </div>
                """
                st.markdown(sim_html_r, unsafe_allow_html=True)

            if st.button(
                "Use current weather at MAD",
                key="btn_weather_route",
            ):
                weather, err = fetch_current_weather_mad()
                if err:
                    st.error(err)
                else:
                    st.info(
                        f"Current weather at MAD: {weather['temp']}°C, "
                        f"{weather['raw_conditions']} - "
                        f"visibility {weather['visibility']} km, "
                        f"cloud cover {weather['cloudcover']}%."
                    )

                    sim_row_live = base_row_route.copy()

                    for col in ["temp", "precip", "windspeed", "visibility", "cloudcover", "cond_grouped"]:
                        if col in sim_row_live.index and weather.get(col) is not None:
                            sim_row_live[col] = weather[col]

                    X_sim_live_r = sim_row_live[features].to_frame().T.copy()

                    for c in cat_features:
                        if c in X_sim_live_r.columns:
                            X_sim_live_r[c] = X_sim_live_r[c].fillna("Unknown").astype(str)
                    X_sim_live_r[[c for c in cat_features if c in X_sim_live_r.columns]] = \
                        X_sim_live_r[[c for c in cat_features if c in X_sim_live_r.columns]].astype(str)

                    proba_live_r = model.predict_proba(X_sim_live_r)[0, 1]
                    proba_live_r_pct = round(proba_live_r * 100)

                    if proba_live_r >= 0.6:
                        live_label_r = "High"
                        live_emoji_r = "🔴"
                    elif proba_live_r >= 0.4:
                        live_label_r = "Medium"
                        live_emoji_r = "🟡"
                    else:
                        live_label_r = "Low"
                        live_emoji_r = "🟢"

                    sim_live_html_r = f"""
                    <div style="
                        background-color:#f0f4ff;
                        border-left: 6px solid #3366ff;
                        padding: 0.75rem 1.25rem;
                        border-radius: 0.5rem;
                        margin-top: 1rem;
                        color: #222222;
                    ">
                      <h4 style="margin-top:0; margin-bottom:0.5rem; color: #222222;">
                        {live_emoji_r} With the <b>current weather</b> at MAD, route delay risk is
                        <b>{live_label_r}</b>
                      </h4>
                      <p style="margin:0; color: #222222;">
                        The model predicts a delay ≥ 25 minutes with probability
                        <b>{proba_live_r_pct}%</b> for the route
                        <b>{airline_route} → {dest_route}</b>.
                      </p>
                    </div>
                    """
                    st.markdown(sim_live_html_r, unsafe_allow_html=True)

    else:
        st.info(
            f"We don't have any destinations in the dataset for airline **{airline_route}**."
        )


def main():
    df_filtered = load_filtered()
    df_filtered = add_model_predictions(df_filtered)

    df_raw = load_raw_flights()
    df_arrivals = load_raw_arrivals()
    df_weather = load_weather()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["Overview", "Explorer", "Predict Delay"],
    )

    if page == "Overview":
        page_overview(df_filtered, df_raw, df_arrivals, df_weather)
    elif page == "Explorer":
        page_explorer(df_filtered)
    elif page == "Predict Delay":
        page_predict(df_filtered)

# Run the app
if __name__ == "__main__":
    main()
