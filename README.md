MAD Flight Delay Predictor

An end-to-end Machine Learning solution to predict flight departure delays at **Adolfo Suárez Madrid-Barajas Airport (MAD)**. This project integrates historical flight data, real-time weather forecasts, and a gradient boosting model to estimate delay risks.

##  Project Overview

Flight delays cause significant disruptions to passengers and airlines. This project aims to predict the probability of a flight being delayed by **≥ 25 minutes** using a **CatBoost** classifier.

The solution is deployed as an interactive **Streamlit Dashboard** that allows users to:
* Explore historical delay patterns by airline and route.
* Predict delay risk for specific flights.
* **Live Simulation:** Connects to a Weather API to fetch real-time forecasts and adjust predictions dynamically.

## Project Structure

The project is organized into four logical blocks:

### 🔹 Block 1: Data Collection & ETL
* **APIs Used:** AviationEdge (Flight history) & Visual Crossing (Weather history).
* **Process:** Scripts to scrape historical departures/arrivals and hourly weather data for MAD.
* **Output:** Raw CSV datasets (`flights_data_raw.csv`, `weather_data_raw.csv`).

### 🔹 Block 2: EDA & Data Engineering
* **Cleaning:** Handling null values, parsing timestamps, and merging flight + weather data.
* **Feature Engineering:** Creation of temporal features (hour, month, day of week) and categorizing weather conditions.
* **Output:** Curated dataset ready for training (`filtered_flights_data.csv`).

### 🔹 Block 3: Predictive Modeling
* **Algorithm:** CatBoost Classifier (optimized for categorical features like Airlines/Destinations).
* **Target:** Binary classification (Delay ≥ 25 min).
* **Artifacts:** The trained model and feature lists are serialized (`.pkl`) for inference.

### 🔹 Block 4: Dashboard & Deployment(app.py)
* **Framework:** Streamlit.
* **Features:**
    * **Overview:** KPIs and Plotly charts for delay analysis.
    * **Explorer:** Filterable data tables.
    * **Predictor:** Real-time API integration to fetch forecast weather for the selected flight date/time.

## Technologies

* **Language:** Python 3.10+
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Plotly Express
* **Machine Learning:** CatBoost, Scikit-learn
* **Dashboard:** Streamlit
* **External APIs:** AviationEdge, Visual Crossing
