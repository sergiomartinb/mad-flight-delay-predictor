# MAD Flight Delay Tracker

This repository contains a **real-time predictive dashboard and ML inference pipeline** for departure delays at **Adolfo Suárez Madrid-Barajas (MAD) Airport**. It utilizes a custom CatBoost classifier to estimate the probability of a delay of 25 minutes or more, based on live operational and weather conditions, and is deployed via **Streamlit Community Cloud** for instant access.

<p align="center">
  <img src="https://github.com/javierferna/mad-flight-delay-tracker/blob/main/assets/landing-a350.gif?raw=true" 
    alt="Landing A350 GIF" width="500"/>
</p>

This project was made possible by the generous support of [AviationEdge](https://aviation-edge.com/), which provided us with a free unlimited API key for crucial real-time flight data integration.

---

**Key features:**

* **Custom CatBoost ML Model:** Predicts significant delays and assigns a **risk category (Low/Medium/High)**.
* **Live APIs for Flight + Weather:** Integrates flight data from **Aviation Edge & Aviationstack**, plus **Visual Crossing** for weather conditions.
* **Airport Congestion Modeling:** Engineered metrics like `hourly_arr_count` and `total_airport_load`.
* **Interactive Dashboard:** Explore historical patterns and simulate risk for upcoming flights through **Streamlit**.
* **Live Deployment:** Access the app directly through our custom [Streamlit Dashboard](https://madrid-airborne.streamlit.app/).

---

## Technologies Used

| Component         | Tool/Service                | Purpose                                          |
|------------------|-----------------------------|--------------------------------------------------|
| Model Engine     | `CatBoost Classifier`       | Predicts delay risk using engineered features    |
| Data Sources     | `Aviationstack`, `Aviation Edge`, `Visual Crossing` | Live flight & weather ingestion |
| Dashboard        | `Streamlit`                 | UI for visualization and prediction simulation   |
| Deployment       | `Streamlit Community Cloud` | Lightweight hosting & public access              |
| Feature Eng.     | Python / Pandas / NumPy     | Congestion metrics and preprocessing             |

