# Forecasting Taxi Demand in NYC - A Random Forest Approach

Development of a demand forecasting model for an American public transport company to optimize vehicle allocation. Using six months of GPS and vehicle start-time data, the goal is to predict hourly taxi demand in the ten most important city clusters. The project involves data quality assessment, clustering, regression modeling per cluster, error analysis, and proposing an integration strategy for daily logistic operations.

##  Objectives

- Analyze spatio-temporal taxi trip patterns in NYC.
- Identify demand hotspots through clustering techniques.
- Build and evaluate Random Forest regression models to predict hourly demand.
- Optimize model performance through feature engineering and hyperparameter tuning.
- Propose a deployment pipeline for practical use in public transport logistics.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
##  Project Workflow

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework to ensure a structured, iterative, and business-aligned machine learning pipeline.

---

### 1. Business Understanding

- Problem: How to efficiently allocate taxis based on forecasted demand?
- Goal: Predict demand per hour in the 10 busiest spatial clusters.
- Output: Cluster-specific demand forecasts with interpretable model performance.

---

### 2. Data Understanding

- Dataset: 4.5 million NYC taxi rides from April to September 2014.
- Fields: `datetime`, `latitude`, `longitude`, `base`.
- Quality: No missing values. Detected clear weekday/weekend and seasonal patterns.

---

### 3. Data Preparation

- Combined and cleaned monthly CSVs.
- Generated temporal features: `hour`, `weekday`, `month`, `is_weekend`, `is_holiday`.
- Weather data from KTEB0 merged but removed later due to degraded model performance.
- Converted coordinates to UTM for improved clustering.
- Created gapless time series by filling missing hours with zeros.

---

### 4. Clustering (MiniBatch K-Means)

- Clustered GPS data using MiniBatch K-Means on UTM coordinates.
- Selected top 10 clusters based on ride volume.
- Mapped clusters to NYC boroughs using GeoJSON shapes.
- Interactive visualization:  
  [View NYC Clusters Map (GitHub Pages)](https://sgbyteninja.github.io/nyc-taxi-demand-forecasting/nyc_clustered_sample_map.html)

---

### 5. Feature Engineering

- Generated lag and rolling features (`lag_1`, `lag_2`, ..., `lag_168` (one week) and `lag_672`lag_672 (one month)).
- Aggregated hourly demand per cluster.
- Built time series windows of 4 weeks to predict the next 24 hours.
- Final models used temporal features only (weather removed due to poor contribution).

---

### 6. Modeling and Evaluation

#### Baseline Modeling

- Random Forest Regressor trained using `hour`, `weekday` and `hour_index` only.
- Applied sliding window evaluation:
  - Train: Previous 4 weeks  
  - Forecast: Next 24 hours  
- RMSE used to compare performance across clusters.

#### Enriched Feature Models

- Same sliding window setup, now using full temporal feature set.
- Showed substantial improvement over the baseline in most clusters.

---

### 7. Hyperparameter Optimization (Grid Search)

- Grid search applied to tune `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`.
- Used **sliding window cross-validation** to preserve time order.
- Each cluster tuned independently due to varying demand patterns.

**Key Insights:**

- No single configuration worked best across all clusters.
- Most clusters used 100 trees; Cluster 9 required 200.
- Clusters 0, 2, 6, and 7 benefited from `max_depth=10`.
- Best models achieved RMSE between **4.48 (Cluster 3)** and **70.39 (Cluster 7)**.
- Optimal hyperparameters were saved and reused for final evaluation.

---

### 8. Final Model Evaluation (24-Hour Forecast)

Each cluster was trained on the last 4 weeks of data using the best-tuned model. The following 24 hours were forecasted and evaluated using:

- **RMSE** (Root Mean Square Error)
- **Daily Demand Error (%)**
- **MAPE** (Mean Absolute Percentage Error)  
  – shows relative error across hourly demand values (Hyndman & Koehler, 2006)

#### Key Results:

| Cluster | RMSE   | Daily Error % | MAPE %  |
|---------|--------|----------------|---------|
| 0       | 30.90  | 2.28           | 12.88   |
| 1       | 9.75   | 9.20           | 32.34   |
| 2       | 34.51  | 0.77           | 15.63   |
| 3       | 4.74   | 15.33          | 60.96   |
| 4       | 8.04   | 7.76           | 23.86   |
| 5       | 14.20  | 1.13           | 17.37   |
| 6       | 5.39   | 1.78           | 15.66   |
| 7       | 71.97  | 4.89           | 12.03   |
| 8       | 24.98  | 3.14           | 11.76   |
| 9       | 11.06  | 3.64           | 14.99   |

- High-volume clusters like 0 and 7 remained volatile but improved.
- Clusters 3 and 6 showed excellent performance with low RMSE.
- Most clusters achieved **<5% daily error**, proving practical reliability.
- MAPE revealed challenges in clusters with irregular or low volume (e.g. Cluster 3).

> The comparison confirms that **feature engineering alone is not sufficient** — combining engineered features with **systematic hyperparameter tuning** was key to achieving robust performance, especially in complex or volatile demand environments.

---

### 9. Deployment Proposal

- Automate daily ingestion of new GPS trip data.
- Retrain models weekly using last 4 weeks of history.
- Deploy API-based service (Flask/FastAPI) for predictions.
- Build interactive dashboard for planners (e.g. Streamlit).
- Consider ensemble methods or deep learning (e.g. LSTM) for long-term forecasting.

---

<small>Project based on the <a href="https://drivendata.github.io/cookiecutter-data-science/" target="_blank">cookiecutter data science</a> template.</small>