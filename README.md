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

- Define the objective: Predict hourly taxi demand for the top 10 spatial clusters in NYC.
- Support goal: Assist transport logistics teams in allocating fleets based on expected demand.

---

### 2. Data Understanding

- Collected ~4.5 million taxi trips from April to September 2014.
- Data columns: `datetime`, `latitude`, `longitude`, `base` (cab center).
- Verified structure and completeness (no missing values).
- Detected time-based demand trends (e.g. weekdays vs. weekends, seasonal variation).

---

### 3. Data Preparation

- Merged all monthly CSV files into a unified dataset.
- Converted `datetime` to pandas `datetime` format.
- Created features: `hour`, `weekday`, `month`, `is_weekend`, `is_holiday`.
- Weather data from KTEB0 station merged via backward fill using `merge_asof`.
- Transformed GPS coordinates into **UTM** for better clustering accuracy.
- Ensured complete hourly time series by filling gaps with zero rides.

---

### 4. Clustering (MiniBatch K-Means)

- Applied MiniBatch K-Means on UTM-transformed coordinates.
- Optimal number of clusters determined via silhouette analysis.
- Selected the **top 10 clusters** by total ride volume.
- Visualized clusters using `Folium` on an interactive NYC map.
- Mapped clusters to dominant boroughs using GeoJSON files.

---

### 5. Feature Engineering

- Constructed additional time-series features:
  - `lag_1`, `lag_2`, `lag_3` (previous hour(s) ride count)
  - `rolling_mean_3`, `rolling_std_6` (3-hour moving average / 6-hour rolling std)
  - `is_weekend`, `is_holiday`, `month`, `weekday`
- Aggregated hourly demand per cluster.
- Prepared data windows (1–4 weeks) for model training/testing.

---

### 6. Initial Modeling with Random Forest

- Used **RandomForestRegressor** to predict hourly ride volume.
- Two model stages:
  - **Baseline Model**: using `hour` and `weekday` only.
  - **Advanced Feature Model**: with full feature set as described above.
- Applied **sliding window evaluation**:
  - Trained on prior 4 weeks, forecasted next 24 hours.
  - Evaluated using **RMSE** and daily demand error.

---

### 7. Advanced Model Optimization

- Performed **Grid Search** on Random Forest hyperparameters:
  - `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`
- Used **TimeSeriesSplit** (5 folds) to avoid temporal leakage.
- Conducted tuning separately for each cluster to capture unique patterns.
- Selected the best model per cluster based on **lowest RMSE** from validation.
- Observations:
  - Deep, unbootstrapped trees (`max_depth=None`, `bootstrap=False`) often performed best.
  - For some clusters (e.g., 0, 1, 2, 7, 9), a lower number of estimators (e.g., 100) was sufficient.
  - In many clusters, the advanced feature model without hyperparameter tuning performed better than the tuned one — highlighting the limits of parameter optimization when data variability is high.

---

### 8. Evaluation

- Forecasted demand for the final 24 hours using the last 4 weeks of data.
- Compared predicted vs. actual hourly demand for each cluster.
- **Best performance**:
  - Cluster 3: RMSE = 2.03, daily error = 0.36%
  - Cluster 6: RMSE = 2.95, daily error = 0.69%
- **Worst performance**:
  - Cluster 7 (largest/most volatile): RMSE = 42.21, daily error = 1.00%
- Weather features were excluded from final models as they slightly reduced performance.

---

### 9. Deployment Proposal

- Automate data ingestion and transformation pipelines.
- Schedule hourly model retraining using recent 4-week windows.
- Serve predictions via API (e.g., Flask or FastAPI).
- Build an interactive dashboard with Power BI or Streamlit for the logistics team.
- Compare with alternative models (e.g., LSTM, TBATS, NeuralProphet) for long-term or multi-seasonal demand forecasting.

---
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
