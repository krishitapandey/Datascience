#  Movie Success Prediction & EDA Pipeline

A comprehensive data science workflow that analyzes movie metadata, cleans complex datasets, performs feature engineering, and trains machine learning models to predict high-quality movies.

## Overview

This project implements a full end-to-end machine learning pipeline on the [TMDB Movies Metadata](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) dataset. It covers everything from raw data ingestion to model training, evaluation, clustering, and API deployment.

---

##  Features

| # | Task |
|---|------|
| 1 | Load CSV & perform basic EDA (head, info, describe) |
| 2 | Handle missing values using multiple strategies |
| 3 | Detect and remove duplicate rows |
| 4 | Label Encoding & One-Hot Encoding |
| 5 | Visualizations: Bar, Line, Scatter charts |
| 6 | Identify and treat outliers using Boxplot + IQR |
| 7 | Feature Engineering (new derived columns) |
| 8 | Normalization and Standardization |
| 9 | Train/Test split |
| 10 | Logistic Regression model + evaluation |
| 11 | Decision Tree model + comparison |
| 12 | K-Means Clustering + Elbow Method |
| 13 | FastAPI for serving predictions |
| 14 | Web scraping with Requests + BeautifulSoup |
| 15 | Browser automation with Selenium |

---

##  Requirements

- Python 3.8+
- pip

### Python Libraries

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
fastapi
uvicorn
requests
beautifulsoup4
selenium
```

---



##  Dataset

Download `movies_metadata.csv` from Kaggle:

👉 [TMDB Movie Metadata – Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Place it in the **root directory** of the project:

```
movie-prediction-pipeline/
├── movies_metadata.csv   ← place here
├── main.py
├── requirements.txt
└── ...
```

> **Note:** The script checks for `movies_metadata.csv` at startup. If not found, it prints a warning and exits.



##  Pipeline Walkthrough

### 1. EDA (Exploratory Data Analysis)

```python
df = pd.read_csv('movies_metadata.csv')
print(df.head())
print(df.info())
print(df.describe())
```

Outputs dataset shape, memory usage, missing value counts, and duplicate count.

---

### 2. Missing Value Handling

Three strategies are applied:

| Strategy | Applied To | Method |
|----------|-----------|--------|
| Drop columns | Columns with >50% missing | `df.drop()` |
| KNN Imputer | Numeric columns | `KNNImputer(n_neighbors=5)` |
| Mode/Unknown | Categorical columns | `fillna(mode)` |

---

### 3. Duplicate Removal

```python
df_clean = df.drop_duplicates().reset_index(drop=True)
```

---

### 4. Encoding

- **Label Encoding** — applied to `original_language`, `status`
- **One-Hot Encoding** — applied to `release_month`

---

### 5. Visualizations

Four plots are generated automatically:

- **Bar Chart** — Top 10 movie languages
- **Line Chart** — Movie release trends over years
- **Scatter Plot** — Budget vs Revenue
- **Boxplot** — Runtime outlier detection

---

### 6. Outlier Treatment (IQR Capping / Winsorization)

```python
def treat_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower, upper)
    return df
```

Applied to: `runtime`, `popularity`, `vote_count`

---

### 7. Feature Engineering

New columns created:

| Column | Formula |
|--------|---------|
| `profit` | `revenue - budget` |
| `profit_margin` | `(profit / budget) * 100` |
| `roi` | `revenue / budget` |
| `budget_log` | `log1p(budget)` |
| `revenue_log` | `log1p(revenue)` |
| `release_year` | Extracted from `release_date` |
| `release_month` | Extracted from `release_date` |

**Target variable:** `is_hit` → `vote_average > 7.0`

---

### 8. Scaling

```python
# Standardization (zero mean, unit variance)
std_scaler = StandardScaler()
df_clean['runtime_std'] = std_scaler.fit_transform(df_clean[['runtime']])

# Normalization (0–1 range)
minmax_scaler = MinMaxScaler()
df_clean['popularity_norm'] = minmax_scaler.fit_transform(df_clean[['popularity']])
```

---

### 9. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

### 10–11. Model Training & Comparison

Three models are trained and compared:

| Model | Accuracy | AUC | CV Mean |
|-------|----------|-----|---------|
| Logistic Regression | — | — | — |
| Decision Tree | — | — | — |
| Random Forest | — | — | — |

> Actual values are printed to console and saved to `robust_results.csv` after running.

Each model outputs:
- Accuracy score
- ROC-AUC score
- 5-fold cross-validation mean ± std
- Confusion matrix heatmap

---

### 12. K-Means Clustering (Elbow Method)

```python
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method for Optimal K')
plt.show()
```

---

## ▶️ Usage

```bash
# Run the full pipeline
python main.py
```

The script will:
1. Check for `movies_metadata.csv`
2. Run all 12 steps automatically
3. Save outputs to disk
4. Display all plots

---

## 📦 Output Files

| File | Description |
|------|-------------|
| `data_cleaned.csv` | Cleaned & preprocessed dataset |
| `robust_results.csv` | Model accuracy/AUC comparison |
| `robust_pipeline.pkl` | Saved models, scaler, encoders |

---

## 🚀 API Usage (FastAPI)

### Start the server

```bash
uvicorn api:app --reload
```

### Predict endpoint

```
POST http://127.0.0.1:8000/predict
```

**Request body:**
```json
{
  "budget": 50000000,
  "revenue": 120000000,
  "runtime": 110,
  "popularity": 18.5,
  "vote_count": 3000,
  "original_language": "en",
  "status": "Released"
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.82,
  "label": "High Quality Movie"
}
```

---

## 🌐 Web Scraping

```bash
python scraper.py
```

Scrapes movie data using **Requests** + **BeautifulSoup** and exports it as `scraped_movies.csv`.

---

## 🤖 Browser Automation (Selenium)

```bash
python selenium_automation.py
```

Automates browser tasks such as login and data extraction using **Selenium WebDriver**.

> Ensure ChromeDriver is installed and matches your Chrome version.

---

## 📊 Results

After running `main.py`, the best model is printed:

```
 BEST: Random Forest (0.8423)
```

Full comparison is saved to `robust_results.csv`.

---


## 🙌 Acknowledgements

- Dataset: [TMDB Movies Metadata on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- Libraries: scikit-learn, pandas, seaborn, FastAPI, Selenium
