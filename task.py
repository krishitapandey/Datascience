import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer, KNNImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print(" Loading dataset...")
file_path = 'movies_metadata.csv'

if not os.path.exists(file_path):
    print(" File 'data.csv' is NOT AVAILABLE!")
    print("Please upload the CSV file named 'data.csv' to proceed with EDA.")
else:
    print(" File 'data.csv' found! Starting comprehensive EDA & Cleaning...")
    
    # Load dataset
    df = pd.read_csv('movies_metadata.csv')
    
    print("\n" + "="*80)
    print(" EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    
    # 1. BASIC OVERVIEW
    print(f"\n Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Total cells: {df.size:,}")
    
    # 2. HEAD
    print("\n First 5 rows:")
    print(df.head())
    
    # 3. INFO
    print("\n Data Info:")
    print(df.info())
    
    # 4. DESCRIBE
    print("\n Numerical Summary:")
    print(df.describe())
    
    print("\n Categorical Summary:")
    print(df.describe(include=['object']))
    
    # 5. MEMORY USAGE
    print(f"\n Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 6. MISSING VALUES
    print("\n Missing Values Analysis:")
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing_data,
        'Missing_%': missing_pct.round(2)
    }).sort_values('Missing_Count', ascending=False)
    
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    # 7. DUPLICATES
    print("\n Duplicates Analysis:")
    dup_count = df.duplicated().sum()
    print(f"   Duplicate rows: {dup_count:,} ({dup_count/len(df)*100:.2f}%)")
    
    print("\n" + "="*80)
    print(" DATA CLEANING OPERATIONS")
    print("="*80)
    
    df_clean = df.copy()
    
    # STEP 1: Remove Duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f" Removed {initial_rows - len(df_clean):,} duplicate rows")
    
    # STEP 2: Handle Missing Values (Multiple Strategies)
    print("\n Missing Value Strategies Applied:")
    
    # Strategy A: Drop columns with >50% missing
    high_missing_cols = missing_df[missing_df['Missing_%'] > 50].index.tolist()
    if high_missing_cols:
        df_clean.drop(columns=high_missing_cols, inplace=True)
        print(f"   Dropped {len(high_missing_cols)} columns (>50% missing): {high_missing_cols}")
    
    # Strategy B: Numerical columns - Median imputation
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"    '{col}': Filled {df[col].isnull().sum()} NaNs with median ({median_val})")
    
    # Strategy C: Categorical columns - Mode imputation
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            df_clean[col].fillna(fill_val, inplace=True)
            print(f"  '{col}': Filled {df[col].isnull().sum()} NaNs with '{fill_val}'")
    
    # Final check
    final_missing = df_clean.isnull().sum().sum()
    print(f"\n Final missing values: {final_missing:,}")
    
    # SAVE CLEANED DATA
    df_clean.to_csv('data_cleaned.csv', index=False)
    
    print("\n" + "="*80)
    print(" CLEANING SUMMARY")
    print("="*80)
    print(f" Original:     {df.shape}")
    print(f" Cleaned:      {df_clean.shape}")
    print(f" Rows removed:  {df.shape[0] - df_clean.shape[0]:,}")
    print(f" Cols removed:  {df.shape[1] - df_clean.shape[1]}")
    print(f"Cleaned file saved as 'data_cleaned.csv'")
    
    print("\n Cleaned Data Preview:")
    print(df_clean.head())


def robust_numeric_conversion(df, numeric_cols):
   
    for col in numeric_cols:
        if col in df.columns:
            # Replace URLs and strings with NaN
            df[col] = df[col].astype(str).str.replace(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\$\\$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '0', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Define and clean numeric columns FIRST
numeric_columns = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']
df = robust_numeric_conversion(df, numeric_columns)


print("\n MISSING DATA ANALYSIS (Post-Cleaning)")
print("="*60)

missing_df = pd.DataFrame({
    'column': df.columns,
    'missing_count': df.isnull().sum(),
    'missing_pct': (df.isnull().sum() / len(df)) * 100
}).sort_values('missing_pct', ascending=False)

print(" Missing Data Summary:")
print(missing_df[missing_df['missing_pct'] > 0].round(2))



print("\n MULTIPLE IMPUTATION STRATEGIES...")

cat_columns = ['original_language', 'status']
print(" Applying KNN Imputer for numeric...")
knn_imputer = KNNImputer(n_neighbors=5)
df[numeric_columns] = pd.DataFrame(
    knn_imputer.fit_transform(df[numeric_columns]),
    columns=numeric_columns,
    index=df.index
)

# Strategy 2: NUMERIC - Median fallback
print(" Applying Median Imputer...")
num_imputer = SimpleImputer(strategy='median')
df[numeric_columns] = pd.DataFrame(
    num_imputer.fit_transform(df[numeric_columns]),
    columns=numeric_columns
)

# Strategy 3: CATEGORICAL - Mode + 'Unknown'
print(" Applying Mode Imputer for categorical...")
for col in cat_columns:
    if col in df.columns:
        mode_val = df[col].mode()
        df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')

# Handle other high-missing columns strategically
high_missing_cols = ['homepage', 'tagline', 'overview', 'poster_path']
for col in high_missing_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Missing')

print(" All missing data handled!")

print("\n DUPLICATE DETECTION")
print("="*60)

exact_dups = df.duplicated().sum()
print(f"Exact duplicates: {exact_dups}")

df_clean = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {exact_dups} duplicates. Shape: {df_clean.shape}")

print("\nFEATURE ENGINEERING")
print("="*60)

df_clean['release_date'] = pd.to_datetime(df_clean['release_date'], errors='coerce')
df_clean['release_year'] = df_clean['release_date'].dt.year.fillna(2000)
df_clean['release_month'] = df_clean['release_date'].dt.month.fillna(7)

df_clean['profit'] = df_clean['revenue'] - df_clean['budget']
df_clean['profit_margin'] = np.where(df_clean['budget'] > 0,
                                   (df_clean['profit'] / df_clean['budget']) * 100, 0)
df_clean['roi'] = np.where(df_clean['budget'] > 0, df_clean['revenue'] / df_clean['budget'], 0)
df_clean['budget_log'] = np.log1p(df_clean['budget'])
df_clean['revenue_log'] = np.log1p(df_clean['revenue'])

# Target: High quality movies (top 20%)
y = (df_clean['vote_average'] >= df_clean['vote_average'].quantile(0.8)).astype(int)
print(f" High Quality movies: {y.mean():.1%}")

print("\n DUAL ENCODING")
print("="*60)

# Label Encoding (low cardinality)
le_cols = ['original_language', 'status']
label_encoders = {}
for col in le_cols:
    if col in df_clean.columns:
        le = LabelEncoder()
        df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le

# One-Hot Encoding (release_month)
df_encoded = pd.get_dummies(df_clean[['release_month']], prefix='month', drop_first=True)
df_clean = pd.concat([df_clean, df_encoded], axis=1)

print(f" Label encoded: {le_cols}")
print(f"One-hot encoded: {list(df_encoded.columns)}")


scale_cols = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average',
              'vote_count', 'profit', 'profit_margin', 'roi', 'release_year',
              'budget_log', 'revenue_log']
scale_cols += [f'{col}_encoded' for col in le_cols if f'{col}_encoded' in df_clean.columns]

# Final cleaning
df_clean[scale_cols] = df_clean[scale_cols].fillna(df_clean[scale_cols].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[scale_cols])

# Add one-hot features
ohe_cols = [col for col in df_clean.columns if col.startswith('month_')]
X_ohe = df_clean[ohe_cols].fillna(0).values

X = np.hstack([X_scaled, X_ohe])
print(f" Feature matrix shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n TRAINING MODELS...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10)
}

model_results = {}
predictions = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    model_results[name] = {
        'Accuracy': acc,
        'AUC': auc,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std()
    }
    predictions[name] = (y_pred, y_prob)

    print(f"  Acc: {acc:.4f}, AUC: {auc:.4f}, CV: {cv_scores.mean():.4f}")
def robust_numeric_conversion(df, numeric_cols):
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

numeric_columns = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']
df = robust_numeric_conversion(df, numeric_columns)
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df_clean = df.dropna(subset=['release_date']).copy()


print("\n GENERATING VISUALIZATIONS...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# BAR CHART: Top Languages
df_clean['original_language'].value_counts().head(10).plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Top 10 Movie Languages (Bar Chart)')

# LINE CHART: Movies Released Over Years
df_clean['year'] = df_clean['release_date'].dt.year
df_clean.groupby('year').size().plot(kind='line', ax=axes[0,1], color='orange', linewidth=2)
axes[0,1].set_title('Release Trends Over Time (Line Chart)')

# SCATTER PLOT: Budget vs Revenue
sns.scatterplot(data=df_clean, x='budget', y='revenue', alpha=0.5, ax=axes[1,0])
axes[1,0].set_title('Budget vs Revenue (Scatter Plot)')

# BOXPLOT: Identifying Outliers in Runtime
sns.boxplot(x=df_clean['runtime'], ax=axes[1,1], color='lightgreen')
axes[1,1].set_title('Identifying Outliers in Runtime (Boxplot)')

plt.tight_layout()
plt.show()

print("\n TREATING OUTLIERS...")

def treat_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Capping (Winsorization) - safer than dropping
    df[column] = np.where(df[column] > upper_bound, upper_bound,
                 np.where(df[column] < lower_bound, lower_bound, df[column]))
    return df

for col in ['runtime', 'popularity', 'vote_count']:
    df_clean = treat_outliers(df_clean, col)
print("\n  APPLYING SCALING...")


std_scaler = StandardScaler()
df_clean['runtime_std'] = std_scaler.fit_transform(df_clean[['runtime']])

minmax_scaler = MinMaxScaler()
df_clean['popularity_norm'] = minmax_scaler.fit_transform(df_clean[['popularity']])

print(f"Standardized Runtime Mean: {df_clean['runtime_std'].mean():.2f}")
print(f"Normalized Popularity Max: {df_clean['popularity_norm'].max():.2f}")

df_clean['is_hit'] = (df_clean['vote_average'] > 7.0).astype(int)

le = LabelEncoder()
df_clean['lang_encoded'] = le.fit_transform(df_clean['original_language'].astype(str))


features = ['runtime_std', 'popularity_norm', 'lang_encoded', 'budget', 'revenue']

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(df_clean[features])
y = df_clean['is_hit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Final Model Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\n Process Complete!")
results_df = pd.DataFrame(model_results).T
print("\n FINAL RESULTS:")
print(results_df.round(4))

# Save everything
joblib.dump({
    'models': models,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'knn_imputer': knn_imputer,
    'df_clean': df_clean
}, 'robust_pipeline.pkl')

results_df.to_csv('robust_results.csv')
print("\n ROBUST PIPELINE COMPLETE!  All files saved!")
print(f" BEST: {results_df['Accuracy'].idxmax()} ({results_df['Accuracy'].max():.4f})")
