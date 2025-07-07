import pandas as pd
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned dataset
df = pd.read_csv('../data/Clean_Dataset.csv')
# Display the first few rows of the dataset
print(f"First 10 records",df.head(10))

# Display basic info
print("Initial shape:", df.shape)
print("Columns:", df.columns)   

# 1. Identify Missing Values:
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Missing Values:\n", missing_values)

# 2. Duplicate Rows:
duplicate_rows = df.duplicated().sum()
print("Duplicate Rows:", duplicate_rows)

# 3. Data Types:   
data_types = df.dtypes
print("Data Types:\n", data_types)

# 4. Summary Statistics:
summary_stats = df.describe()
print("Summary Statistics:\n", summary_stats)

# 5. Categorical Variables:
categorical_vars = df.select_dtypes(include=['object']).columns
print("Categorical Variables:\n", categorical_vars)

# 6. Numerical Variables:
numerical_vars = df.select_dtypes(include=['int64', 'float64']).columns
print("Numerical Variables:\n", numerical_vars)

# Label Encoding for Categorical Variables
label_encoders = {}
for col in categorical_vars:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Drop Price Column for Model Training
X = df.drop(columns=['price'])
y = df['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Save the model
with open('../model/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save features for Streamlit app
with open('../model/features.json', 'w') as features_file:
    json.dump(X.columns.tolist(), features_file)
