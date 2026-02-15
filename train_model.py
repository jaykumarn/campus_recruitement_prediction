"""
Script to train the Ridge regression model and save it along with the scaler.
Run this whenever you need to regenerate model.pkl and scaler.pkl with a new sklearn version.
"""
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Load and preprocess data
df = pd.read_csv('train.csv')
df.drop('sl_no', axis=1, inplace=True)
df.drop_duplicates(inplace=True)
df['salary'] = df['salary'].fillna(0)

# Feature engineering
df['ssc_b_Central'] = df['ssc_b'].map({'Central': 1, 'Others': 0})
df['hsc_b_Central'] = df['hsc_b'].map({'Central': 1, 'Others': 0})
df['workex'] = df['workex'].map({'No': 0, 'Yes': 1})
df['status'] = df['status'].map({'Placed': 1, 'Not Placed': 0})
df['specialisation_fin'] = df['specialisation'].map({'Mkt&HR': 0, 'Mkt&Fin': 1})
df.drop(['ssc_b', 'hsc_b', 'specialisation'], axis=1, inplace=True)

# One-hot encoding
ohe = pd.get_dummies(df[['hsc_s', 'degree_t']], drop_first=True).astype(int)
df1 = pd.concat([ohe, df.drop(['hsc_s', 'degree_t'], axis=1)], axis=1)

# Prepare features and target
X = df1.drop('salary', axis=1)
y = df1['salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
ridge = Ridge(alpha=1)
ridge.fit(X_train_scaled, y_train)

print(f"Model R² score on test set: {ridge.score(X_test_scaled, y_test):.4f}")
print(f"Feature order: {list(X.columns)}")

# Save model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(ridge, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Saved model.pkl and scaler.pkl")
