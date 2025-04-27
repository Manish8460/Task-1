
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Create Hypothetical Dataset
np.random.seed(42)

# 100 samples
data = pd.DataFrame({
    'age': np.random.randint(18, 65, 100),
    'income': np.random.randint(30000, 100000, 100),
    'gender': np.random.choice(['Male', 'Female'], 100),
    'purchased': np.random.choice([0, 1], 100)  # 0 = No, 1 = Yes
})

print("Sample Data:\n", data.head())

# Define Features (X) and Target (y)
X = data.drop('purchased', axis=1)
y = data['purchased']

# Handle categorical variables
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose task: 'classification' or 'regression'
task = 'classification'  # Change to 'regression' if needed

if task == 'regression':
    # Build Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("\nMean Squared Error:", mse)

elif task == 'classification':
    # Build Classification Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", acc)

else:
    print("Invalid task! Choose either 'regression' or 'classification'.")
