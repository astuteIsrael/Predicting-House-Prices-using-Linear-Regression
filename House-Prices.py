# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load and Explore the Dataset
# Assume you have a dataset named 'house_prices.csv'
data = pd.read_csv('house_prices.csv')
print(data.head())
print(data.describe())

# Step 3: Data Preprocessing
# Handling missing values (if any)
data = data.dropna()

# Convert categorical data (if any) using one-hot encoding
data = pd.get_dummies(data)

# Step 4: Exploratory Data Analysis (EDA)
# Visualizing relationships between features and price
sns.pairplot(data)
plt.show()

# Step 5: Building the Model
# Define features and target
X = data.drop('price', axis=1)  # Assuming 'price' is the target variable
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Model Evaluation
# Predict on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 7: Visualization
# Plotting the predictions vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
