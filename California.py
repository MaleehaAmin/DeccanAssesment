import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle

california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
print(california.data.shape, california.target.shape)
# Display the first 5 rows of the dataset
print(df.head())

# Basic information about dataset
print(df.info())

# Summary statistics
print(df.describe())

df = df.iloc[:, :-2]

df['HouseVal'] = california.target

print(df.head())

print(df.isnull().sum())

#Filling missing values with the median (if any)
df.fillna(df.median(), inplace=True)


scaler = StandardScaler()
df_scaled = df.copy()

df_scaled.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
print(df_scaled.head())

correlation_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['HouseVal'], bins=30, kde=True)
plt.title("Distribution of House Value")
plt.xlabel("House Value")
plt.ylabel("Frequency")
plt.show()

# Choose two important features based on correlation
features_to_plot = ['MedInc', 'AveRooms']

# Scatter plots
plt.figure(figsize=(12, 5))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(1, 2, i)
    sns.scatterplot(x=df[feature], y=df['HouseVal'])
    plt.xlabel(feature)
    plt.ylabel('HouseVal')
    plt.title(f"{feature} vs HouseVal")
plt.show()





X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

print(X_train.head())



# Initialize and train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on test data
y_pred_lr = lr_model.predict(X_test)



# Initialize and train model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on test data
y_pred_dt = dt_model.predict(X_test)



# Initialize and train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf_model.predict(X_test)



# Initialize and train model
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict on test data
y_pred_xgb = xgb_model.predict(X_test)



# Function to evaluate models
def evaluate_model(y_test, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"#### {model_name} Performance:")
    print(f"🔹 RMSE: {rmse:.4f}")
    print(f"🔹 MAE: {mae:.4f}")
    print(f"🔹 R² Score: {r2:.4f}\n")

# Evaluate all models
evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_dt, "Decision Tree")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_xgb, "XGBoost")


param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Initialize Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Perform RandomizedSearchCV for optimization
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=20, 
    cv=5, 
    verbose=2,
    n_jobs=-1, 
    random_state=42
)

# Train with best hyperparameters
random_search.fit(X_train, y_train)

# Get best parameters
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

best_rf = RandomForestRegressor(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

# Predict on test set
y_pred_rf_optimized = best_rf.predict(X_test)

# Evaluate the optimized model
evaluate_model(y_test, y_pred_rf_optimized, "Optimized Random Forest")

# Save model to a file
with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(best_rf, file)

print("Model saved successfully as random_forest_model.pkl")