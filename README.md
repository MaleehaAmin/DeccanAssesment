# DeccanAssesment
1. Dataset Description
The dataset used is the California Housing Dataset from Scikit-learn, which is based on the 1990 U.S. Census data. It provides information about housing prices in different districts of California.

The dataset consists of various features:

MedInc represents the median income of households in the block group, measured in tens of thousands of dollars.
HouseAge indicates the median age of houses in the block group.
AveRooms is the average number of rooms per household.
AveBedrms is the average number of bedrooms per household.
Population represents the total population in the block group.
AveOccup is the average number of household members per home.
Latitude and Longitude represent the geographical location of the block group.
The target variable, MedHouseVal, represents the median house value for the block group, measured in hundreds of thousands of dollars.

2. Data Preprocessing & Feature Engineering
Exploratory Data Analysis (EDA)
The dataset was first examined for missing values, and none were found. Feature distributions were analyzed through visualizations to detect patterns and outliers. A correlation heatmap was plotted to observe relationships between features and their impact on house prices.

Feature Engineering
Unnecessary features, specifically Latitude and Longitude, were removed as they did not significantly improve predictions. All numerical features were scaled using StandardScaler to ensure a uniform distribution. Feature selection was performed by analyzing correlations and feature importance scores.

3. Model Selection & Performance Evaluation
Four regression models were trained: Linear Regression, Decision Tree, Random Forest, and XGBoost. The models were evaluated using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² score to measure their predictive performance.

Random Forest achieved the best performance with an R² score of 0.6775, making it the most suitable model for deployment. XGBoost performed similarly but was slightly less interpretable.

Hyperparameter Optimization
To improve the model’s accuracy, GridSearchCV was used to fine-tune the Random Forest hyperparameters. Adjustments were made to:

The number of estimators, increasing from 100 to 200.
The maximum depth of the trees, increasing from 10 to 15.
The minimum number of samples required to split a node, increasing from 2 to 5.
After optimization, the final trained model was saved using Pickle (random_forest_model.pkl) for deployment.

**The pkl file was too big to upload on github. Therefore, the link of this file is provided in the Model_weight_link.txt**
4. Deployment Strategy & API Usage
FastAPI Deployment
A FastAPI application was developed to serve predictions. The API loads the trained model and provides a /predict endpoint that accepts JSON input and returns the predicted house price.

To start the API, the following command was used:
uvicorn app:app --host 0.0.0.0 --port 8000

To test the API, a request can be sent via PowerShell:
$headers = @{ "Content-Type" = "application/json" }
$body = @{
    "MedInc" = 3.5
    "HouseAge" = 25
    "AveRooms" = 5.5
    "AveBedrms" = 1.0
    "Population" = 1200
    "AveOccup" = 3.0
} | ConvertTo-Json -Depth 10

$response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict" -Method Post -Headers $headers -Body $body
Write-Host $response.Content

5. Frontend Interface
A simple web UI was developed using HTML, CSS, and JavaScript to allow users to input house features and get predictions. The interface includes a form with input fields for each feature and a "Predict Price" button to fetch results from the API. The response is displayed dynamically on the page.
