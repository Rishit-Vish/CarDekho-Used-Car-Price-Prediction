CarDekho Used Car Price Prediction

A machine learning regression project that estimates used car selling prices from vehicle specifications and usage-related features. The project combines exploratory data analysis, data preprocessing, regression model comparison, Random Forest hyperparameter tuning, a FastAPI backend, and an interactive web interface.

Overview

Used car prices depend on several factors, including vehicle age, kilometres driven, fuel type, transmission, engine capacity, and maximum power. The objective of this project is to learn patterns from historical car listings and estimate the selling price of a vehicle from its available attributes.

This project covers the machine learning workflow from dataset exploration and model development to API integration, interactive predictions, and inference validation.

Project Highlights
Performed Exploratory Data Analysis (EDA) to understand feature distributions, relationships, missing values, skewness, and outliers.
Used SimpleImputer, ColumnTransformer, PowerTransformer, and scikit-learn's Pipeline to organize preprocessing and model training.
Applied winsorization to cap selected extreme values instead of automatically removing all outliers.
Compared five regression algorithms using five-fold cross-validation with MAE, RMSE, and R².
Evaluated Random Forest as the leading candidate and experimented with hyperparameter tuning.
Saved the trained model using Joblib for reuse during inference.
Built a FastAPI backend with Pydantic request validation and structured prediction responses.
Developed an interactive frontend using HTML, CSS, and JavaScript.
Investigated notebook-to-API prediction inconsistencies by comparing model outputs, inspecting dataset records, and testing multiple vehicle configurations.
Problem Statement

The goal is to estimate a used car's selling price using its specifications and usage-related features.

This is a supervised learning regression problem because the model learns from historical examples with known selling prices and predicts a continuous numerical value for a new vehicle.

The model provides an estimate based on patterns learned from the training data. It does not guarantee the actual market value of a vehicle.

Dataset

The dataset was obtained from Kaggle using KaggleHub.

Dataset: CarDekho Used Car Data
Download library: KaggleHub
Target variable: selling_price

The dataset was downloaded using:

import kagglehub

path = kagglehub.dataset_download(
    "manishkr1754/cardekho-used-car-data"
)

print("Path to dataset files:", path)

Features
Feature	Description
car_name	Name of the car
brand	Vehicle manufacturer
model	Car model
vehicle_age	Age of the vehicle in years
km_driven	Distance driven
seller_type	Seller category
fuel_type	Fuel category
transmission_type	Transmission category
mileage	Reported vehicle mileage
engine	Engine displacement in cc
max_power	Maximum engine power in bhp
seats	Seating capacity
selling_price	Target variable

The original dataset also contained an Unnamed: 0 column, which I reviewed during data cleaning because it was not intended to be a predictive feature.

Project Workflow
1. Exploratory Data Analysis (EDA)

I explored the dataset to understand its structure, data quality, feature distributions, and relationships before training the models.

The analysis included:

Inspecting dataset dimensions, data types, and statistical summaries.
Checking missing values across columns.
Examining numerical and categorical feature distributions.
Analysing skewness in numerical features.
Exploring correlations between numerical variables.
Investigating the relationship between vehicle age and selling price.
Comparing selling-price distributions across fuel types.
Reviewing unnecessary columns and potential outliers.

EDA helped me identify data-quality concerns and understand the characteristics of the dataset before deciding how to prepare it for modelling.

2. Data Cleaning and Preprocessing

I separated the input features (X) from the target variable (y) and split the data into training and test sets using an 80:20 split.

The preprocessing workflow used the following components:

SimpleImputer: Handles missing values according to the configured imputation strategy.
ColumnTransformer: Applies the configured transformations to the appropriate feature groups.
PowerTransformer: Transforms selected numerical features to help manage skewed distributions.
Pipeline: Combines preprocessing and model training into one workflow.

Keeping preprocessing inside the pipeline helps ensure that the same fitted transformations are applied during training and inference. During cross-validation, each pipeline is fitted separately within its training fold, helping prevent data leakage when the transformations are configured correctly.

3. Outlier Treatment Using Winsorization

I used winsorization to cap selected extreme values instead of automatically removing all outliers.

In used-car data, an unusually high selling price may represent a legitimate vehicle rather than an incorrect record. Removing every extreme observation could discard useful information about expensive vehicles.

Winsorization provides an alternative by limiting extreme values while retaining observations. However, its effectiveness depends on the selected thresholds, and aggressive caps can also affect legitimate high-value vehicles.

4. Regression Model Comparison

I compared five regression algorithms using scikit-learn:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
Extra Trees Regressor

I used KFold with five splits, shuffling enabled, and random_state=42. Each model was evaluated inside a preprocessing pipeline using cross_validate().

Evaluation Metrics
MAE (Mean Absolute Error): Measures the average absolute difference between actual and predicted prices. Lower values are better.
RMSE (Root Mean Squared Error): Penalizes larger prediction errors more heavily. Lower values are better.
R² (Coefficient of Determination): Measures model performance relative to predicting the mean target value. Higher values generally indicate better performance.

Scikit-learn returns negative values for its MAE and RMSE scorers. I negated the mean scores to report positive error values.

Cross-Validation Results

The following values were recorded during model comparison. They represent the mean scores across five validation folds on the training data.

Model	Mean MAE	Mean RMSE	Mean R²
Random Forest	102,045.23	331,845.03	0.864406
Extra Trees	103,826.58	341,359.03	0.854563
Gradient Boosting	127,297.73	360,886.76	0.841845
Decision Tree	128,486.42	384,922.81	0.822839
Linear Regression	182,851.16	507,835.63	0.692559

Random Forest achieved the lowest mean MAE and RMSE and the highest mean R² in this comparison. Based on these results, I selected it as the candidate for further evaluation and hyperparameter tuning.

The error metrics are expressed in the target variable's price unit. The dataset's currency and price scale should be confirmed before interpreting their magnitude.

5. Random Forest Baseline and Hyperparameter Tuning

After comparing the five models, I evaluated a baseline Random Forest and experimented with hyperparameter tuning to investigate whether adjusting the model configuration could improve performance.

The baseline pipeline used:

best_pipeline_wo_hpt = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])


Because n_estimators was not explicitly specified, this estimator used scikit-learn's default of 100 trees. In the earlier model-comparison experiment, Random Forest was configured with 200 trees.

Recorded Baseline Results

The baseline pipeline was fitted on X_train and evaluated on X_test, without cross-validation.

Metric	Recorded Baseline Result
MAE	98,486.40
RMSE	209,336.60
R²	0.927927

The MAE value was originally printed with an MSE label because the code used mean_absolute_error(). The calculation therefore represents MAE, not MSE.

These values are the results recorded during development.

Selected Hyperparameters

The recorded tuned configuration was:

Parameter	Value
n_estimators	200
min_samples_split	10
min_samples_leaf	1
max_depth	30
Recorded Tuning Results
Metric	Baseline	After Tuning
MAE	98,486.40	96,362.88
RMSE	209,336.60	209,230.96
R²	0.927927	0.928000

The recorded results show a small reduction in MAE and RMSE and a very small increase in R². This suggests that tuning produced only a modest improvement under the reported evaluation.

Understanding the Different R² Scores

The model-comparison results and baseline results were calculated using different evaluation procedures:

Model comparison: Five-fold cross-validation on X_train, with the reported metrics averaged across the five validation folds.
Baseline evaluation: The model was fitted on X_train and evaluated on the held-out X_test, without cross-validation.

Since the evaluation procedures and data being evaluated differ, their R² scores can also differ. The cross-validation R² of 0.864406 and the baseline test R² of 0.927927 are not directly comparable as if they came from the same evaluation.

For a reliable comparison of the baseline and tuned models, both should be evaluated on the same held-out test set using identical metric calculations and the same target scale. The reported results can also be reproduced to verify the consistency of the evaluation.

6. Model Serialization

I saved the trained model as best_random_forest.pkl using Joblib.

This allows the backend to load the fitted model for predictions without retraining it for every request.

The saved model contains the fitted preprocessing pipeline, so inference must use the expected feature names and feature ordering.

FastAPI Backend

I built a FastAPI backend to expose the trained model through HTTP endpoints.

The application loads the saved model, converts incoming vehicle specifications into a Pandas DataFrame, and passes the data to the model for prediction.

API Endpoints
Method	Endpoint	Purpose
GET	/options	Returns supported car names, brands, models, seller types, fuel types, transmission types, and the car catalog.
POST	/predict	Accepts vehicle specifications and returns the predicted selling price.
GET	/docs	Provides interactive API documentation through FastAPI.
Input Validation

I used Pydantic to define the prediction request schema.

The Features model validates required fields, data types, and numerical constraints. For example, vehicle age and kilometres driven must be non-negative integers, while mileage and maximum power must be non-negative numbers.

Categorical validators check that car names, brands, models, seller types, fuel types, and transmission types belong to the supported options loaded by the backend.

Prediction Response

A successful prediction returns a response in the following format:

{
  "predicted_selling_price": 720840.96
}


The example illustrates the response structure. The predicted selling price is intended to be expressed in Indian rupees.

The backend also uses Python logging to record unexpected prediction failures and returns a general HTTP 500 error message if a prediction fails.

Backend Modules
app.py — Initialises FastAPI, configures CORS, loads the model, defines endpoints, and handles prediction requests.
Config/config.py — Loads final_df.csv, defines the expected feature order, and builds the available vehicle options and catalog.
Schema/schemas.py — Defines the Pydantic request schema and input validators.
Schema/prediction_response.py — Defines the prediction response schema.
best_random_forest.pkl — Stores the trained model artifact.
final_df.csv — Supplies the vehicle data used to build the backend's available options.

The backend currently uses relative paths to load the model and CSV, so the working directory and file locations must match the application's expectations.

Interactive Web Interface

I built a web interface using HTML, CSS, and JavaScript to make the prediction workflow accessible without manually sending API requests.

The frontend communicates with the FastAPI backend:

GET /options supplies the available vehicle options.
POST /predict submits the selected vehicle specifications and receives a prediction.
The returned selling price can then be displayed to the user.

The frontend collects the input, FastAPI handles the request, and the fitted pipeline generates the prediction.

Debugging and Prediction Validation

During testing through FastAPI's interactive documentation, I encountered a prediction that appeared unusually high for the submitted vehicle specifications. I investigated the issue rather than assuming the model output was correct.

My checks included:

Comparing predictions from the notebook and the saved model.
Checking the loaded model path.
Inspecting matching Maruti Ertiga records in the dataset.
Comparing submitted specifications with corresponding records.
Testing ten Ertiga configurations.
Comparing predictions before and after the changes made during debugging.
Using logging and direct prediction comparisons to investigate the behaviour.

After the debugging changes, notebook and API predictions agreed for the inputs I tested, and the checked examples produced more reasonable results relative to the corresponding dataset records.

The exact root cause of the original mismatch was not conclusively established in my recorded investigation, so I have not attributed it to a specific cause.

This exercise reinforced the importance of checking the inference path, validating input consistency, and testing multiple examples when a prediction appears suspicious.

Agreement between notebook and API predictions checks inference consistency. Comparing predictions with actual selling prices helps assess prediction error. Neither check alone establishes general performance across every vehicle category.

Tech Stack
Language: Python
Data Analysis: Pandas, NumPy
Visualisation: Matplotlib, Seaborn
Machine Learning: Scikit-learn
Model Serialization: Joblib
Dataset Acquisition: KaggleHub
Backend: FastAPI
Input Validation: Pydantic
Frontend: HTML, CSS, JavaScript
Development Environment: Google Colab
Project Structure

The following structure reflects the backend files supplied for this project. The frontend and notebook paths should match the actual repository.

CarDekho-Used-Car-Price-Prediction/
├── app.py
├── Config/
│   └── config.py
├── Schema/
│   ├── schemas.py
│   └── prediction_response.py
├── best_random_forest.pkl
├── final_df.csv
├── requirements.txt
├── .gitignore
├── README.md
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── notebooks/
    └── car_price_prediction.ipynb


The frontend/ and notebooks/ directories are illustrative and should only be listed if those files exist at those locations.

Installation and Setup
Prerequisites
Python
Git
Project source code
The dataset and trained model artifact required by the backend
1. Clone the Repository
git clone https://github.com/Rishit-Vish/CarDekho-Used-Car-Price-Prediction.git
cd CarDekho-Used-Car-Price-Prediction

2. Create a Virtual Environment
python -m venv .venv


On Windows:

.venv\Scripts\activate


On macOS or Linux:

source .venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt


The requirements file should include the packages needed by the backend and the saved model's preprocessing pipeline.

4. Start the FastAPI Backend

If app.py is in the repository root, run:

uvicorn app:app --reload


Open the interactive API documentation at:

http://127.0.0.1:8000/docs

Make sure final_df.csv and best_random_forest.pkl are available at the paths expected by the application.

5. Start the Frontend

Follow the instructions for the actual frontend files in the repository. If the frontend consists of standalone HTML, CSS, and JavaScript files, serve them using a local development server.

Ensure that the frontend's API URL points to the running FastAPI backend. If the frontend and backend run on different origins, configure CORS to allow the frontend's origin.

Limitations
Predictions depend on the quality and coverage of the training data.
Actual resale prices can vary with vehicle condition, location, service history, market demand, and other factors that may not be represented in the available features.
Unusual vehicle configurations or price ranges may be less reliable when similar examples are scarce in the training data.
A high R² does not guarantee accurate predictions for every individual vehicle.
Cross-validation results and held-out test results come from different evaluation procedures and should be reported separately.
The baseline and tuned metrics should be reproduced on a common test set before drawing firm conclusions about the improvement from tuning.
The original notebook-to-API prediction mismatch was investigated, but its exact root cause was not conclusively documented.
Future Improvements
Reproduce baseline and tuned evaluation results using identical test data and consistent metric calculations.
Examine prediction errors across different price ranges and vehicle categories.
Add feature-importance analysis to understand which attributes contribute most to predictions.
Add automated tests for API validation and model inference.
Improve deployment configuration and make the application accessible through a hosted demo.
Add application screenshots and reproducible evaluation results.
What I Learned

This project gave me practical experience with the full machine learning workflow, from data exploration and preprocessing to model evaluation, serialization, and API integration.

It also taught me that a good evaluation score is only one part of a useful machine learning application. The inference pipeline must receive the expected features, the API must handle inputs correctly, and predictions need to be checked against realistic examples.

Investigating the notebook-to-API mismatch was particularly useful because it pushed me to compare the inference paths rather than assuming that a successful API response meant the prediction was correct.

Author

Rishit Vishwakarma

GitHub: CarDekho Used Car Price Prediction
