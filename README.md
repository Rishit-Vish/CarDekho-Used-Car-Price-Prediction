# CarDekho Used Car Price Prediction

A machine learning regression project that estimates used car selling prices from vehicle specifications and usage-related features. The project combines exploratory data analysis, data preprocessing, regression model comparison, Random Forest hyperparameter tuning, a FastAPI backend, and an interactive web interface.

## Overview

Used car prices depend on several factors, including vehicle age, kilometres driven, fuel type, transmission, engine capacity, and maximum power. The objective of this project is to learn patterns from historical car listings and estimate the selling price of a vehicle from its available attributes.

This project covers the machine learning workflow from dataset exploration and model development to API integration, interactive predictions, and inference validation.

## Project Highlights

- Performed Exploratory Data Analysis (EDA) to understand feature distributions, relationships, missing values, and outliers.
- Used `SimpleImputer`, `ColumnTransformer`, `PowerTransformer`, and scikit-learn's `Pipeline` to organize preprocessing and model training.
- Applied winsorization to cap extreme values instead of automatically removing all outliers.
- Compared five regression algorithms using MAE, RMSE, and R².
- Tuned Random Forest hyperparameters and compared the results with the baseline model.
- Saved the selected model as a reusable artifact.
- Built a FastAPI backend with separate modules for configuration, request schemas, and prediction responses.
- Developed an interactive frontend using HTML, CSS, and JavaScript.
- Investigated prediction inconsistencies between the notebook and API using direct model comparisons, dataset inspection, logging, and multiple test examples.

## Problem Statement

The goal is to predict a used car's selling price using its available specifications and usage-related features.

This is a **supervised learning regression problem** because the model learns from historical examples with known prices and predicts a continuous numerical value for a new vehicle.

The model provides an estimated price based on learned patterns in the dataset. It is not intended to guarantee a vehicle's actual market value.

## Dataset

The dataset was obtained from Kaggle.

- **Dataset:** [CarDekho Used Car Data](https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data)
- **Download method:** KaggleHub

The dataset was downloaded using the following code:

```python
import kagglehub

path = kagglehub.dataset_download(
    "manishkr1754/cardekho-used-car-data"
)

print("Path to dataset files:", path)
```

### Features

| Feature | Description |
|---|---|
| `car_name` | Car name |
| `brand` | Vehicle manufacturer |
| `model` | Vehicle model |
| `vehicle_age` | Age of the vehicle |
| `km_driven` | Distance driven |
| `seller_type` | Seller category |
| `fuel_type` | Fuel category |
| `transmission_type` | Transmission category |
| `mileage` | Reported vehicle mileage |
| `engine` | Engine capacity |
| `max_power` | Maximum engine power |
| `seats` | Seating capacity |
| `selling_price` | Target variable |

The original dataset also contained an `Unnamed: 0` column, which was reviewed during data cleaning.

## Project Workflow

### 1. Dataset Inspection and Exploratory Data Analysis

The dataset was examined to understand its structure, data types, statistical summaries, missing values, and feature distributions.

The analysis included:

- Dataset structure and information analysis.
- Missing-value assessment.
- Univariate analysis of numerical and categorical features.
- Distribution and skewness analysis of numerical columns.
- Correlation analysis between numerical features.
- Scatter-plot analysis of vehicle age against selling price.
- Bar-plot analysis of selling price across fuel types.

These steps helped identify data-quality concerns and explore how vehicle characteristics relate to selling prices.

### 2. Data Cleaning and Preprocessing

The preprocessing stage included removing unnecessary columns, separating input features from the target variable, handling missing values, and preparing numerical and categorical data for modelling.

- **Missing-value handling:** `SimpleImputer` was used to handle missing values according to the configured imputation strategy.
- **Feature and target separation:** Vehicle attributes were assigned to `X`, while `selling_price` was used as the target variable, `y`.
- **Train-test split:** The data was divided into 80% training data and 20% test data. The held-out test set was used to evaluate predictions on data not used to fit the model.
- **Preprocessing pipeline:** `ColumnTransformer` and scikit-learn's `Pipeline` were used to organize feature-specific preprocessing and model training.
- **Feature transformation:** `PowerTransformer` was applied to numerical features to help manage skewed distributions.

Keeping learned preprocessing operations within the training pipeline helps prevent data leakage when those operations are fitted only on training data and then applied to the test data.

### 3. Outlier Treatment Using Winsorization

Winsorization was used to cap extreme values rather than automatically removing them.

The reasoning was that unusually high car prices may represent legitimate vehicles, including higher-end models. Removing every extreme observation could reduce the model's exposure to these parts of the market.

Capping extreme values was therefore explored as a way to limit their influence while retaining observations.

This approach also has limitations: legitimate high-value observations can be affected by the chosen caps, so the results depend on how the treatment is configured.

### 4. Regression Model Comparison

Five regression algorithms were evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor

The models were compared using three evaluation metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted prices.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily.
- **R² (Coefficient of Determination):** Measures how well the model explains variation in the target relative to a mean-prediction baseline.

#### Model Comparison Results

The following values are from the model-comparison evaluation recorded in the notebook.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 102,045.23 | 331,845.03 | 0.864406 |
| Extra Trees | 103,826.58 | 341,359.03 | 0.854563 |
| Gradient Boosting | 127,297.73 | 360,886.76 | 0.841845 |
| Decision Tree | 128,486.42 | 384,922.81 | 0.822839 |
| Linear Regression | 182,851.16 | 507,835.63 | 0.692559 |

Based on this evaluation, Random Forest achieved the lowest MAE and RMSE and the highest R² among the five models.

This made Random Forest the candidate selected for further experimentation and hyperparameter tuning.

> **Note:** The values above are reproduced from the reported comparison results. Error metrics are expressed in the target variable's price unit. Confirm the dataset's currency and price scale before interpreting their magnitude.

### 5. Random Forest: Baseline vs. Hyperparameter Tuning

The baseline Random Forest model was evaluated before tuning. Hyperparameter tuning was then performed to investigate whether adjusting the model configuration could improve performance.

#### Reported Results

| Metric | Baseline | After Tuning |
|---|---:|---:|
| MAE | 98,486.40 | 96,362.88 |
| RMSE | 209,336.60 | 209,230.96 |
| R² | 0.927927 | 0.928000 |

The recorded results show a small increase in R² and a small reduction in RMSE after tuning. However, the baseline MSE and RMSE values require verification because they do not satisfy the mathematical relationship between these two metrics.

The baseline MAE and tuned MSE were not included in the supplied results.

A fair final comparison should calculate MAE, RMSE, and R² for both models on the same held-out test set, using the same target scale and metric implementation.

#### Selected Hyperparameters

The selected Random Forest configuration was:

| Parameter | Value |
|---|---:|
| `n_estimators` | 200 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 1 |
| `max_depth` | 30 |

The tuned model was selected for use in the prediction application based on the model evaluation performed during development.

### 6. Model Serialization

The selected model was saved as a reusable artifact named `best_random_forest.pkl`.

This allowed the application to load the trained model for predictions without retraining it for every API request.

The inference path needs to use the same feature schema, feature ordering, and fitted preprocessing expected by the saved model.

### 7. FastAPI Backend

A FastAPI backend was developed to serve predictions from the trained model.

The application separates responsibilities into modules for configuration, input schemas, and prediction responses.

FastAPI's interactive documentation at `/docs` was used to submit vehicle specifications and inspect the returned predictions.

This provided a way to test the model through an API rather than relying exclusively on notebook execution.

### 8. Interactive Web Interface

An interactive frontend was built using HTML, CSS, and JavaScript.

Users can enter vehicle details through the interface and receive a predicted selling price from the FastAPI backend.

The frontend and API connect the trained model to a more accessible application experience.

### 9. Engineering Challenge: Notebook-to-API Prediction Consistency

During testing through FastAPI's interactive documentation, I encountered a prediction that appeared unusually high for a set of vehicle specifications. This prompted me to investigate the inference process instead of assuming the model output was correct.

I approached the issue by:

- Comparing predictions from the notebook and the saved model.
- Checking the loaded model path.
- Inspecting matching Ertiga records in the dataset.
- Comparing input features with their corresponding selling prices.
- Testing multiple Ertiga examples, including ten vehicle configurations.
- Checking predictions before and after the changes made during debugging.
- Using Python's logging facilities and direct prediction comparisons to investigate the behavior.

After this investigation, the notebook and API predictions were brought into agreement for the tested inputs, and the checked examples produced more reasonable results relative to the corresponding dataset records.

This experience reinforced the importance of validating model inference, checking input consistency, and testing an application across multiple examples rather than relying on a single successful prediction.

*The exact root cause of the original mismatch was not conclusively recorded, so this section describes the investigation and observed outcome without attributing the issue to a specific cause.*

### 10. Prediction Validation Using Ertiga Examples

A separate validation exercise used multiple Maruti Ertiga configurations drawn from the dataset.

The examples covered different vehicle ages, driven distances, fuel types, mileage values, and engine specifications.

These records were used to compare notebook predictions, API predictions, and known selling prices. This helped assess whether the application was processing representative inputs consistently and returning plausible results for the cases examined.

Agreement between notebook and API predictions checks inference consistency; comparison with actual selling prices helps assess predictive error. Neither check alone establishes general performance across every vehicle type.

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **Dataset Acquisition:** KaggleHub
- **Backend:** FastAPI
- **Frontend:** HTML, CSS, JavaScript
- **Development Environment:** Google Colab

## Project Structure

The following is a suggested structure. Update the filenames and paths to match the actual repository before publishing.

```text
CarDekho-Price-Prediction/
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── config.py
│   └── prediction_response.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── notebooks/
│   └── car_price_prediction.ipynb
├── artifacts/
│   └── best_random_forest.pkl
├── screenshots/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### Prerequisites

- Python
- Git
- Project source code and the required model artifact

### 1. Clone the Repository

```bash
git clone https://github.com/Rishit-Vish/CarDekho-Used-Car-Price-Prediction.git
cd CarDekho-Price-Prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI Backend

Use the module path that matches the actual location of the FastAPI application.

```bash
uvicorn <module_path>:app --reload
```

Open the interactive API documentation at:

[http://127.0.0.1:8000/docs]

### 5. Open the Frontend

Follow the frontend setup documented in the repository. If the frontend consists of standalone HTML, CSS, and JavaScript files, serve them locally using a suitable development server.

Make sure the frontend's API URL matches the address where FastAPI is running.

## Screenshots

<!-- TODO: Add application and API screenshots after uploading them to the screenshots/ directory. -->

<!-- Example:
![Application Interface](screenshots/app-interface.png)
![FastAPI Prediction Response](screenshots/api-prediction.png)
-->

## Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- Actual resale prices can vary with vehicle condition, location, market demand, service history, and other factors that may not be represented in the available features.
- Unusual vehicle configurations or price ranges may be less reliable when similar examples are scarce in the training data.
- A high R² does not guarantee that every individual prediction will be close to the actual selling price.
- Predictions are estimates, not guaranteed market valuations.

## Future Improvements

- Re-evaluate the baseline and tuned models using identical test data and consistent metric calculations.
- Examine performance across different price ranges and vehicle categories.
- Add feature-importance analysis to better understand which attributes influence predictions.
- Strengthen input validation and error handling for invalid or unrealistic specifications.
- Deploy the frontend and API so the application can be accessed without running it locally.

## What I Learned

This project gave me practical experience with the end-to-end machine learning workflow, from exploratory data analysis and preprocessing to model evaluation, model serialization, and API integration.

It also taught me the importance of validating predictions beyond the notebook, checking consistency between model development and inference, and investigating unexpected results through structured testing and logging.

Most importantly, it gave me the opportunity to take a machine learning model beyond experimentation and connect it to an interactive application.

---

**Author:** Rishit Vishwakarma

**GitHub:** (https://github.com/Rishit-Vish/CarDekho-Used-Car-Price-Prediction.git)
