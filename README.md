<div align="center">

# 🚗 CarDekho Used Car Price Prediction

**An end-to-end machine learning project: from raw data to a live prediction API and web interface.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

</div>

---

## 📌 Table of Contents

- [TL;DR](#-tldr)
- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Model Comparison](#-model-comparison)
- [Hyperparameter Tuning](#-random-forest-baseline-and-hyperparameter-tuning)
- [Why Two Different R² Scores?](#-why-do-i-report-two-different-r-scores)
- [FastAPI Backend](#-fastapi-backend)
- [Web Interface](#-interactive-web-interface)
- [Case Study: Notebook-to-API Mismatch](#-case-study-debugging-a-notebook-to-api-prediction-mismatch)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [What I Learned](#-what-i-learned)
- [Author](#-author)

---

## ⚡ TL;DR

| | |
|---|---|
| **Goal** | Estimate the selling price of a used car from its specifications |
| **Type** | Supervised learning, regression |
| **Best model** | Random Forest Regressor |
| **Cross-validated R²** | **0.864** (5-fold, training data) |
| **Held-out test R²** | **0.928** (80:20 split) |
| **Test MAE** | **≈ ₹96,363** (tuned model) |
| **Deployment** | FastAPI backend + HTML/CSS/JS frontend |
| **Validation** | Investigated a suspicious API prediction and verified it against the notebook and dataset records |

---

## 📖 Overview

Used car prices depend on several factors: vehicle age, kilometres driven, fuel type, transmission, engine capacity, and maximum power. This project learns those patterns from historical CarDekho listings and uses them to estimate the selling price of a vehicle.

I built the full workflow myself:

1. Exploratory data analysis
2. Preprocessing with scikit-learn pipelines
3. Comparison of five regression models
4. Random Forest tuning
5. Model serialization
6. A validated REST API
7. A web interface for interactive predictions
8. Inference validation, where I tracked down a suspicious API prediction

### ✨ Project Highlights

- Performed **EDA** on distributions, relationships, missing values, skewness, and outliers.
- Built a clean preprocessing workflow with `SimpleImputer`, `ColumnTransformer`, `PowerTransformer`, and `Pipeline`.
- Used **winsorization** to cap extreme values instead of blindly deleting outliers.
- Compared **5 regression algorithms** using 5-fold cross-validation (MAE, RMSE, R²).
- Selected **Random Forest** as the leading model and tuned its hyperparameters.
- Saved the trained pipeline with **Joblib** for reuse at inference time.
- Built a **FastAPI** backend with **Pydantic** request validation.
- Developed an interactive frontend using **HTML, CSS, and JavaScript**.
- Investigated a **notebook-vs-API prediction mismatch** instead of assuming the output was correct ([see case study](#-case-study-debugging-a-notebook-to-api-prediction-mismatch)).

---

## 🎯 Problem Statement

Estimate a used car's selling price from its specifications and usage-related features.

This is a **supervised regression problem**: the model learns from historical listings with known selling prices and predicts a continuous value for a new vehicle.

> ⚠️ The model gives an *estimate* based on patterns in the training data. It does not guarantee the actual market value of a vehicle.

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Name** | CarDekho Used Car Data |
| **Source** | [Kaggle: manishkr1754/cardekho-used-car-data](https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data) |
| **Download method** | KaggleHub |
| **Target variable** | `selling_price` |

```python
import kagglehub

path = kagglehub.dataset_download("manishkr1754/cardekho-used-car-data")
print("Path to dataset files:", path)
```

### Features

| Feature | Description |
|---|---|
| `car_name` | Name of the car |
| `brand` | Vehicle manufacturer |
| `model` | Car model |
| `vehicle_age` | Age of the vehicle in years |
| `km_driven` | Distance driven |
| `seller_type` | Seller category |
| `fuel_type` | Fuel category |
| `transmission_type` | Transmission category |
| `mileage` | Reported vehicle mileage |
| `engine` | Engine displacement (cc) |
| `max_power` | Maximum engine power (bhp) |
| `seats` | Seating capacity |
| `selling_price` | **Target variable** |

> The original dataset also had an `Unnamed: 0` column. I reviewed it during cleaning, as it is not a predictive feature.

---

## 🔄 Project Workflow

```
Raw Data → EDA → Cleaning & Preprocessing → Model Comparison → Tuning → Save Model → FastAPI → Web UI → Validation
```

### 1. Exploratory Data Analysis (EDA)

Before modelling, I explored the dataset to understand its structure, data quality, and relationships:

- Dataset dimensions, data types, and statistical summaries
- Missing values across columns
- Numerical and categorical feature distributions
- Skewness in numerical features
- Correlations between numerical variables
- Relationship between **vehicle age and selling price**
- Selling-price distributions across **fuel types**
- Unnecessary columns and potential outliers

### 2. Data Cleaning and Preprocessing

I separated the features (`X`) from the target (`y`) and split the data **80:20** into training and test sets.

| Component | Role |
|---|---|
| `SimpleImputer` | Handles missing values using the configured strategy |
| `ColumnTransformer` | Applies the right transformation to each feature group |
| `PowerTransformer` | Reduces skew in selected numerical features |
| `Pipeline` | Combines preprocessing and the model into one workflow |

Keeping preprocessing **inside the pipeline** ensures the same fitted transformations are applied at training and inference time. During cross-validation, each pipeline is fitted separately inside its training fold, which helps prevent data leakage.

### 3. Outlier Treatment with Winsorization

I used **winsorization** to cap selected extreme values rather than removing all outliers.

**Why?** In used-car data, a very high selling price is often a *legitimate* expensive vehicle, not an error. Deleting every extreme row would throw away useful information about premium cars. Winsorization limits their influence while keeping the observations.

> Trade-off: the result depends on the chosen thresholds, and aggressive caps can affect legitimate high-value vehicles.

---

## 🏆 Model Comparison

I compared five regression algorithms from scikit-learn:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor

**Setup:** `KFold` with 5 splits, shuffling enabled, `random_state=42`. Each model was wrapped in the preprocessing pipeline and evaluated with `cross_validate()`.

### Evaluation Metrics

| Metric | Meaning | Better |
|---|---|---|
| **MAE** | Average absolute difference between actual and predicted prices | Lower |
| **RMSE** | Like MAE, but penalises large errors more heavily | Lower |
| **R²** | Performance relative to simply predicting the mean price | Higher |

> scikit-learn returns MAE and RMSE as negative scores. I negated the mean scores to report positive error values.

### Cross-Validation Results

Mean scores across 5 validation folds on the **training data**:

| Rank | Model | Mean MAE | Mean RMSE | Mean R² |
|:---:|---|---:|---:|---:|
| 🥇 | **Random Forest** | **102,045.23** | **331,845.03** | **0.8644** |
| 🥈 | Extra Trees | 103,826.58 | 341,359.03 | 0.8546 |
| 🥉 | Gradient Boosting | 127,297.73 | 360,886.76 | 0.8418 |
| 4 | Decision Tree | 128,486.42 | 384,922.81 | 0.8228 |
| 5 | Linear Regression | 182,851.16 | 507,835.63 | 0.6926 |

**Takeaway:** Random Forest achieved the lowest MAE and RMSE and the highest R², so I selected it for further evaluation and tuning. Linear Regression performed noticeably worse, which suggests the price relationship is non-linear.

> Error metrics are in the target's price unit (Indian rupees).

---

## 🔧 Random Forest Baseline and Hyperparameter Tuning

After picking Random Forest, I evaluated a baseline and then experimented with tuning.

```python
best_pipeline_wo_hpt = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])
```

> Since `n_estimators` was not specified, the baseline used scikit-learn's default of **100 trees**. (The earlier model comparison used 200 trees.)

### Selected Hyperparameters

| Parameter | Value |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 30 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 1 |

### Baseline vs Tuned (Held-out Test Set)

Both models were fitted on `X_train` and evaluated on the held-out `X_test`.

| Metric | Baseline | After Tuning | Change |
|---|---:|---:|---:|
| **MAE** | 98,486.40 | 96,362.88 | ↓ 2,123.52 |
| **RMSE** | 209,336.60 | 209,230.96 | ↓ 105.64 |
| **R²** | 0.927927 | 0.928000 | ↑ 0.000073 |

**Takeaway:** Tuning gave only a **modest** improvement. The baseline Random Forest was already strong, so most of the performance came from the model choice and preprocessing rather than from hyperparameters.

> 📝 *Note: the baseline MAE was originally printed with an "MSE" label in my notebook because the code used `mean_absolute_error()`. The values above are correctly labelled as MAE.*

---

## ❓ Why Do I Report Two Different R² Scores?

You will see **R² = 0.8644** and **R² = 0.9279** in this README. They are *not* a contradiction. They come from two different evaluation procedures:

| | Model Comparison | Baseline Evaluation |
|---|---|---|
| **R² reported** | **0.8644** | **0.9279** |
| **Method** | 5-fold cross-validation | Single train/test split |
| **Cross-validation?** | ✅ Yes | ❌ No |
| **Evaluated on** | Validation folds inside `X_train` | Held-out `X_test` |
| **Hyperparameter tuning?** | No (comparison stage) | Baseline: No / Tuned: Yes |
| **Number of trees** | 200 | 100 (baseline) |
| **Purpose** | Fairly compare 5 models | Check how the chosen model performs on unseen data |

Because the data being scored and the procedure are different, the two numbers should **not** be compared as if they came from the same experiment.

- Use the **cross-validation scores** to compare *models against each other*.
- Use the **held-out test scores** to compare *baseline vs tuned* Random Forest.

> For the most reliable conclusion on tuning, both models should be evaluated on the same test set with identical metric calculations. This is on my improvement list below.

---

## ⚙️ FastAPI Backend

The backend exposes the trained model through HTTP endpoints. It loads the saved model, converts incoming specifications into a Pandas DataFrame, and returns a prediction.

### API Endpoints

| Method | Endpoint | Purpose |
|:---:|---|---|
| `GET` | `/options` | Returns supported car names, brands, models, seller types, fuel types, transmission types, and the car catalog |
| `POST` | `/predict` | Accepts vehicle specifications and returns the predicted selling price |
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |

### Input Validation (Pydantic)

The `Features` schema validates every request:

- **Required fields and data types** are enforced.
- `vehicle_age` and `km_driven` must be **non-negative integers**.
- `mileage` and `max_power` must be **non-negative numbers**.
- **Categorical fields** (car name, brand, model, seller type, fuel type, transmission type) must belong to the options loaded by the backend.

### Example Response

```json
{
  "predicted_selling_price": 720840.96
}
```

The price is expressed in **Indian rupees (₹)**. Unexpected failures are logged with Python's `logging` module, and the API returns a generic HTTP 500 message.

### Backend Modules

| File | Responsibility |
|---|---|
| `app.py` | Initialises FastAPI, configures CORS, loads the model, defines endpoints, handles predictions |
| `Config/config.py` | Loads `final_df.csv`, defines the expected feature order, builds options and catalog |
| `Schema/schemas.py` | Pydantic request schema and input validators |
| `Schema/prediction_response.py` | Prediction response schema |
| `best_random_forest.pkl` | Trained model artifact |
| `final_df.csv` | Vehicle data used to build the available options |

> The backend uses relative paths for the model and CSV, so the working directory must match the expected file locations.

---

## 🌐 Interactive Web Interface

I built a frontend with **HTML, CSS, and JavaScript** so predictions can be made without manually crafting API requests.

```
User input (Frontend)  →  FastAPI request validation  →  Fitted ML pipeline  →  Predicted price shown to user
```

- `GET /options` populates the available vehicle options.
- `POST /predict` submits the selected specifications and returns the estimated price.

<!-- 📸 Add application screenshots here -->
<!-- ![App Screenshot](screenshots/app.png) -->

---

## 🔍 Case Study: Debugging a Notebook-to-API Prediction Mismatch

> 💡 **Why this matters:** A model can score a high R² in a notebook and still return wrong answers in production if the inference path differs. A successful API response (HTTP 200) does not mean the prediction is correct, so I verified it instead of trusting it.

While testing through FastAPI's interactive docs, I noticed a prediction that looked **unusually high** for the vehicle I submitted. Instead of assuming the model was right, I investigated.

### What I Checked

| Step | Purpose |
|---|---|
| Compared notebook vs saved-model predictions | Check inference consistency |
| Verified the loaded model path | Make sure the correct artifact was used |
| Inspected matching **Maruti Ertiga** records in the dataset | Get a realistic price reference |
| Compared submitted specs with dataset records | Check input consistency |
| Tested **10 Ertiga configurations** | Avoid judging from a single example |
| Compared predictions before and after my changes | Confirm the fix |
| Used logging and direct prediction comparisons | Trace the behaviour |

### Outcome

After the debugging changes, notebook and API predictions **agreed on all inputs I tested**, and the checked examples were much closer to the corresponding dataset records.

> I did not conclusively identify the exact root cause of the original mismatch, so I have not attributed it to any specific cause.

### Key Distinction

- **Notebook ↔ API agreement** checks *inference consistency*.
- **Predictions vs actual prices** checks *prediction error*.
- Neither check alone proves performance across every vehicle category.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python |
| **Data Analysis** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn |
| **Model Serialization** | Joblib |
| **Dataset Acquisition** | KaggleHub |
| **Backend** | FastAPI |
| **Input Validation** | Pydantic |
| **Frontend** | HTML, CSS, JavaScript |
| **Development Environment** | Google Colab |

---

## 📁 Project Structure

```
CarDekho-Used-Car-Price-Prediction/
├── app.py                        # FastAPI application
├── Config/
│   └── config.py                 # Data loading, feature order, options
├── Schema/
│   ├── schemas.py                # Request schema + validators
│   └── prediction_response.py    # Response schema
├── best_random_forest.pkl        # Trained model
├── final_df.csv                  # Data for backend options
├── requirements.txt
├── .gitignore
├── README.md
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── notebooks/
    └── car_price_prediction.ipynb
```

---

## 🚀 Installation and Setup

### Prerequisites

- Python
- Git
- The dataset (`final_df.csv`) and trained model (`best_random_forest.pkl`)

### 1. Clone the repository

```bash
git clone https://github.com/Rishit-Vish/CarDekho-Used-Car-Price-Prediction.git
cd CarDekho-Used-Car-Price-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

```bash
uvicorn app:app --reload
```

Open the interactive docs at **http://127.0.0.1:8000/docs**

> Make sure `final_df.csv` and `best_random_forest.pkl` are available at the paths the application expects.

### 5. Start the frontend

Serve the `frontend/` files with a local development server (for example, VS Code Live Server or `python -m http.server`). Make sure the API URL in the frontend points to your running backend. If the frontend and backend run on different origins, CORS must allow the frontend's origin.

---

## ⚠️ Limitations

- Predictions depend on the **quality and coverage** of the training data.
- Real resale prices also depend on condition, location, service history, and market demand, which are not captured by the features.
- **Unusual vehicles or price ranges** may be less reliable where similar training examples are scarce.
- A high R² does not guarantee an accurate prediction for every individual car.
- Cross-validation and held-out test results come from different procedures and are reported separately.
- Baseline vs tuned metrics should be reproduced on a common test set before drawing firm conclusions about tuning.
- The root cause of the original notebook-to-API mismatch was not conclusively documented.

---

## 🔮 Future Improvements

- [ ] Reproduce baseline and tuned results on identical test data with consistent metric calculations
- [ ] Analyse prediction errors across price ranges and vehicle categories
- [ ] Add feature-importance analysis
- [ ] Add automated tests for API validation and model inference
- [ ] Improve deployment configuration and host a live demo
- [ ] Add application screenshots and reproducible evaluation results

---

## 💡 What I Learned

This project gave me hands-on experience with the full machine learning workflow: data exploration, preprocessing, model evaluation, serialization, and API integration.

It also taught me that **a good evaluation score is only one part of a useful ML application.** The inference pipeline must receive the expected features, the API must handle inputs correctly, and predictions must be checked against realistic examples.

Investigating the notebook-to-API mismatch was especially valuable. It pushed me to compare inference paths rather than assume that a successful API response meant a correct prediction.

---

## 👤 Author

**Rishit Vishwakarma**

🔗 GitHub: [Rishit-Vish](https://github.com/Rishit-Vish) · [CarDekho Used Car Price Prediction](https://github.com/Rishit-Vish/CarDekho-Used-Car-Price-Prediction)

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
