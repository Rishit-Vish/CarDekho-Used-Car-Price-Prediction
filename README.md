<div align="center">

# 🚗 CarDekho Used Car Price Prediction

**An end-to-end machine learning project: from raw data to a validated prediction API and web interface.**

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
- [Problem Statement and Business Context](#-problem-statement-and-business-context)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Model Comparison](#-model-comparison)
- [Hyperparameter Tuning](#-random-forest-baseline-and-hyperparameter-tuning)
- [Understanding the Two R² Scores](#-understanding-the-two-r-scores)
- [FastAPI Backend](#-fastapi-backend)
- [Web Interface](#-interactive-web-interface)
- [Case Study: Debugging a Notebook-to-API Prediction Mismatch](#-case-study-debugging-a-notebook-to-api-prediction-mismatch)
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
| **Tuning** | `RandomizedSearchCV` (5-fold CV, 10 iterations), with only a small gain |
| **Deployment** | FastAPI backend + HTML/CSS/JS frontend |
| **Validation** | Investigated a suspicious API prediction and verified it against the notebook and dataset records |

---

## 📖 Overview

Used car prices depend on vehicle age, kilometres driven, fuel type, transmission, engine capacity, and maximum power. This project learns those patterns from historical CarDekho listings and estimates the selling price of a vehicle.

I built the full workflow myself: EDA, preprocessing pipelines, model comparison, tuning, serialization, a validated REST API, a web interface, and an investigation into a suspicious API prediction.

### ✨ Project Highlights

- Performed **EDA** on distributions, relationships, missing values, skewness, and outliers.
- Built a leakage-aware workflow with `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, and `Pipeline`.
- Flagged outliers with the **1.5×IQR rule** and reviewed them by car name instead of deleting them blindly.
- Compared **5 regression algorithms** using 5-fold cross-validation (MAE, RMSE, R²).
- Selected **Random Forest**, tuned it with **`RandomizedSearchCV`**, and reported honestly that the gain was small.
- Built a **FastAPI** backend with **Pydantic** validation, CORS, and logging.
- Developed an interactive frontend with **HTML, CSS, and JavaScript**.
- Investigated a **notebook-vs-API prediction mismatch** instead of assuming the output was correct ([see case study](#-case-study-debugging-a-notebook-to-api-prediction-mismatch)).

---

## 🎯 Problem Statement and Business Context

Estimate a used car's selling price from its specifications and usage-related features. This is a **supervised regression problem**.

**Who could use this?** A used-car buyer or seller who wants a quick, data-driven price reference before negotiating, or a dealer who wants a first-pass valuation.

**How good is "good enough"?**

| Item | Value |
|---|---|
| Test MAE (tuned model) | ₹96,363 |
| Median selling price in the dataset | 👉 **FILL: ₹X** |
| MAE as % of median price | 👉 **FILL: ≈ Y%** |

> ⚠️ The model gives an *estimate* from patterns in historical listings. It is not a guaranteed market value.

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Name** | CarDekho Used Car Data |
| **Source** | [Kaggle: manishkr1754/cardekho-used-car-data](https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data) |
| **Download method** | KaggleHub |
| **Target variable** | `selling_price` (₹) |

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

> The original dataset had an `Unnamed: 0` index column, which I removed because it is not a predictive feature.

---

## 🔄 Project Workflow

```
Raw Data → EDA → Preprocessing → Model Comparison → Tuning → Save Model → FastAPI → Web UI → Validation
```

### 1. Exploratory Data Analysis (EDA)

- Dataset dimensions, data types, and statistical summaries
- Missing values across columns
- Numerical and categorical distributions
- Skewness in numerical features
- Correlations between numerical variables
- Relationship between **vehicle age and selling price**
- Selling-price distributions across **fuel types**
- Unnecessary columns and potential outliers

### 2. Data Preprocessing

I separated the features (`X`) from the target (`y`) and split the data **80:20** into training and test sets.

Numerical and categorical columns are handled by separate pipelines, combined with a `ColumnTransformer`:

| Feature group | Step 1 | Step 2 |
|---|---|---|
| **Numerical** | `SimpleImputer(strategy="median")` | `StandardScaler()` |
| **Categorical** | `SimpleImputer(strategy="most_frequent")` | `OneHotEncoder(handle_unknown="ignore")` |

```python
numeric_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])
```

**Design notes**

- Preprocessing lives **inside the pipeline**, so the same fitted transformations are used at training and inference time.
- During cross-validation, each pipeline is fitted only on its training fold, which helps prevent data leakage.
- `handle_unknown="ignore"` stops the pipeline from crashing if an unseen category appears at inference time.
- Scaling matters for Linear Regression but has little effect on tree-based models such as Random Forest. I kept one shared preprocessor so every model was compared under the same conditions.

### 3. Outlier Handling

I flagged outliers in `selling_price` using the **1.5×IQR rule**:

```python
Q1 = df["selling_price"].quantile(0.25)
Q3 = df["selling_price"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["selling_price"] < lower) | (df["selling_price"] > upper)]
print(outliers[["car_name", "selling_price"]])
print("Outlier count:", len(outliers))
```

I printed the flagged rows with their **car names** to check whether they were data errors or genuine premium vehicles. In used-car data, a very high price is often a *legitimate* expensive car, so I did not delete outliers automatically.

> Trade-off: keeping extreme prices preserves information about premium cars, but it can also increase error metrics such as RMSE, which penalises large errors heavily.

---

## 🏆 Model Comparison

I compared five regression algorithms from scikit-learn:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor

**Setup:** `KFold` with 5 splits, shuffling enabled, `random_state=42`. Each model sat inside the preprocessing pipeline and was evaluated with `cross_validate()`.

| Metric | Meaning | Better |
|---|---|---|
| **MAE** | Average absolute difference between actual and predicted prices | Lower |
| **RMSE** | Like MAE, but penalises large errors more heavily | Lower |
| **R²** | Performance relative to predicting the mean price | Higher |

> scikit-learn returns negative MAE and RMSE scores. I negated the mean scores to report positive errors.

### Cross-Validation Results (training data, mean of 5 folds)

| Rank | Model | Mean MAE | Mean RMSE | Mean R² |
|:---:|---|---:|---:|---:|
| 🥇 | **Random Forest** | **102,045.23** | **331,845.03** | **0.8644** |
| 🥈 | Extra Trees | 103,826.58 | 341,359.03 | 0.8546 |
| 🥉 | Gradient Boosting | 127,297.73 | 360,886.76 | 0.8418 |
| 4 | Decision Tree | 128,486.42 | 384,922.81 | 0.8228 |
| 5 | Linear Regression | 182,851.16 | 507,835.63 | 0.6926 |

**Takeaway:** Random Forest had the lowest MAE and RMSE and the highest R². Linear Regression was clearly the weakest, which suggests the price relationship is non-linear. Random Forest and Extra Trees are close, so the margin between the top two is small.

**Why Random Forest over Gradient Boosting?** It scored better in this comparison. I did not tune Gradient Boosting, so I do not claim Random Forest is the best possible model, only the best of the five as configured here.

---

## 🔧 Random Forest Baseline and Hyperparameter Tuning

### Baseline

```python
best_pipeline_wo_hpt = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])
```

> `n_estimators` was not set, so the baseline used scikit-learn's default of **100 trees**. The model comparison above used 200 trees.

### Tuning Method: `RandomizedSearchCV`

| Setting | Value |
|---|---|
| Search method | `RandomizedSearchCV` |
| Iterations (`n_iter`) | 10 random combinations |
| Cross-validation | 5-fold, on **training data only** |
| Scoring metric | Negative RMSE |
| Random state | 42 |
| Test set used for selection? | **No.** It was used only for the final evaluation. |

**Search space**

| Parameter | Values searched |
|---|---|
| `n_estimators` | 100, 200, 300, 500 |
| `max_depth` | None, 1, 20, 30 |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |

The full grid has 144 combinations, so 10 iterations cover only a small part of it.

```python
best_pipe_rfr = RandomizedSearchCV(
    estimator=best_pipeline_w_hpt,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1
)
best_pipe_rfr.fit(X_train, y_train)
```

### Selected Hyperparameters

| Parameter | Selected | Default |
|---|---:|---:|
| `n_estimators` | 200 | 100 |
| `max_depth` | 30 | None (unlimited) |
| `min_samples_split` | 10 | 2 |
| `min_samples_leaf` | 1 | 1 |

### Baseline vs Tuned (held-out test set)

| Metric | Baseline | After Tuning | Change |
|---|---:|---:|---:|
| **MAE** | 98,486.40 | 96,362.88 | ↓ 2,123.52 |
| **RMSE** | 209,336.60 | 209,230.96 | ↓ 105.64 |
| **R²** | 0.927927 | 0.928000 | ↑ 0.000073 |

**Takeaway:** The gain is very small. An R² change of 0.000073 is within noise, so **I treat tuning as having no meaningful effect on accuracy**. The baseline Random Forest was already strong, so most of the performance comes from the model choice and preprocessing.

> 📝 The baseline MAE was originally printed with an "MSE" label in my notebook because the code used `mean_absolute_error()`. The table above is correctly labelled.

---

## ❓ Understanding the Two R² Scores

| | Model Comparison | Baseline Evaluation |
|---|---|---|
| **R² reported** | **0.8644** | **0.9279** |
| **Method** | 5-fold cross-validation | Single train/test split |
| **Evaluated on** | Validation folds inside `X_train` | Held-out `X_test` |
| **Trees** | 200 | 100 |
| **Purpose** | Compare 5 models fairly | Check the chosen model on unseen data |

Use the **cross-validation scores** to compare models with each other, and the **test scores** to compare baseline vs tuned.

### The gap is real, and I have not fully explained it

Test RMSE (≈ 209k) is noticeably lower than CV RMSE (≈ 332k). Differences in procedure alone do not explain this. Possible reasons I have **not yet verified**:

- The test split may contain fewer extreme-priced cars than the training folds.
- The single split may simply be a favourable one.

**Planned check:** repeat the evaluation across several random splits and report the mean and spread, so the result does not rest on one split.

---

## ⚙️ FastAPI Backend

The backend loads the saved model, converts incoming specifications into a Pandas DataFrame, and returns a prediction.

| Method | Endpoint | Purpose |
|:---:|---|---|
| `GET` | `/options` | Returns supported car names, brands, models, seller types, fuel types, transmission types, and the car catalog |
| `POST` | `/predict` | Accepts vehicle specifications and returns the predicted selling price |
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |

### Input Validation (Pydantic)

- **Required fields and data types** are enforced.
- `vehicle_age` and `km_driven` must be **non-negative integers**.
- `mileage` and `max_power` must be **non-negative numbers**.
- **Categorical fields** must belong to the options loaded by the backend.

### Example Response

```json
{
  "predicted_selling_price": 720840.96
}
```

The price is in **Indian rupees (₹)**. Unexpected failures are logged with Python's `logging` module and the API returns a generic HTTP 500 message.

### Backend Modules

| File | Responsibility |
|---|---|
| `app.py` | Initialises FastAPI, configures CORS, loads the model, defines endpoints |
| `Config/config.py` | Loads `final_df.csv`, defines feature order, builds options and catalog |
| `Schema/schemas.py` | Pydantic request schema and validators |
| `Schema/prediction_response.py` | Prediction response schema |
| `best_random_forest.pkl` | Trained model artifact |
| `final_df.csv` | Data used to build the available options |

> The backend uses relative paths for the model and CSV, so it must be run from the project root.

---

## 🌐 Interactive Web Interface

```
User input (Frontend)  →  FastAPI validation  →  Fitted ML pipeline  →  Predicted price shown to user
```

- `GET /options` populates the available vehicle options.
- `POST /predict` submits the selected specifications and returns the estimated price.

<!-- 📸 Add application screenshots here -->
<!-- ![App Screenshot](screenshots/app.png) -->

---

## 🔍 Case Study: Debugging a Notebook-to-API Prediction Mismatch

**Problem:** While testing through FastAPI's interactive docs, I noticed a prediction that looked **unusually high** for the vehicle I submitted. The API returned HTTP 200, so nothing looked broken, but the number did not look right.

> 💡 **Why this matters:** A model can score a high R² in a notebook and still return wrong answers in production if the inference path differs. A successful API response does not mean the prediction is correct, so I verified it instead of trusting it.

### What I Checked

| Step | Purpose |
|---|---|
| Compared notebook vs saved-model predictions | Check inference consistency |
| Verified the loaded model path | Make sure the correct artifact was used |
| Inspected matching **Maruti Ertiga** records in the dataset | Get a realistic price reference |
| Compared submitted specs with dataset records | Check input consistency |
| Tested **10 Ertiga configurations** | Avoid judging from a single example |
| Compared predictions before and after my changes | Confirm the change helped |
| Used logging and direct prediction comparisons | Trace the behaviour |

### What I Changed

During this investigation I re-ran **hyperparameter tuning** (`RandomizedSearchCV`) and updated the saved model. After that, notebook and API predictions agreed on every input I tested, and the Ertiga examples were much closer to the corresponding dataset records.

> I report this as the sequence of events, not as a proven cause. Tuning changed test R² by only 0.000073, which is too small to explain a large prediction error by itself. A notebook-vs-API disagreement also points to the inference path rather than model accuracy, so the true cause may be something else that changed at the same time, such as which model file was loaded.

### What I Could Not Conclude

I did not conclusively establish the root cause of the original mismatch, so I do not claim one. I treat this as a **mitigated issue, not a fully closed one**. The next step is to add automated tests that compare API output against direct model output for a fixed set of inputs, so any regression is caught immediately.

### Key Distinction

- **Notebook ↔ API agreement** checks *inference consistency*.
- **Predictions vs actual prices** checks *prediction error*.
- Neither alone proves performance across every vehicle category.

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
- `final_df.csv` and `best_random_forest.pkl` in the project root

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

### 5. Start the frontend

Serve the `frontend/` files with a local server (for example VS Code Live Server or `python -m http.server`). Make sure the API URL in the frontend points to your running backend. If they run on different origins, CORS must allow the frontend's origin.

---

## ⚠️ Limitations

- Predictions depend on the **quality and coverage** of the training data.
- Real resale prices also depend on condition, location, service history, and market demand, which the features do not capture.
- **Unusual vehicles or price ranges** may be less reliable where similar training examples are scarce.
- A high R² does not guarantee an accurate prediction for every individual car.
- The gap between CV and test scores is not yet fully explained.
- The tuning search covered only 10 of 144 possible combinations and produced no meaningful improvement. Gradient Boosting was not tuned.
- The root cause of the notebook-to-API mismatch was not established.
- There are no automated tests and no hosted demo yet.

---

## 🔮 Future Improvements

- [ ] Repeat evaluation across multiple splits to explain the CV-vs-test gap
- [ ] Widen the tuning search (more iterations and a sensible `max_depth` range) and tune Extra Trees and Gradient Boosting for a fair comparison
- [ ] Test a log-transformed target for the skewed price distribution
- [ ] Analyse errors by price range and vehicle category
- [ ] Add feature-importance analysis
- [ ] Add automated tests comparing API output with direct model output
- [ ] Containerise with Docker and host a live demo
- [ ] Add application screenshots

---

## 💡 What I Learned

This project gave me hands-on experience with the full machine learning workflow: exploration, preprocessing, evaluation, serialization, and API integration.

It taught me that **a good evaluation score is only one part of a useful ML application.** The inference pipeline must receive the expected features, the API must handle inputs correctly, and predictions must be checked against realistic examples. It also taught me to report results honestly: when tuning barely helped, and when I could not explain a gap or a bug's root cause, I wrote that down instead of hiding it.

---

## 👤 Author

**Rishit Vishwakarma**

🔗 GitHub: [Rishit-Vish](https://github.com/Rishit-Vish) · [CarDekho Used Car Price Prediction](https://github.com/Rishit-Vish/CarDekho-Used-Car-Price-Prediction)

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
