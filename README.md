# Income Prediction Assistant (Capstone Project)

## Project Overview

This project is an end-to-end machine learning application that predicts whether an individual earns more than $50K per year based on demographic and employment-related attributes.

The system combines:

* A trained machine learning model (classification)
* An LLM-style natural language interface for user interaction

Users can input information in plain English, and the system:

1. Extracts structured features from the text
2. Runs the trained model
3. Returns a prediction with explanation

---

## Problem Statement

Income classification is a common real-world problem that can support economic analysis and workforce insights. This project demonstrates how machine learning can be combined with natural language interfaces to make predictive systems more accessible to non-technical users.

---

## Dataset

* **Dataset:** Adult Income Dataset (UCI Repository)
* **Rows:** ~48,000
* **Target Variable:** `income` (<=50K or >50K)

### Features Used

* Age
* Workclass
* Education
* Marital Status
* Occupation
* Relationship
* Race
* Sex
* Hours per Week
* Native Country
* Additional numeric features (capital gain/loss, etc.)

---

## Data Preprocessing

* Missing values (`?`) were converted to NaN and removed
* Categorical variables were encoded using OneHotEncoder
* Numerical features were scaled using StandardScaler
* Target variable (`income`) was converted to binary (0 / 1)

### Design Decision

Rows with missing values were removed instead of imputed because:

* The dataset is large enough to retain sufficient data
* This avoids introducing bias from incorrect imputations

---

## Model Training

Multiple models were trained and compared using MLflow:

* Logistic Regression
* Random Forest (2 configurations)
* Gradient Boosting (2 configurations)

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* AUC

### Best Model

* **Model:** Gradient Boosting Classifier
* **Best F1 Score:** ~0.69

### Model Comparison

| Model               | F1 Score | AUC   |
| ------------------- | -------- | ----- |
| Logistic Regression | ~0.65    | ~0.90 |
| Random Forest       | ~0.66    | ~0.91 |
| Gradient Boosting   | ~0.69    | ~0.92 |

### Justification

Gradient Boosting achieved the best balance between precision and recall, making it more suitable for classification where both false positives and false negatives matter.

---

## Experiment Tracking (MLflow)

MLflow was used to:

* Log parameters and hyperparameters
* Track evaluation metrics
* Store trained model artifacts
* Compare multiple runs

The best model was selected programmatically based on F1 score.

---

## Architecture Overview

The system consists of two main components:

### 1. Machine Learning Pipeline

* Data preprocessing (cleaning, encoding, scaling)
* Model training with multiple configurations
* Experiment tracking using MLflow
* Best model selection based on F1 score

### 2. LLM-Style Interface (Streamlit)

* Parses natural language input into structured features
* Loads the trained model (`best_model.joblib`)
* Generates predictions and explanations
* Handles missing or incomplete inputs gracefully

---

## LLM-Powered Interface

### Workflow

1. User enters a natural language description
2. The system parses key features
3. The trained model is invoked
4. The system returns a prediction with explanation

### Example Input

"I am a 42-year-old male with a Masters degree working in management for 50 hours per week."

### Example Output

* Prediction: >50K
* Confidence score
* Explanation of key factors

---

## Edge Case Handling

Example:

```
Can you predict my income?
```

Output:

* Requests additional information
* Lists missing required fields

---

## Testing

A full test suite was implemented using pytest:

* Preprocessing tests (4)
* Model tests (2)
* Interface tests (2)

Run tests:

```
python -m pytest tests/ -v
```

---

## Project Structure

```
capstone-income-assistant/
├── configs/
├── data/
├── models/
├── reports/
├── src/
├── tests/
├── notebooks/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
└── Dockerfile
```

---

## Setup Instructions

### Install dependencies

```
python -m pip install -r requirements.txt
```

### Run preprocessing

```
python src/preprocess.py
```

### Train models

```
python src/train.py
```

### Launch app

```
python -m streamlit run src/app.py
```

> Note: The trained model is generated during training and is not included in the repository.

---

## Docker

Build and run:

```
docker build -t income-prediction-assistant .
docker run -p 8501:8501 income-prediction-assistant
```

Open:
http://localhost:8501

---

## Demo

### Example Input
"I am a 42-year-old male with a Masters degree working in management for 50 hours per week in the United States. I work in the private sector and I am married."

### Example Output
- Prediction: >50K  
- Confidence: ~78%  
- Explanation: Based on factors such as education level, hours worked, and occupation.

### Edge Case Example
Input:
"Can you predict my income?"

Output:
- The system requests additional information  
- Lists missing required fields instead of making an incorrect prediction

---

## Reflection

### Key Learnings

* Importance of preprocessing pipelines
* Value of experiment tracking
* Challenges of NLP feature extraction

### Challenges

* Handling inconsistent natural language
* Aligning parsed features with model

### Future Improvements

* Integrate real LLM API
* Improve parsing robustness
* Add fairness analysis

---

## Conclusion

This project demonstrates a full ML lifecycle:

* Data → Model → Tracking → Interface → Testing → Deployment
