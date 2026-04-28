import joblib
import pandas as pd


def test_model_loads_successfully():
    model = joblib.load('models/best_model.joblib')

    assert model is not None


def test_model_prediction_shape_and_type():
    model = joblib.load('models/best_model.joblib')

    sample = pd.DataFrame([{
        'age': 42,
        'workclass': 'Private',
        'fnlwgt': 200000,
        'education': 'Masters',
        'education_num': 14,
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 50,
        'native_country': 'United-States'
    }])

    prediction = model.predict(sample)

    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]