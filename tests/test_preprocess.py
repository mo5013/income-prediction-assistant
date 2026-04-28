import pandas as pd

from src.preprocess import clean_data


def test_clean_data_removes_missing_values():
    data = pd.DataFrame({
        'age': [25, None],
        'workclass': ['Private', 'Private'],
        'income': ['<=50K', '>50K']
    })

    cleaned_data = clean_data(data)

    assert cleaned_data.isna().sum().sum() == 0
    assert cleaned_data.shape[0] == 1


def test_clean_data_encodes_income_column():
    data = pd.DataFrame({
        'age': [25, 40],
        'workclass': ['Private', 'Private'],
        'income': ['<=50K', '>50K']
    })

    cleaned_data = clean_data(data)

    assert set(cleaned_data['income'].unique()) == {0, 1}


def test_clean_data_does_not_modify_original_dataframe():
    data = pd.DataFrame({
        'age': [25, None],
        'workclass': ['Private', 'Private'],
        'income': ['<=50K', '>50K']
    })

    original_data = data.copy()
    clean_data(data)

    pd.testing.assert_frame_equal(data, original_data)


def test_clean_data_keeps_expected_columns():
    data = pd.DataFrame({
        'age': [25, 40],
        'workclass': ['Private', 'Private'],
        'income': ['<=50K', '>50K']
    })

    cleaned_data = clean_data(data)

    assert list(cleaned_data.columns) == ['age', 'workclass', 'income']