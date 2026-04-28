import pandas as pd
import yaml


COLUMN_NAMES = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'income'
]


def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def load_adult_data(file_path, is_test=False):
    if is_test:
        data = pd.read_csv(
            file_path,
            names=COLUMN_NAMES,
            skiprows=1,
            na_values=' ?',
            skipinitialspace=True
        )
    else:
        data = pd.read_csv(
            file_path,
            names=COLUMN_NAMES,
            na_values=' ?',
            skipinitialspace=True
        )

    data['income'] = data['income'].str.replace('.', '', regex=False)

    return data


def clean_data(data):
    cleaned_data = data.copy()

    cleaned_data = cleaned_data.dropna()

    cleaned_data['income'] = cleaned_data['income'].map({
        '<=50K': 0,
        '>50K': 1
    })

    return cleaned_data


def save_processed_data(train_data, test_data, config):
    train_path = config['data']['processed_train_path']
    test_path = config['data']['processed_test_path']

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)


def main():
    config = load_config()

    train_data = load_adult_data(config['data']['raw_train_path'])
    test_data = load_adult_data(config['data']['raw_test_path'], is_test=True)

    train_clean = clean_data(train_data)
    test_clean = clean_data(test_data)

    save_processed_data(train_clean, test_clean, config)

    print('Preprocessing complete.')
    print(f'Train shape: {train_clean.shape}')
    print(f'Test shape: {test_clean.shape}')


if __name__ == '__main__':
    main()