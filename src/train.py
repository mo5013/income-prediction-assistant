import os
import joblib
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def load_processed_data(config):
    train_path = config['data']['processed_train_path']
    test_path = config['data']['processed_test_path']
    target_column = config['data']['target_column']

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    features_train = train_data.drop(target_column, axis=1)
    target_train = train_data[target_column]

    features_test = test_data.drop(target_column, axis=1)
    target_test = test_data[target_column]

    return features_train, features_test, target_train, target_test


def build_preprocessor(config):
    numeric_features = config['features']['numeric']
    categorical_features = config['features']['categorical']

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numeric_features),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    return preprocessor


def get_model_configs(random_state):
    model_configs = [
        {
            'run_name': 'logistic_regression',
            'model': LogisticRegression(max_iter=1000, random_state=random_state),
            'params': {
                'model_type': 'LogisticRegression',
                'max_iter': 1000
            }
        },
        {
            'run_name': 'random_forest_100',
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state
            ),
            'params': {
                'model_type': 'RandomForestClassifier',
                'n_estimators': 100,
                'max_depth': 10
            }
        },
        {
            'run_name': 'random_forest_200',
            'model': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=random_state
            ),
            'params': {
                'model_type': 'RandomForestClassifier',
                'n_estimators': 200,
                'max_depth': 15
            }
        },
        {
            'run_name': 'gradient_boosting_100',
            'model': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=random_state
            ),
            'params': {
                'model_type': 'GradientBoostingClassifier',
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3
            }
        },
        {
            'run_name': 'gradient_boosting_150',
            'model': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state
            ),
            'params': {
                'model_type': 'GradientBoostingClassifier',
                'n_estimators': 150,
                'learning_rate': 0.05,
                'max_depth': 3
            }
        }
    ]

    return model_configs


def evaluate_model(model, features_test, target_test):
    predictions = model.predict(features_test)
    probabilities = model.predict_proba(features_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(target_test, predictions),
        'precision': precision_score(target_test, predictions),
        'recall': recall_score(target_test, predictions),
        'f1': f1_score(target_test, predictions),
        'auc': roc_auc_score(target_test, probabilities)
    }

    return metrics


def train_and_track_models(config):
    random_state = config['project']['random_state']
    experiment_name = config['mlflow']['experiment_name']
    metric_to_optimize = config['model']['metric_to_optimize']
    model_output_path = config['model']['output_path']

    features_train, features_test, target_train, target_test = load_processed_data(config)

    mlflow.set_experiment(experiment_name)

    best_score = -1
    best_model = None
    best_run_name = None
    results = []

    model_configs = get_model_configs(random_state)

    for model_config in model_configs:
        preprocessor = build_preprocessor(config)

        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', model_config['model'])
            ]
        )

        with mlflow.start_run(run_name=model_config['run_name']):
            pipeline.fit(features_train, target_train)

            metrics = evaluate_model(pipeline, features_test, target_test)

            mlflow.log_params(model_config['params'])
            mlflow.log_param('data_version', 'adult_income_processed_v1')
            mlflow.log_param('train_rows', features_train.shape[0])
            mlflow.log_param('test_rows', features_test.shape[0])

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, 'model')

            result = {
                'run_name': model_config['run_name'],
                **metrics
            }
            results.append(result)

            print(f"\nRun: {model_config['run_name']}")
            print(metrics)

            if metrics[metric_to_optimize] > best_score:
                best_score = metrics[metric_to_optimize]
                best_model = pipeline
                best_run_name = model_config['run_name']

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(best_model, model_output_path)

    results_data = pd.DataFrame(results)
    os.makedirs('reports', exist_ok=True)
    results_data.to_csv('reports/model_results.csv', index=False)

    print('\nTraining complete.')
    print(f'Best run: {best_run_name}')
    print(f'Best {metric_to_optimize}: {best_score:.4f}')
    print(f'Best model saved to: {model_output_path}')


def main():
    config = load_config()
    train_and_track_models(config)


if __name__ == '__main__':
    main()