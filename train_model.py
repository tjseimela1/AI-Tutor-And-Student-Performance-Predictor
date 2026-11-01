"""Train 3-class performance models on the real xAPI-Edu-Data benchmark.

Features selected for actionability; excludes gender, nationality, birthplace,
parent variables and the target Class. Preprocessing is trained within folds
(to prevent one-hot/scaler leakage). Stratified 60/20/20 train/val/test.
"""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data_preprocessing import (CATEGORICAL_FEATURES, CLASSES, DATASET,
                                FEATURES, NUMERIC_FEATURES, load_dataset)


def build_pipeline(estimator):
    prep = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), NUMERIC_FEATURES),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([('preprocess', prep), ('model', estimator)])


def calculate_metrics(model, x, y):
    predicted = model.predict(x)
    return {
        'accuracy': round(float(accuracy_score(y, predicted)), 4),
        'balanced_accuracy': round(float(balanced_accuracy_score(y, predicted)), 4),
        'precision_macro': round(float(precision_score(y, predicted, labels=CLASSES, average='macro', zero_division=0)), 4),
        'recall_macro': round(float(recall_score(y, predicted, labels=CLASSES, average='macro', zero_division=0)), 4),
        'f1_macro': round(float(f1_score(y, predicted, labels=CLASSES, average='macro', zero_division=0)), 4),
        'confusion_matrix': confusion_matrix(y, predicted, labels=CLASSES).tolist(),
        'per_class': classification_report(y, predicted, labels=CLASSES, output_dict=True, zero_division=0),
    }


def train(data_path=DATASET, model_dir='models'):
    df = load_dataset(data_path)
    x, y = df[FEATURES], df['Class']
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    algorithms = {
        'Logistic regression': LogisticRegression(max_iter=700, class_weight='balanced', random_state=42),
        'Random forest': RandomForestClassifier(n_estimators=180, min_samples_leaf=2, class_weight='balanced', random_state=42),
        'Deep neural network (MLP)': MLPClassifier(hidden_layer_sizes=(48, 24, 12),
            alpha=0.1, tol=0.002, early_stopping=False, n_iter_no_change=28,
            max_iter=650, random_state=42),
    }
    fits, validation = {}, {}
    for name, estimator in algorithms.items():
        model = build_pipeline(estimator)
        model.fit(x_train, y_train)
        fits[name] = model
        validation[name] = calculate_metrics(model, x_val, y_val)
    chosen = max(fits, key=lambda k: (validation[k]['f1_macro'], validation[k]['recall_macro']))
    test = calculate_metrics(fits[chosen], x_test, y_test)
    folder = Path(model_dir)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': fits[chosen], 'features': FEATURES,
                 'model_name': chosen, 'classes': list(fits[chosen].classes_)}, folder / 'risk_model.joblib')
    joblib.dump(fits['Deep neural network (MLP)'], folder / 'deep_model.joblib')
    output = {
        'dataset': 'xAPI-Edu-Data (Kalboard 360 LMS; Kaggle, aljarah/xAPI-Edu-Data)',
        'dataset_url': 'https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data',
        'records': len(df), 'train_records': len(x_train), 'validation_records': len(x_val),
        'test_records': len(x_test), 'target': 'Class: L=Low, M=Middle, H=High PERFORMANCE',
        'features': FEATURES,
        'excluded_features': ['gender', 'NationalITy', 'PlaceofBirth', 'SectionID', 'Relation',
                              'ParentAnsweringSurvey', 'ParentschoolSatisfaction'],
        'selected_model': chosen, 'selection_metric': 'validation macro F1',
        'validation_results': validation, 'held_out_test': test,
        'class_labels': CLASSES,
        'risk_thresholds': {'low': '< 30% P(Class=L)', 'moderate': '30–<60%', 'high': '>=60%'},
        'limitations': 'Performance categories, not pass/fail or dropout outcomes. No marks, precise attendance, student IDs, or dated history. In-dataset display is illustrative; test metrics use held-out rows.',
    }
    (folder / 'metrics.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    return output


if __name__ == '__main__':
    print(json.dumps(train(), indent=2))