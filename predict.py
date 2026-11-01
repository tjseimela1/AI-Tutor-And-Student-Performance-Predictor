"""Multiclass xAPI performance prediction, with separate low-performance probability."""
from pathlib import Path
import joblib
import numpy as np
from data_preprocessing import DATASET, FEATURES
from train_models import train


def load_model(path='models/risk_model.joblib'):
    path = Path(path)
    if not path.exists():
        train(DATASET, path.parent)
    package = joblib.load(path)
    if package.get('features') != FEATURES:
        raise ValueError('Incompatible old model; delete models/risk_model.joblib and retrain.')
    return package


def risk_label(probability):
    # These are demonstration-only prioritisation bands, NOT calibrated probabilities.
    if probability >= 0.60:
        return 'High'
    if probability >= 0.30:
        return 'Moderate'
    return 'Low'


def study_recommendation(row):
    suggestions = []
    if row['StudentAbsenceDays'] == 'Above-7':
        suggestions.append('Absence recorded as more than seven days. Ask a teacher and parent or guardian about any barriers to attending school.')
    if row['VisITedResources'] < 30:
        suggestions.append('Learning-resource visits are below 30; plan short, regular practice with a teacher or parent/guardian.')
    if row['raisedhands'] < 25:
        suggestions.append('Raised-hand participation is below 25; encourage the learner to ask questions during lessons.')
    if row['Discussion'] < 25:
        suggestions.append('Discussion participation is below 25; encourage supported classroom group work.')
    if row['AnnouncementsView'] < 25:
        suggestions.append('Announcements viewed are below 25; help the learner check school or class updates.')
    if not suggestions:
        suggestions.append('Continue learning habits, give positive feedback, and review progress with the class teacher.')
    return suggestions


def score_students(frame, model_path='models/risk_model.joblib'):
    package = load_model(model_path)
    result = frame.copy()
    probabilities = package['pipeline'].predict_proba(result[FEATURES])
    labels = package['pipeline'].classes_
    p = {label: probabilities[:, list(labels).index(label)] for label in ('L', 'M', 'H')}
    result['predicted_class'] = package['pipeline'].predict(result[FEATURES])
    result['probability_low'] = np.round(p['L'] * 100, 1)
    result['probability_middle'] = np.round(p['M'] * 100, 1)
    result['probability_high'] = np.round(p['H'] * 100, 1)
    result['risk_probability'] = result['probability_low']  # compatibility: probability of low performance
    result['risk'] = [risk_label(x) for x in p['L']]
    result['recommendations'] = [study_recommendation(row) for _, row in result.iterrows()]
    return result
