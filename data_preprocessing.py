"""Load and validate the genuine xAPI-Edu-Data benchmark CSV (480 records).

Important: these anonymous records have no learner identity, timestamp, precise
mark, or attendance percentage. The row IDs below are display-only identifiers.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
DATASET = BASE / 'data' / 'xAPI-Edu-Data.csv'
NUMERIC_FEATURES = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
CATEGORICAL_FEATURES = ['StageID', 'GradeID', 'Topic', 'Semester', 'StudentAbsenceDays']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = 'Class'
CLASSES = ['L', 'M', 'H']  # Low/Middle/High academic PERFORMANCE, not pass/fail
RAW_COLUMNS = [
    'gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
    'SectionID', 'Topic', 'Semester', 'Relation', 'raisedhands',
    'VisITedResources', 'AnnouncementsView', 'Discussion',
    'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
    'StudentAbsenceDays', 'Class',
]


def prepare_records(frame, *, require_target=True):
    """Validate real xAPI schema. Retain target for auditing, never as an input."""
    missing = [c for c in FEATURES if c not in frame.columns]
    if require_target and TARGET not in frame:
        missing.append(TARGET)
    if missing:
        raise ValueError('Missing xAPI columns: ' + ', '.join(missing))
    if not 1 <= len(frame) <= 5000:
        raise ValueError('Dataset must contain between 1 and 5,000 records.')
    out = frame.copy().reset_index(drop=True)
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors='coerce')
        if out[col].isna().any() or (~out[col].between(0, 100)).any():
            raise ValueError(f'{col} must be numeric and within 0–100.')
    for col in CATEGORICAL_FEATURES:
        if out[col].isna().any() or (out[col].astype(str).str.strip() == '').any():
            raise ValueError(f'{col} must not be blank.')
        out[col] = out[col].astype(str).str.strip().str.slice(0, 80)
    if 'StudentAbsenceDays' in out and not out['StudentAbsenceDays'].isin(['Under-7', 'Above-7']).all():
        raise ValueError('StudentAbsenceDays must be Under-7 or Above-7.')
    if TARGET in out:
        if not out[TARGET].isin(CLASSES).all():
            raise ValueError('Class must contain L, M, or H only.')
    # Data has no identifiers. Never invent people: these are dataset row numbers.
    out['student_id'] = [f'XAPI-{i:04d}' for i in range(1, len(out) + 1)]
    out['student_name'] = [f'Dataset record {i:03d}' for i in range(1, len(out) + 1)]
    out['learner_id'] = out['student_id']
    out['learner_label'] = out['student_name']
    out['course'] = out['Topic']
    return out


def load_dataset(path=DATASET):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Missing xAPI dataset: {path}. Place the supplied CSV at data/xAPI-Edu-Data.csv.')
    return prepare_records(pd.read_csv(path), require_target=True)


def validate_student_csv(file_like):
    """Restricted-mode optional CSV must use xAPI field names, not invented marks."""
    return prepare_records(pd.read_csv(file_like, nrows=5001), require_target=False)


if __name__ == '__main__':
    df = load_dataset()
    print('xAPI-Edu-Data loaded:', len(df), 'records,', len(FEATURES), 'chosen model inputs')
    print('Performance classes:', df.Class.value_counts().to_dict())