import io
import pandas as pd
import pytest

from chatbot import answer
from data_preprocessing import FEATURES, DATASET, load_dataset, validate_student_csv, CLASSES
from predict import risk_label, score_students
from reports import dashboard_summary, stage_overview, grade_overview, school_context
from time_series import weekly_trend, semester_comparison
from train_models import train


def test_actual_xapi_dataset():
    rows = load_dataset()
    assert len(rows) == 480
    assert rows['Class'].value_counts().to_dict() == {'M': 211, 'H': 142, 'L': 127}
    assert rows.student_id.is_unique
    assert rows.student_id.iloc[0] == 'XAPI-0001'
    assert set(rows.Class) == set(CLASSES)


def test_features_exclude_leakage_and_demographics():
    assert 'Class' not in FEATURES
    assert 'gender' not in FEATURES
    assert 'NationalITy' not in FEATURES
    assert 'PlaceofBirth' not in FEATURES
    assert 'ParentAnsweringSurvey' not in FEATURES
    assert {'VisITedResources', 'StudentAbsenceDays'} <= set(FEATURES)


def test_train_and_score_real_xapi(tmp_path):
    result = train(DATASET, tmp_path / 'models')
    assert result['records'] == 480
    assert result['test_records'] == 96
    assert result['selected_model'] in result['validation_results']
    assert 0 <= result['held_out_test']['f1_macro'] <= 1
    assert len(result['held_out_test']['confusion_matrix']) == 3
    scored = score_students(load_dataset().head(12), tmp_path / 'models' / 'risk_model.joblib')
    assert scored.predicted_class.isin(CLASSES).all()
    assert scored.probability_low.between(0, 100).all()
    assert (scored[['probability_low', 'probability_middle', 'probability_high']].sum(axis=1) - 100).abs().max() < 0.2
    assert dashboard_summary(scored)['students'] == 12


def test_semesters_are_cohort_comparisons_not_history():
    groups = semester_comparison(load_dataset())
    assert len(groups) == 2
    assert sum(x['records'] for x in groups) == 480
    assert not weekly_trend()['forecast']
    assert not weekly_trend()['actual']


def test_real_dated_history_supports_forecast():
    frame = pd.DataFrame({'week_start': pd.date_range('2026-07-01', periods=5, freq='7D'),
                          'VisITedResources': [20, 35, 40, 36, 55]})
    result = weekly_trend(frame)
    assert len(result['actual']) == 5
    assert len(result['forecast']) == 3
    assert result['forecast'][0]['date'] == '2026-08-05'
    assert all(0 <= row['visits'] <= 100 for row in result['forecast'])


def test_chatbot_nlp():
    assert answer('What is a Python function?')['matched']
    assert not answer('zxqvwertyuio nonsense')['matched']


def test_csv_requires_xapi_fields():
    real = pd.read_csv(DATASET).head(3)
    imported = validate_student_csv(io.BytesIO(real.to_csv(index=False).encode()))
    assert len(imported) == 3
    assert imported.student_id.iloc[0] == 'XAPI-0001'
    with pytest.raises(ValueError):
        validate_student_csv(io.BytesIO(b'marks,attendance,lms_activity\n70,80,20\n'))
    real.loc[0, 'raisedhands'] = 150
    with pytest.raises(ValueError):
        validate_student_csv(io.BytesIO(real.to_csv(index=False).encode()))


def test_risk_bands():
    assert [risk_label(p) for p in [.29, .3, .59, .6]] == ['Low', 'Moderate', 'Moderate', 'High']


def test_stage_grade_and_parent_context():
    data = load_dataset()
    groups = stage_overview(data)
    assert [(g['code'], g['records']) for g in groups] == [
        ('lowerlevel', 199), ('MiddleSchool', 248), ('HighSchool', 33)]
    assert len(grade_overview(data)) == 10
    parent = school_context(data)
    assert parent['survey_total'] == 480
    assert parent['satisfaction_total'] == 480
    assert parent['satisfaction_good'] == 292