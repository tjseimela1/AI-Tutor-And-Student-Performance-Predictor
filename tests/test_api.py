import os
os.environ['DEMO_MODE'] = 'true'
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)
EXAMPLE = {
    'raisedhands': 40, 'VisITedResources': 35,
    'AnnouncementsView': 30, 'Discussion': 30,
    'StageID': 'MiddleSchool', 'GradeID': 'G-07',
    'Topic': 'IT', 'Semester': 'F', 'StudentAbsenceDays': 'Under-7',
}


def test_health_and_pages():
    assert client.get('/health').json()['dataset'] == 'xAPI-Edu-Data'
    for page in ['/', '/dashboard', '/students', '/forecast', '/reports', '/tutor', '/try-prediction']:
        response = client.get(page)
        assert response.status_code == 200, (page, response.text[:400])
        assert 'EduPulse' in response.text
    assert '480' in client.get('/dashboard').text
    assert 'School Stages in the xAPI Dataset' in client.get('/dashboard').text
    assert 'Middle school' in client.get('/dashboard').text
    assert 'Grade-Level Breakdown' in client.get('/reports').text
    assert 'school-education' == client.get('/health').json()['audience']
    assert 'no timestamps' in client.get('/forecast').text


def test_chat_and_multiclass_api():
    answer = client.post('/api/chat', json={'question': 'How can I prepare for exams?'})
    assert answer.status_code == 200 and answer.json()['matched']
    scored = client.post('/api/predict', json=EXAMPLE)
    assert scored.status_code == 200, scored.text
    assert scored.json()['predicted_class'] in ('L', 'M', 'H')
    assert 0 <= scored.json()['probability_low'] <= 100
    bad = dict(EXAMPLE, raisedhands=121)
    assert client.post('/api/predict', json=bad).status_code == 422
    assert client.post('/api/predict', json={'marks': 60, 'attendance': 80, 'lms_activity': 12}).status_code == 422


def test_student_benchmark_and_export():
    assert client.get('/students/XAPI-0001').status_code == 200
    assert client.get('/students/DEMO-0001').status_code == 404
    assert client.get('/import').status_code == 403
    exported = client.get('/reports/download')
    assert exported.status_code == 200
    assert 'probability_low' in exported.text
    assert 'marks' not in exported.text.splitlines()[0]


def test_manual_prediction_form():
    with TestClient(app) as session:
        page = session.get('/try-prediction')
        # regex keeps tests dependency-free; bs4 is not needed at runtime.
        import re
        token = re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)
        outcome = session.post('/try-prediction', data={**EXAMPLE, 'csrf': token})
        assert outcome.status_code == 200
        assert 'estimated performance group' in outcome.text
        assert 'Probability of Low (L)' in outcome.text


def test_school_stage_grade_and_filters():
    dashboard = client.get('/dashboard').text
    for stage in ['Lower level', 'Middle school', 'High school']:
        assert stage in dashboard
    response = client.get('/learners?stage=HighSchool&grade=G-12')
    assert response.status_code == 200
    assert '11 results' in response.text
    assert client.get('/learners/XAPI-0001').status_code == 200
    # School stage and grade pairing is checked against actual source combinations.
    wrong = {**EXAMPLE, 'StageID': 'HighSchool', 'GradeID': 'G-02'}
    assert client.post('/api/predict', json=wrong).status_code == 422
    invalid_subject = {**EXAMPLE, 'Topic': 'University Accounting'}
    assert client.post('/api/predict', json=invalid_subject).status_code == 422


def test_school_tutoring():
    response = client.post('/api/chat', json={'question': 'What is photosynthesis?'})
    assert response.status_code == 200
    assert response.json()['matched']
    assert 'plants' in response.json()['answer'].lower()
    page = client.get('/tutor').text
    assert 'School Study Tutor' in page
    assert 'photosynthesis' in page
