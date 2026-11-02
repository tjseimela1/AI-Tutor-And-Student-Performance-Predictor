"""School-level xAPI learner performance classification (benchmark demo by default).

No real school learner records are connected. Teacher access must be independently
authorised before processing actual records. Low class does not prove failure.
"""
import csv
import hmac
import io
import json
import os
import secrets
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError, model_validator
from starlette.middleware.sessions import SessionMiddleware
from chatbot import answer
from data_preprocessing import DATASET, FEATURES, load_dataset, validate_student_csv
from predict import score_students, study_recommendation
from reports import alert_rows, dashboard_summary, grouped_by_course, stage_overview, grade_overview, school_context
from time_series import chart_points, weekly_trend, semester_comparison, load_optional_history
from xapi_simulation import run_xapi_simulation, simulated_chart_points, METRIC_LABELS, STAGES, SEMESTERS
from train_models import train

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
MODELS = BASE / 'models'
INSTANCE = BASE / 'instance'
DEMO_MODE = os.getenv('DEMO_MODE', 'true').strip().lower() in ('1', 'true', 'yes')
SECRET = os.getenv('APP_SECRET_KEY', '')
if not SECRET or 'CHANGE_ME' in SECRET:
    if not DEMO_MODE:
        raise RuntimeError('Configure a long APP_SECRET_KEY before restricted local use.')
    SECRET = secrets.token_urlsafe(48)
app = FastAPI(title='School Learner Performance Predictor & Study Tutor', docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=SECRET, same_site='lax',
                   https_only=os.getenv('COOKIE_SECURE', 'false').lower() == 'true')
app.mount('/static', StaticFiles(directory=str(BASE / 'static')), name='static')
templates = Jinja2Templates(directory=str(BASE / 'templates'))


def asset_ready():
    if not DATASET.exists():
        raise RuntimeError('Missing data/xAPI-Edu-Data.csv. Restore the bundled dataset.')
    if not (MODELS / 'risk_model.joblib').exists() or not (MODELS / 'metrics.json').exists():
        train(DATASET, MODELS)


def is_admin(request: Request):
    return bool(request.session.get('is_admin'))


def authorised(request: Request):
    return DEMO_MODE or is_admin(request)


def needs_access(request: Request):
    if authorised(request):
        return None
    raise HTTPException(403, 'This feature is unavailable without authorised teacher access.')


def csrf_token(request: Request):
    if 'csrf' not in request.session:
        request.session['csrf'] = secrets.token_urlsafe(32)
    return request.session['csrf']


def check_csrf(request: Request, token: str):
    if not token or not hmac.compare_digest(str(request.session.get('csrf', '')), str(token)):
        raise HTTPException(403, 'Invalid CSRF token. Reload the page.')


def render(request: Request, page: str, **context):
    return templates.TemplateResponse(request=request, name=page, context={
        'request': request, 'demo_mode': DEMO_MODE, 'admin': is_admin(request), **context})


def student_frames():
    asset_ready()
    imported = INSTANCE / 'private_students.csv'
    using_private = not DEMO_MODE and imported.exists()
    frame = validate_student_csv(imported) if using_private else load_dataset()
    scored = score_students(frame, MODELS / 'risk_model.joblib').sort_values('risk_probability', ascending=False)
    history = load_optional_history(DATA / 'weekly_history.csv') if not using_private else pd.DataFrame()
    return scored, history, using_private


def metrics():
    asset_ready()
    return json.loads((MODELS / 'metrics.json').read_text(encoding='utf-8'))


@app.get('/health')
def health():
    return {'status': 'ok', 'audience': 'school-education', 'dataset': 'xAPI-Edu-Data', 'mode': 'public-benchmark' if DEMO_MODE else 'restricted'}


@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    asset_ready()
    return render(request, 'home.html')


@app.get('/dashboard', response_class=HTMLResponse)
def dashboard(request: Request):
    guard = needs_access(request)
    if guard: return guard
    students, hist, private = student_frames()
    trend = weekly_trend(hist)
    return render(request, 'dashboard.html', summary=dashboard_summary(students),
        alerts=alert_rows(students, 8), courses=grouped_by_course(students),
        semesters=semester_comparison(students), stages=stage_overview(students),
        grades=grade_overview(students), school=school_context(students),
        trend=trend, chart=chart_points(trend),
        private=private, model_name=metrics()['selected_model'])


@app.get('/learners', response_class=HTMLResponse)
@app.get('/students', response_class=HTMLResponse, include_in_schema=False)
def students_page(request: Request, risk: str = '', q: str = '', stage: str = '', grade: str = ''):
    guard = needs_access(request)
    if guard: return guard
    students, _, private = student_frames()
    if risk in ('High', 'Moderate', 'Low'):
        students = students[students.risk == risk]
    if stage in ('lowerlevel', 'MiddleSchool', 'HighSchool'):
        students = students[students.StageID == stage]
    if grade in load_dataset().GradeID.unique():
        students = students[students.GradeID == grade]
    if q.strip():
        term = q.strip()[:80]
        students = students[students.student_name.str.contains(term, case=False, na=False, regex=False) |
                            students.student_id.str.contains(term, case=False, na=False, regex=False) |
                            students.Topic.str.contains(term, case=False, na=False, regex=False)]
    return render(request, 'students.html', students=students.head(500).to_dict('records'),
                  total=len(students), selected_risk=risk, selected_stage=stage, selected_grade=grade,
                  stages=stage_overview(load_dataset()), grades=sorted(load_dataset().GradeID.unique()),
                  query=q[:80], private=private)


@app.get('/learners/{student_id}', response_class=HTMLResponse)
@app.get('/students/{student_id}', response_class=HTMLResponse, include_in_schema=False)
def student_detail(request: Request, student_id: str):
    guard = needs_access(request)
    if guard: return guard
    students, hist, private = student_frames()
    match = students[students.student_id == student_id]
    if match.empty: raise HTTPException(404, 'Dataset record not found')
    student = match.iloc[0].to_dict()
    trend = weekly_trend(hist, student_id=student_id)
    return render(request, 'student.html', student=student, trend=trend,
                  chart=chart_points(trend), private=private)


@app.get('/forecast', response_class=HTMLResponse)
def forecast(request: Request, metric: str = 'VisITedResources', semester: str = 'all', stage: str = 'all'):
    guard = needs_access(request)
    if guard: return guard
    # This is ALWAYS generated from the bundled anonymous xAPI source, never from
    # uploaded personal records. The synthetic weeks are explicitly hypothetical.
    if metric not in METRIC_LABELS or semester not in SEMESTERS or stage not in STAGES:
        raise HTTPException(status_code=400, detail='Unsupported simulation filter.')
    dataset = load_dataset()
    simulation = run_xapi_simulation(dataset, metric=metric, semester=semester, stage=stage)
    return render(request, 'forecast.html', simulation=simulation,
                  plot=simulated_chart_points(simulation),
                  metric_options=METRIC_LABELS, stage_options=STAGES,
                  semester_options=SEMESTERS,
                  semesters=semester_comparison(dataset))


@app.get('/reports', response_class=HTMLResponse)
def reports_page(request: Request):
    guard = needs_access(request)
    if guard: return guard
    students, _, private = student_frames()
    return render(request, 'reports.html', summary=dashboard_summary(students),
                  groups=grouped_by_course(students), stages=stage_overview(students),
                  grades=grade_overview(students), school=school_context(students),
                  metrics=metrics(), private=private)


@app.get('/reports/download')
def download_report(request: Request):
    guard = needs_access(request)
    if guard: return guard
    students, _, _ = student_frames()
    fields = ['learner_id', 'Topic', 'StageID', 'GradeID', 'Semester', 'StudentAbsenceDays',
              'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion',
              'predicted_class', 'probability_low', 'risk', 'Class']
    fields = [c for c in fields if c in students]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(fields)
    for values in students[fields].itertuples(index=False, name=None):
        writer.writerow(["'" + x if isinstance(x, str) and x.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else x for x in values])
    return Response(output.getvalue(), media_type='text/csv', headers={
        'Content-Disposition': 'attachment; filename=xapi_performance_predictions.csv', 'Cache-Control': 'no-store'})


@app.get('/tutor', response_class=HTMLResponse)
def tutor(request: Request):
    return render(request, 'tutor.html')


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@app.post('/api/chat')
def chat_api(body: Question):
    return answer(body.question)


class StudentFeatures(BaseModel):
    raisedhands: float = Field(ge=0, le=100)
    VisITedResources: float = Field(ge=0, le=100)
    AnnouncementsView: float = Field(ge=0, le=100)
    Discussion: float = Field(ge=0, le=100)
    StageID: str = Field(pattern='^(lowerlevel|MiddleSchool|HighSchool)$')
    GradeID: str = Field(pattern='^G-(02|04|05|06|07|08|09|10|11|12)$')
    Topic: str = Field(pattern='^(IT|Math|Arabic|Science|English|Quran|Spanish|French|History|Biology|Chemistry|Geology)$')
    Semester: str = Field(pattern='^(F|S)$')
    StudentAbsenceDays: str = Field(pattern='^(Under-7|Above-7)$')

    @model_validator(mode='after')
    def valid_school_stage_grade(self):
        # Ground valid stage/grade pairs in the source CSV (not assumed country standards).
        observed = set(zip(load_dataset().StageID, load_dataset().GradeID))
        if (self.StageID, self.GradeID) not in observed:
            raise ValueError('This educational stage and grade pairing does not appear in the xAPI benchmark.')
        return self


@app.post('/api/predict')
def predict_api(request: Request, body: StudentFeatures):
    if not authorised(request): raise HTTPException(401, 'Admin access required')
    record = pd.DataFrame([body.model_dump()])
    scored = score_students(record, MODELS / 'risk_model.joblib').iloc[0]
    return {'predicted_class': scored.predicted_class,
            'probability_low': float(scored.probability_low), 'risk': scored.risk,
            'probability_middle': float(scored.probability_middle),
            'probability_high': float(scored.probability_high),
            'recommendations': scored.recommendations,
            'warning': 'School-level xAPI research benchmark: L/M/H describes performance groups, not a verified pass/fail, a specific mark, or a dropout prediction. For human review only.'}


@app.get('/try-prediction', response_class=HTMLResponse)
def try_prediction(request: Request):
    guard = needs_access(request)
    if guard: return guard
    frame = load_dataset()
    grades_by_stage = {str(stage): sorted(g.GradeID.unique()) for stage, g in frame.groupby('StageID')}
    return render(request, 'predict_form.html', token=csrf_token(request), result=None,
        grades_by_stage=grades_by_stage, subjects=sorted(frame.Topic.unique()),
        stages=['lowerlevel', 'MiddleSchool', 'HighSchool'])


@app.post('/try-prediction', response_class=HTMLResponse)
def try_prediction_action(request: Request, raisedhands: float = Form(...),
    VisITedResources: float = Form(...), AnnouncementsView: float = Form(...), Discussion: float = Form(...),
    StageID: str = Form(...), GradeID: str = Form(...), Topic: str = Form(...), Semester: str = Form(...),
    StudentAbsenceDays: str = Form(...), csrf: str = Form('')):
    if not authorised(request): raise HTTPException(401, 'Access denied')
    check_csrf(request, csrf)
    try:
        row = StudentFeatures(raisedhands=raisedhands, VisITedResources=VisITedResources,
            AnnouncementsView=AnnouncementsView, Discussion=Discussion, StageID=StageID,
            GradeID=GradeID, Topic=Topic, Semester=Semester, StudentAbsenceDays=StudentAbsenceDays)
    except ValidationError as exc:
        raise HTTPException(422, 'School activity inputs must match the available xAPI categories and valid stage/grade combinations.') from exc
    prediction = score_students(pd.DataFrame([row.model_dump()]), MODELS / 'risk_model.joblib').iloc[0]
    frame = load_dataset()
    grades_by_stage = {str(stage): sorted(g.GradeID.unique()) for stage, g in frame.groupby('StageID')}
    return render(request, 'predict_form.html', token=csrf_token(request), result=prediction.to_dict(),
        grades_by_stage=grades_by_stage, subjects=sorted(frame.Topic.unique()),
        stages=['lowerlevel', 'MiddleSchool', 'HighSchool'])




@app.get('/import', response_class=HTMLResponse)
def import_page(request: Request, message: str = ''):
    if DEMO_MODE: raise HTTPException(403, 'CSV import is disabled in public benchmark mode.')
    guard = needs_access(request)
    if guard: return guard
    return render(request, 'import.html', token=csrf_token(request), message=message)


@app.post('/import', response_class=HTMLResponse)
async def import_csv(request: Request, file: UploadFile = File(...), csrf: str = Form('')):
    if DEMO_MODE or not is_admin(request): raise HTTPException(403, 'Admin access required.')
    check_csrf(request, csrf)
    if not (file.filename or '').lower().endswith('.csv'):
        return render(request, 'import.html', token=csrf_token(request), message='Choose a CSV file.')
    payload = await file.read(2_000_001)
    if len(payload) > 2_000_000:
        return render(request, 'import.html', token=csrf_token(request), message='CSV exceeds 2 MB.')
    try:
        frame = validate_student_csv(io.BytesIO(payload))
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        return render(request, 'import.html', token=csrf_token(request), message=str(exc))
    INSTANCE.mkdir(parents=True, exist_ok=True)
    dest, temp = INSTANCE / 'private_students.csv', INSTANCE / 'private_students.tmp'
    frame.to_csv(temp, index=False)
    temp.replace(dest)
    return RedirectResponse('/dashboard', status_code=303)


@app.exception_handler(404)
def not_found(request: Request, exception):
    return templates.TemplateResponse(request=request, name='error.html', status_code=404, context={
        'request': request, 'demo_mode': DEMO_MODE, 'admin': is_admin(request),
        'code': 404, 'message': 'The page or dataset record could not be found.'})
