"""Honest time-series module: xAPI contains no timestamps; optional dated records only.

Do NOT infer weeks by row index or equate different students across semesters.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def semester_comparison(df):
    """Cross-sectional semester-group comparison; this is NOT longitudinal history."""
    if 'Semester' not in df or 'Class' not in df:
        return []
    result = []
    for name, group in df.groupby('Semester'):
        result.append({
            'semester': 'First (F)' if name == 'F' else 'Second (S)' if name == 'S' else str(name),
            'records': len(group),
            'low': int((group.Class == 'L').sum()),
            'middle': int((group.Class == 'M').sum()),
            'high': int((group.Class == 'H').sum()),
            'average_visits': round(float(group.VisITedResources.mean()), 1),
            'low_percent': round(float((group.Class == 'L').mean() * 100), 1),
        })
    return sorted(result, key=lambda x: x['semester'])


def load_optional_history(path='data/weekly_history.csv'):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in ('week_start', 'VisITedResources'):
        if c not in df:
            raise ValueError('Optional history needs week_start and VisITedResources columns.')
    df['week_start'] = pd.to_datetime(df['week_start'], errors='raise')
    df['VisITedResources'] = pd.to_numeric(df['VisITedResources'], errors='raise')
    if df['VisITedResources'].isna().any() or not df['VisITedResources'].between(0, 100).all():
        raise ValueError('Visits must be between 0 and 100.')
    return df


def weekly_trend(history=None, student_id=None, periods=3):
    """Observed dated weekly visits and a baseline forecast only with real dates."""
    if history is None or history.empty:
        return {'actual': [], 'forecast': [],
                'method': 'Unavailable: xAPI-Edu-Data has no dated weekly observations. Optional weekly_history.csv required.'}
    data = history.copy()
    if student_id:
        if 'student_id' not in data:
            return {'actual': [], 'forecast': [], 'method': 'History has no matching student identifier.'}
        data = data[data.student_id.astype(str) == str(student_id)]
    if data.empty:
        return {'actual': [], 'forecast': [], 'method': 'No dated observations for this record.'}
    data['week_start'] = pd.to_datetime(data['week_start'])
    grouped = data.groupby('week_start', as_index=False)['VisITedResources'].mean().sort_values('week_start')
    points = [{'week': i, 'visits': round(float(r.VisITedResources), 1),
               'date': str(r.week_start.date())} for i, r in enumerate(grouped.itertuples(index=False), start=1)]
    if len(points) < 3:
        return {'actual': points, 'forecast': [], 'method': 'At least three distinct dated weeks required for a forecast.'}
    recent = points[-min(5, len(points)):]
    slope, intercept = np.polyfit([x['week'] for x in recent], [x['visits'] for x in recent], 1)
    last_date = grouped.week_start.iloc[-1]
    future = []
    for step in range(1, periods + 1):
        week = len(points) + step
        future.append({'week': week, 'visits': round(float(np.clip(slope * week + intercept, 0, 100)), 1),
                       'date': str((last_date + pd.Timedelta(weeks=step)).date())})
    return {'actual': points, 'forecast': future,
            'method': 'Baseline straight-line extrapolation of last five dated observations; illustrative, not validated.'}

def chart_points(data, width=650, height=240):
    points = data['actual'] + data['forecast']
    if not points:
        return {'actual': '', 'forecast': '', 'ticks': [], 'height': height, 'width': width}
    lo, hi = min(x['week'] for x in points), max(x['week'] for x in points)
    def loc(pt):
        x = 46 + (pt['week']-lo) * (width-80) / max(1, hi-lo)
        y = height - 30 - pt['visits'] * (height-60) / 100
        return f'{x:.1f},{y:.1f}'
    return {'actual': ' '.join(loc(p) for p in data['actual']),
            'forecast': ' '.join(loc(p) for p in data['actual'][-1:] + data['forecast']),
            'ticks': [{'label': f'W{p["week"]}', 'x': 46+(p['week']-lo)*(width-80)/max(1,hi-lo)}
                      for p in points], 'width': width, 'height': height}
