"""Explicitly SYNTHETIC weekly activity demonstration derived from xAPI-Edu-Data.

The xAPI source has learner-level totals, semester labels and NO event timestamps.
We simulate how each learner's source activity total *might* be spread across
8 hypothetical weeks, with a fixed random seed. All resulting weekly values,
backtests and forecasts are simulation results, NOT historical observations.

The generator does not use `Class`, inferred grades, or learner IDs as targets;
the single linked source measurement is the selected activity counter.
"""

import numpy as np
import pandas as pd

from data_preprocessing import NUMERIC_FEATURES

WEEKS = 8
FUTURE_WEEKS = 3
SEED = 2026
METRIC_LABELS = {
    'VisITedResources': 'LMS resource visits',
    'raisedhands': 'Class participation (raised hands)',
    'AnnouncementsView': 'Announcements viewed',
    'Discussion': 'Discussion activity',
}
METHODS = {
    'last_value': 'Previous-week baseline',
    'moving_average': 'Three-week moving average',
    'linear_trend': 'Four-week linear trend',
}
STAGES = {
    'all': 'All school stages',
    'lowerlevel': 'Lower level',
    'MiddleSchool': 'Middle school',
    'HighSchool': 'High school',
}
SEMESTERS = {'all': 'Both semesters (pooled)', 'F': 'First semester (F)', 'S': 'Second semester (S)'}


def generate_weekly_allocations(df, metric='VisITedResources', *, seed=SEED, weeks=WEEKS):
    """Create integer-valued synthetic weeks, preserving each real source total.

    Uses one reproducible hypothetical weekly distribution shared by the group,
    with independent multinomial allocation per dataset row. This is a modelling
    assumption, not a discovered actual weekly pattern.
    """
    if metric not in METRIC_LABELS or metric not in NUMERIC_FEATURES:
        raise ValueError('Unsupported xAPI activity metric.')
    if weeks < 5 or weeks > 20:
        raise ValueError('Simulation requires between 5 and 20 hypothetical weeks.')
    if df.empty or metric not in df:
        raise ValueError('The xAPI data and requested metric must be present.')
    values = pd.to_numeric(df[metric], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 100).any() or (values % 1 != 0).any():
        raise ValueError('The source activity counter must consist of integers from 0 to 100.')
    rng = np.random.default_rng(seed + list(METRIC_LABELS).index(metric) * 301)
    # One assumed week-by-week weighting (not measured seasonality).
    hypothetical_week_weights = rng.dirichlet(np.full(weeks, 12.0))
    allocations = np.array([rng.multinomial(int(v), hypothetical_week_weights)
                            for v in values], dtype=int)
    assert np.array_equal(allocations.sum(axis=1), values.astype(int))
    return allocations, hypothetical_week_weights


def forecast_next(observed, method):
    """One-step forecast based *only* on earlier synthetic weekly values."""
    vals = np.asarray(observed, dtype=float)
    if len(vals) < 4 or not np.isfinite(vals).all() or method not in METHODS:
        raise ValueError('At least four finite weekly values and a known method are required.')
    if method == 'last_value':
        value = vals[-1]
    elif method == 'moving_average':
        value = float(vals[-3:].mean())
    else:
        value = float(np.polyval(np.polyfit(np.arange(4), vals[-4:], 1), 4))
    return round(float(np.clip(value, 0, 100)), 3)


def evaluate_simulated_series(values, horizon=FUTURE_WEEKS):
    """Expanding-window one-step comparisons, then iterative synthetic extension."""
    values = np.asarray(values, dtype=float)
    if len(values) < 6 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Six or more finite simulated weekly values are needed.')
    if not 1 <= horizon <= 6:
        raise ValueError('Use a forecast horizon between one and six weeks.')
    backtests = []
    for i in range(4, len(values)):
        observed = values[:i]
        trial = {'week': i+1, 'simulated': round(float(values[i]), 3)}
        for code in METHODS:
            trial[code] = forecast_next(observed, code)
            trial[code+'_error'] = round(abs(trial[code] - values[i]), 3)
        backtests.append(trial)
    scores = [
        {'code': code, 'name': name,
         'mae': round(float(np.mean([x[code+'_error'] for x in backtests])), 3),
         'n': len(backtests)}
        for code, name in METHODS.items()
    ]
    best = min(scores, key=lambda x: (x['mae'], list(METHODS).index(x['code'])))
    seq = values.tolist()
    future = []
    for week in range(len(seq) + 1, len(seq) + horizon + 1):
        value = forecast_next(seq, best['code'])
        future.append({'week': week, 'value': value})
        seq.append(value)
    return {'backtest': backtests, 'models': scores, 'chosen': best, 'forecast': future}


def run_xapi_simulation(df, *, metric='VisITedResources', semester='all', stage='all',
                        seed=SEED):
    """Report simulated weekly means for one xAPI cohort, with honest labels."""
    if semester not in SEMESTERS or stage not in STAGES:
        raise ValueError('Unknown xAPI school-stage or semester filter.')
    if 'Semester' not in df or 'StageID' not in df:
        raise ValueError('Expected xAPI cohort fields not present.')
    allocations, weights = generate_weekly_allocations(df, metric, seed=seed)
    mask = np.ones(len(df), dtype=bool)
    if semester != 'all':
        mask &= (df['Semester'].to_numpy() == semester)
    if stage != 'all':
        mask &= (df['StageID'].to_numpy() == stage)
    subset = allocations[mask]
    if len(subset) < 6:
        raise ValueError('Select a cohort with at least six source records.')
    means = subset.mean(axis=0).round(3)
    results = evaluate_simulated_series(means)
    return {
        'metric': metric, 'metric_label': METRIC_LABELS[metric],
        'semester': semester, 'stage': stage,
        'semester_label': SEMESTERS[semester], 'stage_label': STAGES[stage],
        'learners': int(len(subset)), 'seed': int(seed),
        'weeks': [{'week': i + 1, 'value': round(float(val), 3)}
                  for i, val in enumerate(means)],
        'forecast': results['forecast'], 'models': results['models'],
        'backtest': results['backtest'], 'chosen': results['chosen'],
        'mean_source_total': round(float(subset.sum(axis=1).mean()), 2),
        'source_total': int(subset.sum()),
        'allocation_total': int(subset.sum()),
        'assumptions': (
            'For each xAPI record, the recorded activity counter is randomly allocated '
            'across 8 hypothetical weeks using a fixed seed (2026). The allocations '
            'sum exactly to the real source counter but their timing is invented. '
            'Group means, backtest errors and forecasts are therefore simulated, '
            'not observed or externally validated.'
        ),
    }


def simulated_chart_points(report, width=860, height=300):
    """SVG coordinates with y-axis starting at zero, clear forecast boundary."""
    all_points = report['weeks'] + report['forecast']
    last_week = all_points[-1]['week']
    max_y = max(1.0, max(p['value'] for p in all_points)*1.25)
    max_y = float(np.ceil(max_y*2)/2)
    xmin, xmax, ymin, ymax = 56, width-32, 24, height-54

    def xy(p):
        return (round(xmin+(p['week']-1)*(xmax-xmin)/(last_week-1), 1),
                round(ymax-p['value']*(ymax-ymin)/max_y, 1))

    actual = [{**p, 'x': xy(p)[0], 'y': xy(p)[1]} for p in report['weeks']]
    future = [{**p, 'x': xy(p)[0], 'y': xy(p)[1]} for p in report['forecast']]
    return {
        'width': width, 'height': height, 'actual': actual, 'future': future,
        'actual_path': ' '.join(f"{p['x']},{p['y']}" for p in actual),
        'future_path': ' '.join(f"{p['x']},{p['y']}" for p in actual[-1:]+future),
        'boundary': round((actual[-1]['x']+future[0]['x'])/2, 1),
        'ticks': [{'value': round(max_y*i/4, 1), 'y': round(ymax-(ymax-ymin)*i/4, 1)}
                  for i in range(5)],
        'xlabel': [{'week': p['week'], 'x': p['x']} for p in actual+future],
    }
