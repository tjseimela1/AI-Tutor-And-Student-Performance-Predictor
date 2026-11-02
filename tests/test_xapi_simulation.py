"""Check hypothetical-week generation, honest labels, and forecast isolation."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app import app
from data_preprocessing import load_dataset
from xapi_simulation import (generate_weekly_allocations, forecast_next,
                             evaluate_simulated_series, run_xapi_simulation,
                             simulated_chart_points, METHODS, METRIC_LABELS)


def test_source_activity_totals_preserved_for_every_record_and_metric():
    records = load_dataset()
    for metric in METRIC_LABELS:
        distribution, weights = generate_weekly_allocations(records, metric)
        assert distribution.shape == (480, 8)
        assert np.array_equal(distribution.sum(axis=1), records[metric].to_numpy(dtype=int))
        assert np.all(distribution >= 0)
        assert np.isclose(weights.sum(), 1)


def test_source_is_not_modified_and_seeded_run_is_repeatable():
    source = load_dataset()
    copy = source.copy(deep=True)
    r1 = run_xapi_simulation(source)
    r2 = run_xapi_simulation(source)
    assert r1 == r2
    assert source.equals(copy)
    assert r1['learners'] == 480
    assert len(r1['weeks']) == 8 and len(r1['forecast']) == 3
    assert r1['source_total'] == r1['allocation_total'] == int(source.VisITedResources.sum())


def test_performance_label_not_used_to_generate_timeline():
    source = load_dataset()
    original = run_xapi_simulation(source)
    source['Class'] = source['Class'].map({'L': 'H', 'M': 'L', 'H': 'M'})
    changed = run_xapi_simulation(source)
    assert changed['weeks'] == original['weeks']
    assert changed['forecast'] == original['forecast']


def test_filters_match_real_subset_size_but_output_is_simulated():
    source = load_dataset()
    result = run_xapi_simulation(source, metric='Discussion', semester='S', stage='HighSchool')
    selected = source[(source.Semester == 'S') & (source.StageID == 'HighSchool')]
    assert result['learners'] == len(selected) == 10
    assert result['source_total'] == int(selected.Discussion.sum())
    assert round(sum(week['value'] for week in result['weeks']), 2) == pytest.approx(
        selected.Discussion.mean(), abs=0.01)
    assert 'imaginary' in result['assumptions'] or 'hypothetical' in result['assumptions']


def test_backtests_only_look_back_at_prior_generated_values():
    values = [3.0, 3.5, 4.0, 5.0, 4.3, 5.2, 4.0, 6.5]
    report = evaluate_simulated_series(values)
    assert [r['week'] for r in report['backtest']] == [5, 6, 7, 8]
    for i, row in enumerate(report['backtest'], start=4):
        for method in METHODS:
            assert row[method] == forecast_next(values[:i], method)
    assert report['chosen']['code'] in METHODS
    assert len(report['forecast']) == 3
    assert report['forecast'][0]['week'] == 9


def test_chart_correctly_segments_simulated_versus_projected_weeks():
    report = run_xapi_simulation(load_dataset())
    chart = simulated_chart_points(report)
    assert len(chart['actual']) == 8
    assert len(chart['future']) == 3
    assert chart['boundary'] > chart['actual'][-1]['x']
    assert chart['boundary'] < chart['future'][0]['x']
    assert len(chart['ticks']) == 5


def test_simulation_api_and_clear_disclaimers():
    client = TestClient(app)
    response = client.get('/forecast')
    assert response.status_code == 200
    for text in ('SIMULATED TIME SERIES', 'no timestamps', 'SIMULATION',
                 'Eight', 'Walk-forward backtest', 'Real xAPI semester comparison'):
        assert text.lower() in response.text.lower()
    alternate = client.get('/forecast?stage=HighSchool&semester=S&metric=raisedhands')
    assert alternate.status_code == 200
    assert '10' in alternate.text
    assert 'Class participation (raised hands)' in alternate.text
    assert client.get('/forecast?metric=Class').status_code == 400
    assert client.get('/forecast?stage=University').status_code == 400

