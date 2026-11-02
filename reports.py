"""Anonymous dataset dashboard aggregates and human-review priority lists."""

def dashboard_summary(scored):
    n = len(scored)
    return {
        'students': n,
        'high': int((scored.risk == 'High').sum()),
        'moderate': int((scored.risk == 'Moderate').sum()),
        'low': int((scored.risk == 'Low').sum()),
        'observed_low': int((scored.Class == 'L').sum()) if 'Class' in scored else None,
        'average_resources': round(float(scored.VisITedResources.mean()), 1) if n else 0,
        'above_seven_absences': int((scored.StudentAbsenceDays == 'Above-7').sum()),
    }


def grouped_by_course(scored):
    rows = []
    for topic, group in scored.groupby('Topic'):
        rows.append({
            'course': str(topic), 'students': len(group),
            'high': int((group.risk == 'High').sum()),
            'average_resources': round(float(group.VisITedResources.mean()), 1),
        })
    return sorted(rows, key=lambda row: (-row['high'], row['course']))


def alert_rows(scored, limit=12):
    return scored[scored.risk == 'High'].sort_values('risk_probability', ascending=False).head(limit).to_dict('records')


STAGE_NAMES = {
    'lowerlevel': 'Lower level',
    'MiddleSchool': 'Middle school',
    'HighSchool': 'High school',
}


def stage_overview(scored):
    """School-stage summaries; observed Class is separate from predicted review bands."""
    rows = []
    for stage in ('lowerlevel', 'MiddleSchool', 'HighSchool'):
        sub = scored[scored.StageID == stage]
        if sub.empty:
            continue
        rows.append({
            'code': stage, 'stage': STAGE_NAMES[stage], 'records': len(sub),
            'low': int((sub.Class == 'L').sum()) if 'Class' in sub else None,
            'middle': int((sub.Class == 'M').sum()) if 'Class' in sub else None,
            'high': int((sub.Class == 'H').sum()) if 'Class' in sub else None,
            'review': int((sub.risk == 'High').sum()) if 'risk' in sub else None,
            'mean_visits': round(float(sub.VisITedResources.mean()), 1),
            'grades': ', '.join(sorted(sub.GradeID.unique())),
        })
    return rows


def grade_overview(scored):
    """Per-grade totals without assuming the grade conventions of any country."""
    rows = []
    for grade, sub in sorted(scored.groupby('GradeID')):
        rows.append({
            'grade': grade, 'records': len(sub),
            'low': int((sub.Class == 'L').sum()) if 'Class' in sub else None,
            'high_priority': int((sub.risk == 'High').sum()) if 'risk' in sub else None,
            'mean_visits': round(float(sub.VisITedResources.mean()), 1),
        })
    return rows


def school_context(scored):
    """Descriptive parental fields only, never inputs to the prediction model."""
    result = {}
    if 'ParentAnsweringSurvey' in scored:
        result['survey_yes'] = int((scored.ParentAnsweringSurvey == 'Yes').sum())
        result['survey_total'] = int(scored.ParentAnsweringSurvey.notna().sum())
    if 'ParentschoolSatisfaction' in scored:
        result['satisfaction_good'] = int((scored.ParentschoolSatisfaction == 'Good').sum())
        result['satisfaction_total'] = int(scored.ParentschoolSatisfaction.notna().sum())
    return result
