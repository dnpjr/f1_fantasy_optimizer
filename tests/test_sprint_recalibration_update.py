"""Raw Dutch snapshot -> canonical history -> fitted production bonus regressions."""
import json
import math
import platform
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1fantasy.historical_scores import DEFAULT_CANONICAL_DATASET_PATH, load_canonical_scores
from f1fantasy.model import normalise_sprint_baseline_inputs
from f1fantasy.sprint_shadow import load_sprint_production_calibration
from scripts.build_historical_fantasy_scores import _snapshot_official, _schedules
from scripts.build_2026_sprint_final_candidate import prepare_candidate_data, run_build
from scripts.recalibrate_sprint_ev import runtime_candidate_payload

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/generated/sprint_ev_calibration/sources/official_2026_round_12.json'
MARKET = SOURCE.parent / 'market_2026_round_13.json'
SCHEDULE = ROOT / 'data/cache/schedule_2026.csv'
PRIOR = ROOT / 'reports/2026_sprint_partial_pooling'



# CI's x86_64 fit differs from macOS arm64 by at most two binary64 ULPs.
# Allow four ULPs only on explicitly fitted scalars, never whole model trees.
FITTED_FIELDS = {
    'driver': ('form_mean', 'form_sd', 'group_intercept', 'group_slope',
               'within_variance', 'tau_squared'),
    'constructor': ('intercept', 'slope'),
}


def _assert_refitted_model(actual, expected, entity):
    assert actual.keys() == expected.keys()
    for field in FITTED_FIELDS[entity]:
        reference = expected[field]
        assert type(actual[field]) is type(reference) is float
        assert math.isfinite(actual[field]) and math.isfinite(reference)
        assert actual[field] == pytest.approx(reference, rel=0, abs=4 * math.ulp(reference)), field
    # Includes ordered personal histories, IDs, counts, names, means and fixed weights.
    assert {k: v for k, v in actual.items() if k not in FITTED_FIELDS[entity]} == {
        k: v for k, v in expected.items() if k not in FITTED_FIELDS[entity]
    }


@pytest.mark.parametrize('entity,field', [
    (entity, field) for entity, fields in FITTED_FIELDS.items() for field in fields
])
def test_refit_comparison_rejects_parameter_drift(entity, field):
    active = json.loads((SOURCE.parent.parent / 'sprint_ev_2026_v1.json').read_text())[entity]
    changed = deepcopy(active)
    changed[field] += 4 * math.ulp(active[field])
    _assert_refitted_model(changed, active, entity)
    changed[field] += math.ulp(active[field])
    with pytest.raises(AssertionError):
        _assert_refitted_model(changed, active, entity)


@pytest.mark.parametrize('mutation', ['count', 'identity', 'order', 'history_mean', 'extra_key'])
def test_refit_comparison_keeps_history_exact(mutation):
    active = json.loads((SOURCE.parent.parent / 'sprint_ev_2026_v1.json').read_text())['driver']
    changed = deepcopy(active)
    if mutation == 'count':
        changed['personal_history'][0]['observation_count'] += 1
    elif mutation == 'identity':
        changed['personal_history'][0]['canonical_entity_id'] = 'unexpected'
    elif mutation == 'order':
        changed['personal_history'].reverse()
    elif mutation == 'history_mean':
        value = changed['personal_history'][0]['personal_mean_bonus']
        changed['personal_history'][0]['personal_mean_bonus'] = math.nextafter(value, math.inf)
    else:
        changed['unexpected'] = 0
    with pytest.raises(AssertionError):
        _assert_refitted_model(changed, active, 'driver')


def test_raw_snapshot_replays_exact_new_rows_and_canonical_keys():
    added, warnings = _snapshot_official(SOURCE, _schedules())
    assert not warnings
    canonical = load_canonical_scores(DEFAULT_CANONICAL_DATASET_PATH)
    latest = canonical[canonical.season.eq(2026) & canonical['round'].eq(12)]
    pd.testing.assert_frame_equal(added.fillna("").reset_index(drop=True), latest.fillna("").reset_index(drop=True), check_dtype=False)
    assert len(latest) == 33
    assert latest.groupby('entity_type').size().to_dict() == {'driver': 22, 'constructor': 11}
    assert not canonical.duplicated(['season', 'round', 'entity_type', 'canonical_entity_id']).any()
    assert latest.event_name.eq('Dutch Grand Prix').all()
    assert latest.event_date.eq('2026-08-23').all()
    lawson = latest[latest.canonical_entity_id.eq('lawson')].iloc[0]
    assert str(lawson.source_entity_id) == '116'
    assert 'tsunoda' in set(latest.canonical_entity_id)
    assert 'hadjar' not in set(latest.canonical_entity_id)


def test_truncated_snapshot_is_rejected(tmp_path):
    payload = json.loads(SOURCE.read_text())
    payload['payloads'] = [p for p in payload['payloads'] if p['player_id'] != 18]
    path = tmp_path / 'partial.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Incomplete snapshot roster'):
        _snapshot_official(path, _schedules())


def test_refit_reproduces_active_parameters_and_all_observations(tmp_path, capsys):
    result = run_build(DEFAULT_CANONICAL_DATASET_PATH, SCHEDULE, MARKET, PRIOR, tmp_path, maximum_round=None)
    observations = result['prepared']['observations']
    assert set(observations['round']) == {2, 4, 5, 9, 12}
    assert not observations.duplicated(['season', 'round', 'entity_type', 'entity_id']).any()
    valid = observations[observations.observation_valid]
    assert valid.groupby('entity_type').size().to_dict() == {'driver': 104, 'constructor': 53}
    new = runtime_candidate_payload(result, 'sprint_ev_2026_v1')
    active = json.loads((SOURCE.parent.parent / 'sprint_ev_2026_v1.json').read_text())
    # Emit both models even on success so CI numerical differences remain inspectable.
    with capsys.disabled():
        print('\nSprint refit environment:', platform.system(), platform.machine(),
              platform.python_version(), 'numpy', np.__version__, 'pandas', pd.__version__,
              'BLAS', getattr(np.__config__, 'CONFIG', {}).get('Build Dependencies', {}).get('blas', {}).get('name'))
        for entity, fields in FITTED_FIELDS.items():
            for field in fields:
                ref, actual = active[entity][field], new[entity][field]
                print('Sprint refit:', json.dumps({
                    'field': f'{entity}.{field}', 'reference': ref, 'actual': actual,
                    'absolute_delta': abs(actual - ref),
                    'relative_delta': abs((actual - ref) / ref) if ref else None,
                    'ulps': abs(actual - ref) / math.ulp(ref),
                }, sort_keys=True))
    for entity in FITTED_FIELDS:
        _assert_refitted_model(new[entity], active[entity], entity)
    for field in ('model_version', 'sprint_rounds', 'completed_rounds',
                  'calibration_season', 'source_data_version', 'source_research_model'):
        assert new[field] == active[field], field
    # Personal history follows human identity across changed Fantasy IDs.
    history = {r['canonical_entity_id']: r for r in new['driver']['personal_history']}
    assert history['hadjar']['observation_count'] == 4
    assert history['lawson']['observation_count'] == 4  # Dutch component missing
    assert history['tsunoda']['observation_count'] == 1
    assert history['hulkenberg']['personal_mean_bonus'] == pytest.approx(-4.2)


def test_missing_components_are_omitted_and_old_targets_are_unchanged():
    prepared = prepare_candidate_data(DEFAULT_CANONICAL_DATASET_PATH, SCHEDULE, MARKET, maximum_round=None)
    history = prepared['history']
    latest = history[history['round'].eq(12)].set_index('canonical_entity_id')
    for asset in ('lawson', 'bortoleto', 'arvid_lindblad', 'cadillac'):
        assert pd.isna(latest.loc[asset, 'normalised_score'])
    assert latest.loc['hulkenberg', 'extra_sprint_points'] == -10
    assert latest.loc['hulkenberg', 'normalised_score'] == 15
    previous = pd.read_csv(ROOT / 'reports/2026_personalised_sprint_adjustments/sprint_observations.csv')
    actual = prepared['observations']
    keys = ['season', 'round', 'entity_type', 'entity_id']
    merged = previous.merge(actual, on=keys, suffixes=('_old', '_new'), validate='one_to_one')
    assert len(merged) == len(previous)
    for field in ('extra_sprint_points', 'base_weekend_points'):
        np.testing.assert_allclose(merged[field+'_old'], merged[field+'_new'], equal_nan=True)


@pytest.mark.parametrize('problem', ['duplicate', 'partial', 'future'])
def test_invalid_canonical_update_is_rejected(tmp_path, problem):
    data = pd.read_csv(DEFAULT_CANONICAL_DATASET_PATH)
    latest = data.season.eq(2026) & data['round'].eq(12)
    if problem == 'duplicate':
        data = pd.concat([data, data[latest].iloc[[0]]])
    elif problem == 'partial':
        data = data.drop(data[latest].index[0])
    else:
        data.loc[latest, 'event_date'] = '2099-01-01'
    path = tmp_path / 'invalid.csv'
    data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        prepare_candidate_data(path, SCHEDULE, MARKET, maximum_round=None)


def test_baseline_removes_sprint_once_preserves_gp_and_missing_splits():
    rows = pd.DataFrame([
        {'season': 2026, 'round': 1, 'weekend_points': 25, 'sprint_points': np.nan},
        {'season': 2026, 'round': 2, 'weekend_points': 40, 'sprint_points': 10},
        {'season': 2026, 'round': 2, 'weekend_points': 5, 'sprint_points': np.nan, 'sprint_qualifying_points': -10},
        {'season': 2025, 'round': 2, 'weekend_points': 50, 'sprint_points': np.nan},
    ])
    original = rows.copy(deep=True)
    result = normalise_sprint_baseline_inputs(rows, {(2026, 2), (2025, 2)})
    assert result.weekend_points.tolist() == [25, 30, 15]
    pd.testing.assert_frame_equal(rows, original)
    # A future bonus is added to GP-equivalent form, not whole-weekend form.
    assert result.weekend_points.mean() + 7 == pytest.approx(30.333333333333332)
