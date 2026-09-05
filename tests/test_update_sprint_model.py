"""Maintenance orchestration tests; all imports/promotions use isolated files."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import update_sprint_model as maintenance


ROOT = Path(__file__).resolve().parents[1]


def hashes(directory):
    return {str(p.relative_to(directory)): maintenance.sha(p) for p in directory.rglob('*') if p.is_file()}


@pytest.fixture
def scenario(tmp_path):
    canonical_dir = tmp_path / 'canonical'
    shutil.copytree(maintenance.DEFAULT_CANONICAL_DATASET_PATH.parent, canonical_dir)
    active = tmp_path / 'calibration' / 'active.json'
    active.parent.mkdir()
    frozen_fit = ROOT / 'reports/sprint_recalibration_2026_round_12/runtime_calibration_candidate.json'
    baseline = json.loads(frozen_fit.read_text())
    baseline['calibration_status'] = 'approved_production'
    maintenance.dump(active, baseline)
    canonical = canonical_dir / maintenance.DEFAULT_CANONICAL_DATASET_PATH.name
    schedule_dir = tmp_path / 'cache'
    schedule_dir.mkdir()
    schedule = schedule_dir / maintenance.DEFAULT_SCHEDULE.name
    metadata = pd.read_csv(maintenance.DEFAULT_SCHEDULE)
    data = pd.read_csv(canonical)
    data = data[~data.season.eq(maintenance.SEASON) | data['round'].isin(baseline['completed_rounds'])]
    data.to_csv(canonical, index=False)
    data.to_parquet(canonical.with_suffix('.parquet'), index=False)
    last_round = int(data.loc[data.season.eq(maintenance.SEASON), 'round'].max())
    next_round = last_round + 1
    previous = metadata[metadata['round'].eq(last_round)].iloc[0]
    event_date = pd.Timestamp(previous.date) + pd.Timedelta(days=1)
    metadata.loc[metadata['round'].eq(next_round), ['raceName', 'date', 'sprint_date', 'sprint_time']] = [
        'Maintenance Test Grand Prix', event_date.date().isoformat(),
        (event_date - pd.Timedelta(days=1)).date().isoformat(), '10:00:00Z',
    ]
    metadata.to_csv(schedule, index=False)
    market = schedule_dir / 'market.json'
    shutil.copy2(maintenance.DEFAULT_SOURCES / "market_2026_round_13.json", market)
    # A retained real payload supplies the schema; all event identities are
    # relabelled as an explicitly synthetic next-round fixture, never production.
    snapshot = json.loads((maintenance.DEFAULT_SOURCES / 'official_2026_round_12.json').read_text())
    snapshot['round'] = next_round
    for item in snapshot['payloads']:
        for match in item['payload']['Value'].get('MatchWiseStats', []):
            for session in match['RaceDayWise']:
                session['MeetingNumber'] = next_round
                session['MeetingName'] = 'Maintenance Test Grand Prix'
    sources = tmp_path / 'sources'
    sources.mkdir()
    source = sources / 'official_synthetic.json'
    maintenance.dump(source, snapshot)
    options = dict(canonical=canonical, active=active, schedule=schedule,
                   market=market, sources=sources, output=tmp_path / 'updates')
    return SimpleNamespace(**options, options=options, source=source, round=next_round,
                           snapshot=snapshot, canonical_dir=canonical_dir)


def candidate(scenario):
    return Path(maintenance.create_update(**scenario.options)['bundle'])


def pass_checks(monkeypatch, bundle):
    def successful_run(command, **kwargs):
        assert all(test in command for test in maintenance.TESTS)
        return SimpleNamespace(returncode=0, stdout='Maintenance regression suite passed (mock subprocess).\n')
    monkeypatch.setattr(maintenance.subprocess, 'run', successful_run)
    return maintenance.check_bundle(bundle)


def test_up_to_date_is_noop_with_no_fetch_or_write(monkeypatch, scenario):
    # Freeze the completed catalogue too: this test must still pass when the real
    # repository is awaiting its next maintenance update.
    schedule = pd.read_csv(scenario.schedule)
    schedule[schedule['round'].lt(scenario.round)].to_csv(scenario.schedule, index=False)
    canonical_before = hashes(scenario.canonical_dir)
    active_before = maintenance.sha(scenario.active)
    def forbidden(*args, **kwargs):
        raise AssertionError('No new weekend must not fetch or fit')
    monkeypatch.setattr(maintenance, 'capture_snapshots', forbidden)
    monkeypatch.setattr(maintenance, 'generate_candidate', forbidden)
    for audit in (True, False):
        result = maintenance.create_update(**scenario.options, audit=audit, fetch=True)
        assert result['status'] == 'up_to_date'
        assert result['canonical_sprint_rounds'] == result['active_sprint_rounds']
    assert not scenario.output.exists()
    assert hashes(scenario.canonical_dir) == canonical_before
    assert maintenance.sha(scenario.active) == active_before


def test_audit_discovers_future_fixture_counts_without_mutation(scenario):
    before, active = hashes(scenario.canonical_dir), maintenance.sha(scenario.active)
    result = maintenance.create_update(**scenario.options, audit=True)
    assert result['rounds_to_import'] == [scenario.round]
    assert result['status'] == 'ready'
    assert len(result['available_observations']) == 2
    assert result['candidate_data']['samples']['driver'] > 0
    assert not scenario.output.exists()
    assert before == hashes(scenario.canonical_dir)
    assert active == maintenance.sha(scenario.active)


def test_missing_sources_are_reported_and_do_not_mean_completed(scenario):
    scenario.source.unlink()
    result = maintenance.create_update(**scenario.options)
    assert result['status'] == 'awaiting_sources'
    assert result['sources_unavailable'] == [scenario.round]
    assert not scenario.output.exists()


def test_fetch_is_opt_in_and_retained(monkeypatch, scenario):
    snapshot = deepcopy(scenario.snapshot)
    scenario.source.unlink()
    calls = []
    def fetch(rounds, market, canonical):
        calls.append(rounds)
        return {scenario.round: snapshot}
    monkeypatch.setattr(maintenance, 'capture_snapshots', fetch)
    bundle = Path(maintenance.create_update(**scenario.options, fetch=True)['bundle'])
    assert calls == [[scenario.round]]
    assert len(list((bundle / 'inputs/sources').glob('*.json'))) == 1


def test_capture_includes_inactive_ids_and_does_not_assume_gameday_is_round(monkeypatch, scenario):
    payloads = {p['player_id']: deepcopy(p['payload']) for p in scenario.snapshot['payloads']}
    for value in payloads.values():
        for key in ('GamedayWiseStats', 'MatchWiseStats'):
            for gameday in value['Value'][key]:
                gameday['GamedayId'] = 900
    seen = []
    def fetch(player_id):
        seen.append(player_id)
        return payloads[player_id]
    monkeypatch.setattr(maintenance, 'fetch_player_stats', fetch)
    snapshots = maintenance.capture_snapshots([scenario.round], json.loads(scenario.market.read_text()),
                                               maintenance.load_canonical_scores(scenario.canonical))
    assert set(seen) == set(payloads)
    retained = snapshots[scenario.round]['payloads']
    assert any(p['payload']['Value']['GamedayWiseStats'] for p in retained)
    assert all(g['GamedayId'] == 900 for p in retained for g in p['payload']['Value']['GamedayWiseStats'])


def test_candidate_is_versioned_reproducible_and_repeated_generation_is_idempotent(monkeypatch, scenario):
    before, active = hashes(scenario.canonical_dir), maintenance.sha(scenario.active)
    bundle = candidate(scenario)
    original_bundle = hashes(bundle)
    repeated = maintenance.create_update(**scenario.options)
    assert repeated['status'] == 'candidate_exists'
    assert Path(repeated['bundle']) == bundle
    assert hashes(bundle) == original_bundle
    assert hashes(scenario.canonical_dir) == before
    assert maintenance.sha(scenario.active) == active
    receipt = pass_checks(monkeypatch, bundle)
    assert receipt['passed']
    manifest = maintenance.verify_bundle(bundle)
    assert manifest['after_samples']['driver'] > manifest['before_samples']['driver']
    assert (bundle / 'UPDATE_REPORT.md').exists()
    assert (bundle / 'parameter_comparison.csv').exists()
    proposed = maintenance.validate_pair(bundle / 'dataset' / scenario.canonical.name)
    old = maintenance.load_canonical_scores(scenario.canonical)
    maintenance.equal_rows(old, proposed[~(proposed.season.eq(maintenance.SEASON) & proposed['round'].eq(scenario.round))])


def test_explicit_checked_activation_and_following_noop(monkeypatch, scenario):
    bundle = candidate(scenario)
    with pytest.raises(ValueError, match='Run --check'):
        maintenance.activate_bundle(bundle, accept_anomalies=True)
    pass_checks(monkeypatch, bundle)
    with pytest.raises(ValueError, match='Review UPDATE_REPORT'):
        maintenance.activate_bundle(bundle)
    result = maintenance.activate_bundle(bundle, accept_anomalies=True)
    active = maintenance.load_sprint_production_calibration(scenario.active)
    assert active.model_version == result['activated_version']
    assert (scenario.active.parent / 'archive').is_dir()
    proposed = maintenance.validate_pair(scenario.canonical)
    assert len(proposed[proposed.season.eq(maintenance.SEASON) & proposed['round'].eq(scenario.round)]) == 33
    assert maintenance.create_update(**scenario.options)['status'] == 'up_to_date'


@pytest.mark.parametrize('damage', ['duplicate_asset', 'duplicate_session', 'wrong_name', 'wrong_season', 'partial'])
def test_bad_snapshot_is_rejected_before_fitting(monkeypatch, scenario, damage):
    raw = deepcopy(scenario.snapshot)
    if damage == 'duplicate_asset':
        raw['payloads'].append(raw['payloads'][0])
    elif damage == 'partial':
        raw['payloads'] = raw['payloads'][:1]
    else:
        sessions = raw['payloads'][0]['payload']['Value']['MatchWiseStats'][0]['RaceDayWise']
        if damage == 'duplicate_session':
            sessions.append(sessions[0])
        elif damage == 'wrong_name':
            sessions[0]['MeetingName'] = 'Wrong Grand Prix'
        else:
            sessions[0]['Season'] = maintenance.SEASON - 1
    maintenance.dump(scenario.source, raw)
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid source must not reach fitter')
    monkeypatch.setattr(maintenance, 'generate_candidate', forbidden)
    with pytest.raises((ValueError, AssertionError)):
        maintenance.create_update(**scenario.options)
    assert not scenario.output.exists()


def test_parquet_disagreement_and_duplicate_canonical_keys_rejected(scenario):
    parquet = scenario.canonical.with_suffix('.parquet')
    frame = pd.read_parquet(parquet)
    frame.loc[0, 'fantasy_points_total'] += 1
    frame.to_parquet(parquet, index=False)
    with pytest.raises(AssertionError):
        maintenance.create_update(**scenario.options, audit=True)


def test_append_replay_is_exact_and_conflicting_existing_weekend_is_rejected(scenario):
    data = maintenance.load_canonical_scores(scenario.canonical)
    last = data[data.season.eq(maintenance.SEASON) & data['round'].eq(scenario.round - 1)]
    maintenance.equal_rows(maintenance.append_weekend(data, last), data)
    changed = last.copy()
    changed.iloc[0, changed.columns.get_loc('fantasy_points_total')] += 1
    with pytest.raises(AssertionError):
        maintenance.append_weekend(data, changed)


@pytest.mark.parametrize('stale', ['active', 'canonical', 'candidate', 'code', 'failed_tests'])
def test_stale_or_failed_candidate_cannot_activate(monkeypatch, scenario, stale):
    bundle = candidate(scenario)
    pass_checks(monkeypatch, bundle)
    if stale == 'active':
        scenario.active.write_text(scenario.active.read_text() + '\n')
    elif stale == 'canonical':
        scenario.canonical.write_text(scenario.canonical.read_text() + '\n')
    elif stale == 'candidate':
        path = bundle / 'fit/runtime_calibration_candidate.json'
        path.write_text(path.read_text() + '\n')
    elif stale == 'code':
        monkeypatch.setattr(maintenance, 'code_hashes', lambda: {'changed': 'code'})
    else:
        receipt = json.loads((bundle / 'checks.json').read_text())
        receipt['passed'] = False
        maintenance.dump(bundle / 'checks.json', receipt)
    before, active = hashes(scenario.canonical_dir), maintenance.sha(scenario.active)
    with pytest.raises(ValueError):
        maintenance.activate_bundle(bundle, accept_anomalies=True)
    assert hashes(scenario.canonical_dir) == before
    assert maintenance.sha(scenario.active) == active


def test_fit_failure_leaves_no_candidate_or_production_changes(monkeypatch, scenario):
    before = hashes(scenario.canonical_dir)
    def fail(*args, **kwargs):
        raise ValueError('deliberate fit failure')
    monkeypatch.setattr(maintenance, 'generate_candidate', fail)
    with pytest.raises(ValueError, match='deliberate'):
        maintenance.create_update(**scenario.options)
    assert not scenario.output.exists()
    assert hashes(scenario.canonical_dir) == before


def test_promotion_failure_rolls_back_dataset_and_active(monkeypatch, scenario):
    bundle = candidate(scenario)
    pass_checks(monkeypatch, bundle)
    before, active = hashes(scenario.canonical_dir), maintenance.sha(scenario.active)
    def fail(*args, **kwargs):
        raise ValueError('deliberate promotion failure')
    monkeypatch.setattr(maintenance, 'promote_candidate', fail)
    with pytest.raises(ValueError, match='deliberate'):
        maintenance.activate_bundle(bundle, accept_anomalies=True)
    assert hashes(scenario.canonical_dir) == before
    assert maintenance.sha(scenario.active) == active
    assert not (bundle / 'activation_pending.json').exists()


def test_interrupted_activation_is_detected(scenario):
    bundle = candidate(scenario)
    maintenance.dump(bundle / 'activation_pending.json', {})
    with pytest.raises(ValueError, match='Interrupted activation'):
        maintenance.activate_bundle(bundle, accept_anomalies=True)


def test_already_ingested_sprint_still_needs_fit_without_reimport(scenario):
    first = candidate(scenario)
    for source in (first / 'dataset').iterdir():
        shutil.copy2(source, scenario.canonical_dir / source.name)
    scenario.source.unlink()
    result = maintenance.create_update(**{**scenario.options, 'output': scenario.output / 'second'})
    assert result['rounds_to_import'] == []
    manifest = maintenance.verify_bundle(Path(result['bundle']))
    assert manifest['plan']['sprint_rounds_added_to_fit'] == [scenario.round]
    assert manifest['before_samples']['driver'] < manifest['after_samples']['driver']


def test_failed_check_invalidates_receipt(monkeypatch, scenario):
    bundle = candidate(scenario)
    pass_checks(monkeypatch, bundle)
    monkeypatch.setattr(maintenance.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stdout='deliberate test failure'))
    with pytest.raises(ValueError, match='Targeted tests failed'):
        maintenance.check_bundle(bundle)
    with pytest.raises(ValueError, match='Run --check'):
        maintenance.activate_bundle(bundle, accept_anomalies=True)


@pytest.mark.parametrize('is_sprint', [False, True])
def test_current_runtime_uses_any_active_version_exactly_once(is_sprint):
    from f1fantasy.sprint_shadow import (
        calculate_sprint_production_adjustment, apply_sprint_production_adjustment,
        driver_personal_weight,
    )
    calibration = maintenance.load_sprint_production_calibration()
    personal = calibration.driver_personal_history[0]
    schedule = pd.DataFrame([
        {'season': maintenance.SEASON, 'round': 1, 'date': '2026-07-01', 'raceName': 'Past'},
        {'season': maintenance.SEASON, 'round': 2, 'date': '2026-08-20', 'raceName': 'Next',
         'sprint_date': '2026-08-19' if is_sprint else None, 'sprint_time': '10:00:00Z'},
    ])
    history = pd.DataFrame([
        {'season': maintenance.SEASON, 'round': 1, 'entity_type': 'driver',
         'canonical_entity_id': personal['entity_id'], 'source_entity_id': '900',
         'name': personal['name'], 'fantasy_points_total': 20., 'sprint_points': None, 'sprint_qualifying_points': None},
        {'season': maintenance.SEASON, 'round': 1, 'entity_type': 'constructor',
         'canonical_entity_id': 'ferrari', 'source_entity_id': '901', 'name': 'Ferrari',
         'fantasy_points_total': 40., 'sprint_points': None, 'sprint_qualifying_points': None},
    ])
    driver = pd.DataFrame([{'id': '900', 'name': personal['name'], 'price': 10., 'next_race_expected_points': 100.}])
    constructor = pd.DataFrame([{'id': '901', 'name': 'Ferrari', 'price': 20., 'next_race_expected_points': 400.}])
    result = calculate_sprint_production_adjustment(
        history, schedule, driver, constructor, [(maintenance.SEASON, 1)], 1.,
        (maintenance.SEASON, 2), production_history_mode='all_supported',
    )
    weight = driver_personal_weight(personal['observation_count'], calibration)
    group = calibration.driver_group_intercept + calibration.driver_group_slope * (
        (20 - calibration.calibration_form_mean) / calibration.calibration_form_sd)
    bonus = weight * (personal['personal_mean_bonus'] or 0.) + (1 - weight) * group
    for assets, adjustment, expected_bonus in (
        (driver, result.drivers, bonus),
        (constructor, result.constructors, calibration.constructor_intercept + calibration.constructor_slope),
    ):
        final = apply_sprint_production_adjustment(assets, adjustment)
        assert final.iloc[0].next_race_expected_points == pytest.approx(
            assets.iloc[0].next_race_expected_points + (expected_bonus if is_sprint else 0.))
        assert final.iloc[0].sprint_calibration_version == calibration.model_version
        with pytest.raises(ValueError, match='already been applied'):
            apply_sprint_production_adjustment(final, adjustment)


def test_missing_intervening_normal_weekend_is_imported_for_form(scenario):
    normal_round, sprint_round = scenario.round, scenario.round + 1
    schedule = pd.read_csv(scenario.schedule)
    date = pd.Timestamp(schedule.loc[schedule['round'].eq(normal_round), 'date'].iloc[0])
    schedule.loc[schedule['round'].eq(normal_round), ['sprint_date', 'sprint_time', 'sprint_qualifying_date', 'sprint_qualifying_time']] = None
    schedule.loc[schedule['round'].eq(sprint_round), ['raceName', 'date', 'sprint_date', 'sprint_time']] = [
        'Later Maintenance Grand Prix', (date + pd.Timedelta(days=1)).date().isoformat(), date.date().isoformat(), '10:00:00Z']
    schedule.to_csv(scenario.schedule, index=False)
    later = deepcopy(scenario.snapshot)
    later['round'] = sprint_round
    for item in later['payloads']:
        for meeting in item['payload']['Value']['MatchWiseStats']:
            for session in meeting['RaceDayWise']:
                session['MeetingNumber'] = sprint_round
                session['MeetingName'] = 'Later Maintenance Grand Prix'
    maintenance.dump(scenario.sources / 'official_later.json', later)
    normal = deepcopy(scenario.snapshot)
    for item in normal['payloads']:
        value = item['payload']['Value']
        removed = 0.
        for meeting in value['MatchWiseStats']:
            sessions = meeting['RaceDayWise']
            for session in sessions:
                if 'sprint' in session['SessionType'].lower():
                    removed += sum(float(stat['Value']) for stat in session['StatsWise'] if stat['Event'].lower() == 'total')
            meeting['RaceDayWise'] = [s for s in sessions if 'sprint' not in s['SessionType'].lower()]
        for gameday in value['GamedayWiseStats']:
            for stat in gameday['StatsWise']:
                if stat['Event'].lower() == 'total' and stat['Value'] is not None:
                    stat['Value'] = float(stat['Value']) - removed
    maintenance.dump(scenario.source, normal)
    result = maintenance.create_update(**scenario.options)
    assert result['rounds_to_import'] == [normal_round, sprint_round]
    assert result['sprint_rounds_added_to_fit'] == [sprint_round]
    for row in result['available_observations']:
        if row['round'] == normal_round:
            assert row['weekend_format'] == 'normal'
            assert row['missing_sprint_components'] is None
    bundle = Path(result['bundle'])
    proposed = maintenance.validate_pair(bundle / 'dataset' / scenario.canonical.name)
    assert {normal_round, sprint_round}.issubset(set(proposed['round']))
