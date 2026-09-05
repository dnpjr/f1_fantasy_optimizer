#!/usr/bin/env python3
"""Audit, stage, check and explicitly activate a reproducible Sprint-model update."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from f1fantasy.historical_scores import (
    CANONICAL_KEY, DEFAULT_CANONICAL_DATASET_PATH, load_canonical_scores,
    validate_canonical_scores,
)
from f1fantasy.model import normalise_sprint_baseline_inputs
from f1fantasy.player_stats import fetch_player_stats
from f1fantasy.sprint_shadow import (
    DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH, load_sprint_production_calibration,
    parse_sprint_production_calibration,
)
from scripts.analyse_sprint_multiplier import load_schedule_metadata, _normalise_event_name
from scripts.build_historical_fantasy_scores import _snapshot_official, write_canonical_outputs
from scripts.build_2026_sprint_final_candidate import prepare_candidate_data
from scripts.calibrate_asset_sprint_adjustments import SEASON
from scripts.recalibrate_sprint_ev import (
    DEFAULT_MARKET, DEFAULT_PREVIOUS_REPORT, DEFAULT_SCHEDULE, _flatten,
    generate_candidate, promote_candidate,
)

DEFAULT_SOURCES = DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH.parent / "sources"
DEFAULT_OUTPUT = ROOT / "reports/sprint_updates"
TESTS = (
    "tests/test_update_sprint_model.py", "tests/test_sprint_ev_components.py",
    "tests/test_sprint_ev_shadow.py", "tests/test_sprint_shadow.py",
    "tests/test_race_selection.py",
)


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def equal_rows(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        left.sort_values(CANONICAL_KEY).reset_index(drop=True).fillna(""),
        right.sort_values(CANONICAL_KEY).reset_index(drop=True).fillna(""),
        check_dtype=False, check_exact=True,
    )


def validate_pair(path: Path) -> pd.DataFrame:
    data = load_canonical_scores(path)
    if data.empty:
        raise ValueError(f"Missing canonical history: {path}")
    parquet = path.with_suffix(".parquet")
    if parquet.exists():
        equal_rows(data, pd.read_parquet(parquet))
    return data


def code_hashes() -> dict[str, str]:
    # Bind validation to the actual dirty working tree, not merely a git commit.
    paths = list((ROOT / "f1fantasy").glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    paths += [ROOT / name for name in TESTS]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def capture_snapshots(rounds: list[int], market: dict, canonical: pd.DataFrame) -> dict[int, dict]:
    """Fetch through the repository client, retaining inactive and replaced asset IDs."""
    players = {int(p["playerId"]): dict(p) for p in market.get("player_assets", market["players"])}
    teams = {int(p["teamId"]): dict(p) for p in market.get("constructor_assets", market["teams"])}
    for row in canonical[canonical.season.eq(SEASON)].sort_values("round").to_dict("records"):
        target, id_field = (players, "playerId") if row["entity_type"] == "driver" else (teams, "teamId")
        target.setdefault(int(row["source_entity_id"]), {
            id_field: int(row["source_entity_id"]), "name": row["name"],
            "tla": row["abbreviation"], "team": str(row.get("constructor_name") or ""),
        })
    payloads = []
    for player_id in sorted(set(players) | set(teams)):
        payload = fetch_player_stats(player_id)
        if int(payload["Value"]["PlayerId"]) != player_id:
            raise ValueError(f"Playerstats identity mismatch for {player_id}")
        payloads.append({"player_id": player_id, "payload": payload,
                         "source_url": f"https://fantasy.formula1.com/feeds/popup/playerstats_{player_id}.json"})
    expected = canonical[canonical.season.eq(SEASON)].groupby(
        ["round", "entity_type"]
    ).size().groupby("entity_type").max().astype(int).to_dict()
    snapshots = {}
    for round_no in rounds:
        retained = deepcopy(payloads)
        for item in retained:
            value = item["payload"]["Value"]
            # Use MeetingNumber, not GamedayId, to select the canonical meeting.
            matches = [m for m in value.get("MatchWiseStats", []) if any(
                int(s.get("Season", 0)) == SEASON and int(s.get("MeetingNumber", 0)) == round_no
                for s in m.get("RaceDayWise", [])
            )]
            gamedays = {m["GamedayId"] for m in matches}
            value["MatchWiseStats"] = matches
            value["GamedayWiseStats"] = [g for g in value.get("GamedayWiseStats", []) if g["GamedayId"] in gamedays]
        snapshots[round_no] = {
            "season": SEASON, "round": round_no,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "expected_played_counts": expected,
            "players": list(players.values()), "teams": list(teams.values()), "payloads": retained,
        }
    return snapshots


def snapshot_rows(snapshot: dict, schedule: pd.DataFrame, directory: Path) -> pd.DataFrame:
    if int(snapshot["season"]) != SEASON:
        raise ValueError("Snapshot season does not match this scoring model.")
    event = schedule[schedule['round'].eq(int(snapshot['round']))]
    if len(event) != 1:
        raise ValueError('Unknown snapshot season/round.')
    seen = set()
    for item in snapshot["payloads"]:
        identity = int(item["player_id"])
        if identity in seen or int(item["payload"]["Value"]["PlayerId"]) != identity:
            raise ValueError("Duplicate or mismatched raw asset identity.")
        seen.add(identity)
        value = item['payload']['Value']
        for collection in ('GamedayWiseStats', 'MatchWiseStats'):
            ids = [g['GamedayId'] for g in value.get(collection, [])]
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate raw gameday would double count sessions.')
        for meeting in value.get('MatchWiseStats', []):
            sessions = meeting.get('RaceDayWise', [])
            ids = [session['RaceDayId'] for session in sessions]
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate raw session would double count points.')
            for session in sessions:
                if (int(session['Season']), int(session['MeetingNumber'])) != (SEASON, int(snapshot['round'])) or _normalise_event_name(session['MeetingName']) != _normalise_event_name(event.iloc[0].raceName):
                    raise ValueError('Raw session season/round/name conflicts with schedule.')
    path = directory / f"official_{SEASON}_round_{int(snapshot['round'])}.json"
    dump(path, snapshot)
    # Existing parser accepts one labelled Sprint component; wholly missing stays null.
    return _snapshot_official(path, {SEASON: schedule})[0]


def append_weekend(existing: pd.DataFrame, added: pd.DataFrame) -> pd.DataFrame:
    keys = added[["season", "round"]].drop_duplicates()
    if len(keys) != 1:
        raise ValueError("One snapshot must describe exactly one canonical weekend.")
    season, round_no = keys.iloc[0]
    prior = existing[existing.season.eq(season) & existing['round'].eq(round_no)]
    if not prior.empty:
        equal_rows(prior, added)  # Never silently revise even a partial existing event.
        return existing
    result = pd.concat([existing, added], ignore_index=True).sort_values(CANONICAL_KEY).reset_index(drop=True)
    validate_canonical_scores(result)
    return result


def validate_inputs(canonical: Path, schedule: Path, market: Path) -> dict:
    data = validate_pair(canonical)
    prepared = prepare_candidate_data(canonical, schedule, market, maximum_round=None)
    observations, history = prepared["observations"], prepared["history"]
    if observations.duplicated(["season", "round", "entity_type", "entity_id"]).any():
        raise ValueError("Duplicate fitting observations.")
    keys = {(int(r.season), int(r.round)) for r in prepared['events'].itertuples()
            if r.weekend_format == 'sprint'}
    # Check the existing production baseline helper against the fitter's definition.
    for kind in ('driver', 'constructor'):
        source = history[history.entity_type.eq(kind)].copy()
        source['weekend_points'] = source['fantasy_points_total']
        normalized = normalise_sprint_baseline_inputs(source, keys)
        expected = source[source.normalised_score.notna()]
        np.testing.assert_allclose(normalized.weekend_points, expected.normalised_score, rtol=0, atol=1e-9)
    counts = observations.groupby(['round', 'entity_type']).agg(
        grid_rows=('entity_id', 'size'), available=('observation_valid', 'sum')
    ).reset_index()
    counts['missing'] = counts['grid_rows'] - counts['available']
    counts['round'] = counts['round'].astype(int)
    return {
        'sprint_rounds': sorted(r for _, r in keys),
        'completed_rounds': sorted(int(r) for r in prepared['events']['round']),
        'samples': observations[observations.observation_valid].groupby('entity_type').size().astype(int).to_dict(),
        'coverage': counts.to_dict('records'), 'canonical_rows': len(data),
    }


def parameter_comparison(before: dict, after: dict) -> tuple[list[dict], list[str]]:
    old, new = _flatten(before), _flatten(after)
    rows, anomalies = [], []
    for key in sorted(set(old) | set(new)):
        if not key.startswith(('driver.', 'constructor.')):
            continue
        a, b = old.get(key), new.get(key)
        delta = b - a if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        rows.append({'parameter': key, 'old': a, 'new': b, 'change': delta})
        if delta is not None and 'personal_history' not in key and abs(delta) > .25 * max(abs(a), 1):
            anomalies.append(f'{key}: change exceeds 25% of max(|old|, 1): {a:.6g} -> {b:.6g}')
    return rows, anomalies


def plan_update(canonical: Path, active: Path, schedule: Path, market: Path,
                sources: Path, fetch: bool, work: Path) -> tuple[dict, pd.DataFrame, list[Path]]:
    if schedule.name != DEFAULT_SCHEDULE.name:
        raise ValueError(f'This fitter requires a {DEFAULT_SCHEDULE.name} file; season migration is manual.')
    active_raw = json.loads(active.read_text())
    calibration = load_sprint_production_calibration(active)
    if calibration.calibration_season != SEASON:
        raise ValueError(f'This fitter supports scoring season {SEASON} only.')
    data = validate_pair(canonical)
    metadata = load_schedule_metadata(schedule.parent, seasons=(SEASON,))
    # The existing fitter also conservatively requires a past UTC calendar date.
    past = metadata[pd.to_datetime(metadata.date, utc=True).lt(pd.Timestamp.now(tz='UTC').normalize())]
    past_sprints = set(past.loc[past.weekend_format.eq('sprint'), 'round'].astype(int))
    existing = set(data.loc[data.season.eq(SEASON), 'round'].astype(int))
    included = sorted(past_sprints & existing)
    fitted = active_raw.get('sprint_rounds')
    if not isinstance(fitted, list) or not set(fitted).issubset(included):
        raise ValueError('Active calibration Sprint provenance is missing or inconsistent with canonical data.')
    pending = sorted(past_sprints - existing)
    needed = sorted(set(range(1, max(past_sprints, default=0) + 1)) - existing) if pending else []
    snapshots = {}
    for path in sorted(sources.glob('official_*.json')):
        raw = json.loads(path.read_text())
        if int(raw['season']) == SEASON and int(raw['round']) in needed:
            r = int(raw['round'])
            if r in snapshots and snapshots[r] != raw:
                raise ValueError(f'Conflicting local snapshots for round {r}.')
            snapshots[r] = raw
    if fetch and needed:
        fetched = capture_snapshots([r for r in needed if r not in snapshots], json.loads(market.read_text()), data) if set(needed) - snapshots.keys() else {}
        snapshots.update(fetched)
    plan = {'season': SEASON, 'active_version': calibration.model_version,
            'sprint_catalogue': past.loc[past.weekend_format.eq('sprint'), ['season', 'round', 'raceName', 'date']].to_dict('records'),
            'canonical_sprint_rounds': included, 'active_sprint_rounds': fitted,
            'scheduled_past_sprints_missing': pending, 'rounds_to_import': needed,
            'sources_unavailable': sorted(set(needed) - snapshots.keys())}
    source_dir = work / 'sources'
    source_dir.mkdir()
    available = []
    for r in sorted(snapshots):
        added = snapshot_rows(snapshots[r], metadata, source_dir)
        fmt = metadata.loc[metadata['round'].eq(r), 'weekend_format'].iloc[0]
        for kind, group in added.groupby('entity_type'):
            components = group[['sprint_points', 'sprint_qualifying_points']].notna().any(axis=1)
            available.append({'round': r, 'entity_type': kind, 'weekend_format': fmt, 'played_rows': len(group),
                              'available_sprint_components': int(components.sum()) if fmt == 'sprint' else None,
                              'missing_sprint_components': int((~components).sum()) if fmt == 'sprint' else None})
        data = append_weekend(data, added)
    plan['available_observations'] = available
    if plan['sources_unavailable']:
        plan['status'] = 'awaiting_sources'
        plan['actions'] = ['Refresh local schedule/verified market and supply complete snapshots, or use --fetch. No fit or activation.']
        return plan, data, []
    added_to_fit = sorted(past_sprints - set(fitted))
    plan['sprint_rounds_added_to_fit'] = added_to_fit
    if not added_to_fit:
        plan.update(status='up_to_date', actions=['Nothing new to ingest or fit; no files changed.'])
        return plan, data, []
    proposed = work / canonical.name
    data.to_csv(proposed, index=False)
    audit = validate_inputs(proposed, schedule, market)
    plan.update(status='ready', candidate_data=audit,
                actions=['Stage canonical data and source snapshots', 'Fit candidate and report',
                         'Run --check, review report, then explicitly --activate'])
    return plan, data, sorted(source_dir.glob('*.json'))


def create_update(*, canonical: Path = DEFAULT_CANONICAL_DATASET_PATH,
                  active: Path = DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH,
                  schedule: Path = DEFAULT_SCHEDULE, market: Path = DEFAULT_MARKET,
                  sources: Path = DEFAULT_SOURCES, output: Path = DEFAULT_OUTPUT,
                  previous_report: Path = DEFAULT_PREVIOUS_REPORT,
                  audit: bool = False, fetch: bool = False) -> dict:
    # Everything is prepared outside production; failures cannot damage current data.
    with tempfile.TemporaryDirectory(prefix='sprint-maintenance-') as temp:
        work = Path(temp)
        plan, data, snapshot_paths = plan_update(canonical, active, schedule, market, sources, fetch, work)
        if audit or plan['status'] != 'ready':
            return plan
        bundle = work / 'bundle'
        bundle.mkdir()
        inputs = bundle / 'inputs'
        inputs.mkdir()
        shutil.copy2(schedule, inputs / f'schedule_{SEASON}.csv')
        shutil.copy2(market, inputs / 'market.json')
        shutil.copytree(previous_report, inputs / 'previous_report')
        shutil.copytree(canonical.parent, bundle / 'before_dataset')
        shutil.copy2(active, bundle / 'before_active.json')
        shutil.copytree(work / 'sources', inputs / 'sources')
        destination = canonical.parent.resolve()
        write_canonical_outputs(data, bundle / 'dataset', published_root=destination)
        staged = bundle / 'dataset' / canonical.name
        if canonical.with_suffix('.parquet').exists() and not staged.with_suffix('.parquet').exists():
            raise ValueError('Parquet generation failed; refusing an inconsistent candidate.')
        audit_data = validate_inputs(staged, inputs / schedule.name, inputs / 'market.json')
        result = generate_candidate(bundle / 'fit', active_path=bundle / 'before_active.json',
                                    canonical_path=staged, schedule_path=inputs / schedule.name,
                                    market_path=inputs / 'market.json', previous_report=inputs / 'previous_report')
        before = json.loads(active.read_text())
        candidate = result['candidate']
        comparison, anomalies = parameter_comparison(before, candidate)
        for row in audit_data['coverage']:
            if row['round'] in plan['sprint_rounds_added_to_fit'] and row['missing']:
                anomalies.append(f"R{row['round']} {row['entity_type']}: {row['missing']} missing grid observations (absences or missing components); omitted, not zero.")
        dump(bundle / 'comparison.json', comparison)
        pd.DataFrame(comparison).to_csv(bundle / 'parameter_comparison.csv', index=False)
        before_fit = load_canonical_scores(bundle / 'before_dataset' / canonical.name)
        before_fit = before_fit[~before_fit.season.eq(SEASON) | before_fit['round'].isin(before['completed_rounds'])]
        before_fit_path = work / 'active_fit_history.csv'
        before_fit.to_csv(before_fit_path, index=False)
        before_audit = validate_inputs(before_fit_path, inputs / schedule.name, inputs / 'market.json')
        manifest = {'schema': 1, 'plan': plan, 'candidate_version': candidate['model_version'],
                    'canonical_destination': str(canonical.resolve()), 'active_destination': str(active.resolve()),
                    'before_active_sha256': sha(active), 'before_dataset_sha256': {p.name: sha(p) for p in canonical.parent.iterdir() if p.is_file()},
                    'before_samples': before_audit['samples'], 'after_samples': audit_data['samples'],
                    'anomalies': anomalies, 'code_sha256': code_hashes(),
                    'environment': {'python': sys.version, 'pandas': pd.__version__, 'numpy': np.__version__}}
        report = [f"# Sprint maintenance candidate {candidate['model_version']}", '',
                  f"Active baseline: {before['model_version']}. Production data and calibration are unchanged.", '',
                  f"Sprint rounds before: {before['sprint_rounds']}; after: {candidate['sprint_rounds']}.",
                  f"Imported rounds (including intervening normal-form history): {plan['rounds_to_import']}.",
                  f"Valid samples before (active fit rounds): {before_audit['samples']}; after: {audit_data['samples']}.", '',
                  'Existing official parser, normal-equivalent definition, missing-value rules and fitter are reused without statistical changes.',
                  'Inactive drivers remain in historical form; replacement IDs are resolved by the existing canonical normaliser.',
                  'See parameter_comparison.csv for every old/new/change, fit/sprint_observations.csv for inclusions/exclusions, and inputs/sources for raw provenance.',
                  'Constructor differences can also reflect the frozen current-market price ranks. Copied research validation metrics are historical, not validation of this new fit.', '',
                  '## Parameters', '', '| Parameter | Old | New | Change |', '|---|---:|---:|---:|',
                  *[f"| {r['parameter']} | {r['old']:.9g} | {r['new']:.9g} | {r['change']:+.9g} |"
                    for r in comparison if r['change'] is not None and 'personal_history' not in r['parameter']], '',
                  '## Review flags', '', *(anomalies or ['No material parameter-change flags.']), '',
                  '## Exact next steps', '', 'Run `python scripts/update_sprint_model.py --check BUNDLE`.',
                  'Inspect this report and parameter_comparison.csv. Then run `python scripts/update_sprint_model.py --activate BUNDLE`.',
                  'Activation requires a passing check for these exact files and code, unchanged production inputs, and explicit acknowledgement of flags.',
                  'Use `--accept-anomalies` with --activate only after reviewing any flags. Stop the app during activation; multiple data files cannot be replaced as one filesystem operation.',
                  'Commit this complete bundle, the installed canonical dataset and runtime calibration/archive together.', '',
                  '## Reproduction', '',
                  'The bundle freezes the prior active calibration, prior dataset, proposed dataset, schedule, market, raw additions, prior research comparison inputs and code hashes.',
                  'The --check command refits these frozen inputs and compares model parameters and histories exactly, then runs the maintenance regression suite.',
                  'No future-season scoring migration or automatic public-feed repair is attempted.', '']
        (bundle / 'UPDATE_REPORT.md').write_text('\n'.join(report))
        (bundle / 'fit/PROMOTION_REVIEW.md').write_text('Use update_sprint_model.py --check BUNDLE and --activate BUNDLE. Do not promote this calibration independently of its staged dataset. See ../UPDATE_REPORT.md.\n')
        # The raw source list supplements the existing dataset writer's provenance.
        provenance_path = bundle / 'dataset/provenance_manifest.json'
        provenance = json.loads(provenance_path.read_text())
        provenance['maintenance_source_snapshots'] = {p.name: sha(p) for p in snapshot_paths}
        dump(provenance_path, provenance)
        manifest['files_sha256'] = {str(p.relative_to(bundle)): sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
        dump(bundle / 'manifest.json', manifest)
        identity = {key: manifest[key] for key in ('before_active_sha256', 'before_dataset_sha256', 'code_sha256')}
        identity['inputs'] = candidate['input_sha256']
        identity['previous_report'] = {str(p.relative_to(inputs)): sha(p) for p in sorted((inputs / 'previous_report').rglob('*')) if p.is_file()}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        target = output.resolve() / f"{candidate['model_version']}_{fingerprint}"
        if target.exists():
            saved = verify_bundle(target)
            if saved['before_active_sha256'] != sha(active) or saved['code_sha256'] != code_hashes():
                raise ValueError(f'Existing bundle is stale; retain it for audit and choose a new --output: {target}')
            plan['bundle'] = str(target)
            plan['status'] = 'candidate_exists'
            return plan
        output.mkdir(parents=True, exist_ok=True)
        # Publish a complete bundle via a sibling rename, never a half-written fit.
        staging = Path(tempfile.mkdtemp(prefix='.candidate-', dir=output))
        try:
            shutil.copytree(bundle, staging, dirs_exist_ok=True)
            os.rename(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        plan.update(bundle=str(target), candidate_version=candidate['model_version'])
        return plan


def verify_bundle(bundle: Path) -> dict:
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['schema'] != 1:
        raise ValueError('Unsupported maintenance bundle schema.')
    for name, expected in manifest['files_sha256'].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha(path) != expected:
            raise ValueError(f'Bundle changed or missing: {name}')
    return manifest


def check_bundle(bundle: Path) -> dict:
    manifest = verify_bundle(bundle)
    if manifest['code_sha256'] != code_hashes():
        raise ValueError('Fitting/production/test code changed since candidate generation; generate a new bundle.')
    canonical_name = Path(manifest['canonical_destination']).name
    validate_inputs(bundle / 'dataset' / canonical_name, bundle / 'inputs' / DEFAULT_SCHEDULE.name, bundle / 'inputs/market.json')
    with tempfile.TemporaryDirectory(prefix='sprint-refit-') as temp:
        rebuilt = generate_candidate(Path(temp), active_path=bundle / 'before_active.json',
                                     canonical_path=bundle / 'dataset' / canonical_name,
                                     schedule_path=bundle / 'inputs' / DEFAULT_SCHEDULE.name,
                                     market_path=bundle / 'inputs/market.json', previous_report=bundle / 'inputs/previous_report')['candidate']
    candidate = json.loads((bundle / 'fit/runtime_calibration_candidate.json').read_text())
    for key in ('driver', 'constructor', 'model_version', 'sprint_rounds', 'completed_rounds', 'input_sha256'):
        if rebuilt[key] != candidate[key]:
            raise ValueError(f'Candidate does not reproduce: {key}')
    # Avoid the local broken readline extension without suppressing any test.
    command = [sys.executable, '-c', "import sys; sys.modules['readline']=None; import pytest; raise SystemExit(pytest.main(sys.argv[1:]))", *TESTS, '-q']
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (bundle / 'TEST_RESULTS.txt').write_text(result.stdout)
    if verify_bundle(bundle) != manifest or code_hashes() != manifest['code_sha256']:
        (bundle / 'checks.json').unlink(missing_ok=True)
        raise ValueError('Bundle or code changed while checking; generate and check a fresh candidate.')
    receipt = {'passed': result.returncode == 0, 'manifest_sha256': sha(bundle / 'manifest.json'),
               'code_sha256': code_hashes(), 'checked_at_utc': datetime.now(UTC).isoformat(), 'command': command}
    dump(bundle / 'checks.json', receipt)
    if result.returncode:
        raise ValueError(f'Targeted tests failed; see {bundle / "TEST_RESULTS.txt"}')
    return receipt


@contextmanager
def activation_lock(active: Path):
    lock = active.parent / '.sprint-maintenance.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(f'Another activation or interrupted process owns {lock}; inspect before removing it.') from None
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink()


def atomic_copy(source: Path, destination: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix='.sprint-', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(source.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def activate_bundle(bundle: Path, *, accept_anomalies: bool = False) -> dict:
    manifest = verify_bundle(bundle)
    candidate_path = bundle / 'fit/runtime_calibration_candidate.json'
    candidate = json.loads(candidate_path.read_text())
    parse_sprint_production_calibration(candidate, allowed_statuses=('candidate',))
    active = Path(manifest['active_destination'])
    canonical = Path(manifest['canonical_destination'])
    with activation_lock(active):
        if (bundle / 'activation_pending.json').exists():
            raise ValueError('Interrupted activation: restore files from before_dataset and before_active.json before retrying; see maintenance documentation.')
        if sha(active) != manifest['before_active_sha256']:
            raise ValueError('Active calibration changed since candidate generation; refusing stale activation.')
        for name, expected in manifest['before_dataset_sha256'].items():
            if sha(canonical.parent / name) != expected:
                raise ValueError(f'Canonical dataset changed since candidate generation: {name}')
        receipt = json.loads((bundle / 'checks.json').read_text()) if (bundle / 'checks.json').exists() else {}
        if not receipt.get('passed') or receipt.get('manifest_sha256') != sha(bundle / 'manifest.json') or receipt.get('code_sha256') != code_hashes() or manifest['code_sha256'] != code_hashes():
            raise ValueError('Run --check successfully for this exact bundle and code before activation.')
        if manifest['anomalies'] and not accept_anomalies:
            raise ValueError('Review UPDATE_REPORT.md flags, then explicitly use --accept-anomalies to activate.')
        validate_pair(bundle / 'dataset' / canonical.name)
        installed = []
        dump(bundle / 'activation_pending.json', {'active': str(active), 'canonical': str(canonical)})
        try:
            for source in sorted((bundle / 'dataset').iterdir()):
                destination = canonical.parent / source.name
                installed.append(destination)
                atomic_copy(source, destination)
            validate_pair(canonical)
            promote_candidate(candidate_path, active_path=active)
        except BaseException:
            for destination in installed:
                backup = bundle / 'before_dataset' / destination.name
                if backup.exists():
                    atomic_copy(backup, destination)
                else:
                    destination.unlink(missing_ok=True)
            atomic_copy(bundle / 'before_active.json', active)
            (bundle / 'activation_pending.json').unlink()
            raise
        receipt = {'activated_version': candidate['model_version'], 'activated_at_utc': datetime.now(UTC).isoformat(),
                   'active_sha256': sha(active), 'canonical_sha256': sha(canonical)}
        dump(bundle / 'activation.json', receipt)
        (bundle / 'activation_pending.json').unlink()
        return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--audit', '--dry-run', action='store_true', help='Validate/discover in temporary storage; do not save a candidate or change production')
    mode.add_argument('--check', type=Path, metavar='BUNDLE', help='Reproduce a candidate and run the targeted maintenance tests')
    mode.add_argument('--activate', type=Path, metavar='BUNDLE', help='Explicitly install a checked, reviewed candidate and staged canonical data')
    parser.add_argument('--accept-anomalies', action='store_true', help='Acknowledge report flags during explicit activation')
    parser.add_argument('--fetch', action='store_true', help='Fetch missing played weekends using the cached verified market and public playerstats client')
    parser.add_argument('--canonical', type=Path, default=DEFAULT_CANONICAL_DATASET_PATH)
    parser.add_argument('--active', type=Path, default=DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH)
    parser.add_argument('--schedule', type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument('--market', type=Path, default=DEFAULT_MARKET)
    parser.add_argument('--sources', type=Path, default=DEFAULT_SOURCES)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        if args.activate:
            result = activate_bundle(args.activate.resolve(), accept_anomalies=args.accept_anomalies)
        elif args.check:
            result = check_bundle(args.check.resolve())
        else:
            if args.accept_anomalies:
                parser.error('--accept-anomalies requires --activate')
            result = create_update(canonical=args.canonical.resolve(), active=args.active.resolve(),
                                   schedule=args.schedule.resolve(), market=args.market.resolve(),
                                   sources=args.sources.resolve(), output=args.output.resolve(),
                                   audit=args.audit, fetch=args.fetch)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2 if result.get('status') == 'awaiting_sources' else 0
    except (ValueError, AssertionError, OSError, KeyError, ImportError) as exc:
        print(f'Sprint maintenance stopped: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
