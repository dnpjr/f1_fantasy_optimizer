# F1 Fantasy Optimizer

A Python project for modelling F1 Fantasy expected points and optimising team selection under official squad constraints.
It combines live fantasy market data, historical race results, a transparent scoring model, and mixed-integer optimisation to rank teams, simulate chips, and evaluate transfer moves.

> This project is not affiliated with Formula 1, FIA, or the official F1 Fantasy game.

---

## What it does

- downloads the latest available official F1 Fantasy market snapshot automatically
- pulls historical race, qualifying, sprint, and schedule data from an Ergast-compatible source
- computes driver and constructor fantasy scores from raw results
- estimates expected value over an upcoming race horizon
- optimises the best legal squad under a configurable budget cap
- evaluates chip scenarios separately
- suggests transfer moves from a saved current squad, including realistic transfer budget handling

The model is designed to stay simple, inspectable, and easy to tweak rather than chase noisy marginal features.

---

## Current features

- **Automatic fantasy market feed detection**
  - the code probes the latest available `drivers/{n}_en.json` fantasy feed automatically
  - no manual round-number update is needed after price changes
- **Live driver and constructor prices**
- **Historical results ingestion**
  - race
  - qualifying
  - sprint
  - schedule
- **Rule-based fantasy scoring engine**
  - drivers and constructors handled separately
  - qualifying, race, sprint, DNFs, DSQs, and constructor aggregation included
- **Expected-value model with transparent weighting**
- **MILP optimisation via PuLP**
- **Chip scenarios**
  - 2x Boost
  - 3x Boost
  - No Negative
  - Limitless
- **Transfer recommendations**
  - uses current squad IDs from `data/current_team.json`
  - prints current team and current EV first
  - prints recommendations with both IDs and names
  - outputs an updated suggested `current_team.json`
  - updates bank correctly after suggested moves
- **Transfer budget handling based on current team value + bank**
- **Roster ID helper utility**
- **Debug / validation utilities**

---

## Project structure

```text
f1fantasy/
    __init__.py
    debug_checks.py
    ergast.py
    fantasy_api.py
    fantasy_prices.py
    model.py
    optimize.py
    print_roster_ids.py
    recommend.py
    transfers.py
    update_cache.py

data/
    cache/

tests/
    test_smoke.py
```

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/dnpjr/f1_fantasy_optimizer.git
cd f1_fantasy_optimizer

python -m venv .venv
# Windows
.venv\Scriptsctivate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Basic usage

Run the main optimiser:

```bash
python -m f1fantasy.recommend
```

This prints the top teams under the configured budget cap and evaluates chip scenarios separately.

You can also use the installed console entry points:

```bash
f1fantasy-recommend
f1fantasy-debug
```

---

## Transfer recommendations

To generate transfer suggestions, create:

```text
data/current_team.json
```

Example:

```json
{
  "drivers": [121, 11031, 114, 129, 131],
  "constructors": [27, 29],
  "free_transfers": 2,
  "bank": 0.5
}
```

Fields:

- `drivers`: list of current driver `playerId`s
- `constructors`: list of current constructor `teamId`s
- `free_transfers`: free transfers still available
- `bank`: money left in the bank

To print the currently available IDs:

```bash
python -m f1fantasy.print_roster_ids
```

Transfer recommendations use:

```text
transfer budget = current market value of held squad + bank
```

This means the transfer solver reflects price changes in your existing squad, rather than assuming a flat `100.0` budget.

---

## Budget cap: where to change it

For fresh-team optimisation, the configurable cap lives in:

```text
f1fantasy/recommend.py
```

Look for:

```python
TEAM_BUDGET_CAP = 100.0
```

You can change this to values such as:

```python
TEAM_BUDGET_CAP = 101.8
```

if you want to test the best full rebuild team at a different effective budget.

This is separate from transfer mode, which calculates its own budget from:

```text
current team value + bank
```

---

## Weighting / scaling model

The expected-value logic is in:

```text
f1fantasy/model.py
```

The current setup separates:

- **current-season form**
- **historical track-aware signal**

### Current-season share

Controlled by:

```python
def _current_season_share(...)
```

Default behaviour:

- 0 completed races -> `0.00` current / `1.00` historical
- 1 completed race -> `0.50` current / `0.50` historical
- scales linearly up to
- 10+ completed races -> `0.75` current / `0.25` historical

### Within-season recency weighting

Controlled by:

```python
def _current_round_weight(...)
```

Default behaviour:

- latest completed race = `1.00`
- previous race = `0.95`
- then `0.95^2`, `0.95^3`, ...

### Historical season decay

Controlled by:

```python
def _historical_season_weight_hist_only(...)
```

Default behaviour:

- previous season = `0.75^0 = 1.0`
- two seasons back = `0.75^1`
- three seasons back = `0.75^2`
- and so on

### Important modelling choice

The current-season block uses **only completed races from the current season**.
Historical data is blended in only on the historical side of the forecast, so prior seasons do not leak into the current-season form signal.

---

## Scoring model overview

### Drivers

The scoring engine includes:

- qualifying position points
- race finishing points
- sprint finishing points
- positions gained / lost proxy
- overtake proxy
- DNF / NC / DSQ handling
- fastest lap where available

### Constructors

The scoring engine includes:

- both drivers' qualifying totals
- both drivers' race totals
- both drivers' sprint totals
- qualifying stage bonuses
- constructor-level DNF / DSQ handling

### Not currently modelled / simplified

- true official overtake counts
- sprint fastest lap
- Driver of the Day
- pit stop scoring
- world-record pit stop bonus
- stochastic simulation / distributions

The project intentionally stays fairly simple and rule-consistent rather than trying to overfit noisy inputs.

---

## Debugging and validation

Useful helper scripts:

```bash
python -m f1fantasy.debug_checks
python -m f1fantasy.print_roster_ids
```

These are useful for checking:

- raw data ingestion
- constructor aggregation
- DNF handling
- current vs historical blend behaviour
- current fantasy roster IDs and prices

---

## Limitations

- depends on public / unofficial data feeds that may change schema or availability
- some fantasy sub-components are approximated or omitted
- expected value is deterministic rather than fully probabilistic
- no historical backtest framework is included yet

---

## License

MIT License — see `LICENSE`.
