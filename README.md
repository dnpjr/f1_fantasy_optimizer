# F1 Fantasy Optimizer

A Python-based optimisation engine for Formula 1 Fantasy.  
The project builds a rule-consistent scoring model from historical race data, estimates expected points using recency-weighted performance, and solves the optimal team selection problem under official game constraints using mixed-integer linear programming (PuLP).

> This project is not affiliated with Formula 1, FIA, or the official F1 Fantasy game.

---

## Features

- Historical race, qualifying, and sprint ingestion (Ergast-compatible API)
- Full driver and constructor fantasy scoring engine
- Recency-weighted expected value modelling
- Circuit-aware horizon weighting (configurable upcoming race window)
- DNF-aware scoring adjustments
- MILP team optimisation (budget, roster constraints)
- Chip scenario modelling:
  - 2x Boost
  - 3x Boost
  - No Negative
  - Limitless
- Transfer-aware team suggestions
- Debug / validation utilities
- Roster ID helper utility

---

## Project Structure

```
f1fantasy/
    ergast.py
    fantasy_api.py
    model.py
    optimize.py
    transfers.py
    recommend.py
    debug_checks.py
    print_roster_ids.py
```

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/dnpjr/f1_fantasy_optimizer.git
cd f1_fantasy_optimizer

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### Run the optimiser

```bash
python -m f1fantasy.recommend
```

This prints the top teams under:

- Budget ≤ 100
- 5 drivers + 2 constructors
- All chip scenarios evaluated separately

---

## Transfer-aware recommendations (optional)

To generate transfer suggestions, the optimiser needs your current Fantasy team (driver IDs + constructor IDs).

### 1) Print live roster IDs

Run:

```bash
python -m f1fantasy.print_roster_ids
```

This prints:

- All drivers with their `playerId`
- All constructors with their `teamId`

Use these IDs in your team configuration file.

### 2) Create `current_team.json`

Create:

```
f1fantasy/data/current_team.json
```

Example:

```json
{
  "drivers": [121, 11031, 114, 129, 131],
  "constructors": [27, 29],
  "free_transfers": 2
}
```

- `drivers` → list of `playerId`
- `constructors` → list of `teamId`
- `free_transfers` → number of free transfers remaining

### 3) Run optimiser again

```bash
python -m f1fantasy.recommend
```

Transfer penalties will be applied beyond `free_transfers`.

---

## Scoring Model Overview

### Drivers

Per weekend scoring includes:

- Qualifying position points (P1–P10)
- Race finishing points (25–1 scale)
- Sprint finishing points (8–1 scale)
- DNF / NC / DSQ penalties
- Positions gained/lost proxy (`grid - finish_position`)
- Capped overtake proxy
- Race fastest lap bonus (when available)

Not currently modelled:

- True overtake counts (proxy used)
- Sprint fastest lap
- Driver of the Day
- Pit stop points (constructors)

### Constructors

- Sum of both drivers’ qualifying points
- Sum of both drivers’ race points (excluding DOTD)
- Sum of both drivers’ sprint points
- Qualifying stage bonuses (Q2/Q3 reach)
- DSQ tracked separately from DNF
- Moderated DNF impact to avoid over-penalisation

---

## Expected Value Methodology

Expected scores are computed across a configurable upcoming race horizon (default: next 5 races):

- Horizon weighting: nearer races weighted more heavily
- Recency weighting:
  - Current season: equal weighting
  - Previous season: high weight
  - Older seasons: exponential decay
- Driver EV scaled by current constructor strength to reduce unrealistic mismatches

The optimisation objective maximises total expected points subject to roster and budget constraints.

---

## Limitations

- Relies on public data feeds (may change without notice)
- Overtake and pit stop metrics approximated or omitted
- Deterministic expected value model (no probabilistic simulation)
- No historical price evolution modelling

---

## Future Improvements

- True overtake integration
- Sprint fastest lap support
- Constructor pit stop scoring
- Probabilistic modelling / variance-aware optimisation
- Historical backtesting across seasons

---

## License

MIT License — see `LICENSE` file.
