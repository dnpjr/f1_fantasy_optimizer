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

---

## Project Structure

```
f1fantasy/
    ergast.py          # Historical data ingestion
    fantasy_api.py     # Current roster + price feeds
    model.py           # Scoring + expected value model
    optimize.py        # MILP optimisation logic
    transfers.py       # Transfer-aware logic
    recommend.py       # CLI entry point
    debug_checks.py    # Sanity checks and validation tools
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
- Chip scenarios evaluated separately

### Transfer-aware recommendations (optional)

Create:

`f1fantasy/data/current_team.json`

Example:

```json
{
  "drivers": [131, 117, 1982, 18, 11031],
  "constructors": [27, 28],
  "free_transfers": 2
}
```

Then run:

```bash
python -m f1fantasy.recommend
```

### Debug / validation

```bash
python -m f1fantasy.debug_checks
```

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
- Race fastest lap bonus (when available in data)

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
- No probabilistic simulation (deterministic expected value model)
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
