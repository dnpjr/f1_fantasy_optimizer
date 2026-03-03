# F1 Fantasy Optimizer

A Python project that builds an expected-points model for **Formula 1 Fantasy** and selects optimal teams under budget and chip constraints using mixed-integer linear programming (PuLP).

Built as a side project around F1 and data, this repository focuses on modelling fantasy scoring rules and solving the resulting team selection problem under real game constraints.

> Not affiliated with Formula 1, the FIA, or the official F1 Fantasy game.

---

## What this project does

- Pulls historical race, qualifying, and sprint data from an Ergast-compatible API  
- Pulls current F1 Fantasy roster and prices from public feeds  
- Converts weekend results into fantasy points for drivers and constructors  
- Builds expected scores using recency and circuit-aware weighting  
- Optimizes a full fantasy team under:
  - 5 drivers + 2 constructors  
  - Budget ≤ 100  
  - Boost chip (2x / 3x scenarios)  
  - No Negative and Limitless scenarios  
- Supports transfer-aware recommendations with free-transfer allowances  

---

## How scoring is calculated

### Drivers (per weekend)
- Qualifying position points (10..1 for P1..P10)  
- Race finishing points (25/18/15/12/10/8/6/4/2/1)  
- Sprint finishing points (8..1)  
- DNF/NC/DSQ penalties  
- Positions gained/lost proxy via `grid - finish_position`  
- Overtake proxy (capped): `max(0, grid - finish)`  
- Race fastest lap (+10 when available in data)  

Not currently modeled:
- True overtakes (only proxy used)  
- Sprint fastest lap  
- Driver of the Day  
- Pit stop points (constructors)  

### Constructors (per weekend)
- Sum of both drivers’ qualifying points  
- Sum of both drivers’ race points (excluding DOTD)  
- Sum of both drivers’ sprint points  
- Qualifying stage bonuses (Q2/Q3 reached)  
- DSQ tracked separately from DNF  
- Slightly softened DNF impact (to avoid over-penalising teams)  

---

## Expected score model

Expected scores are computed for a configurable set of upcoming races (default: next 5):

- Horizon weighting: nearer races weighted more heavily  
- Recency weighting:
  - Current season: equal weighting across completed races  
  - Last season: strong weight  
  - Older seasons: exponential decay  
- Driver EV adjusted by current constructor strength to avoid unrealistic mismatches  

---

## Design Focus

The project focuses on:

- Converting structured race data into rule-consistent fantasy scoring  
- Building a recency-weighted expected value model  
- Solving a constrained selection problem via mixed-integer optimisation  
- Comparing chip and budget scenarios under the same objective  

---

## Future improvements

- True overtake data integration  
- Sprint fastest lap support  
- Pit stop scoring for constructors  
- Probabilistic modelling and distribution-based optimisation  
- Historical backtesting across seasons  

---

## License

MIT License (see LICENSE file).

---

## Disclaimer

This project uses public endpoints and may break if upstream providers change formats or block requests.
