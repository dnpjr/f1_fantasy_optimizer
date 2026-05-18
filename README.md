# F1 Fantasy Optimiser

An unofficial F1 Fantasy helper app for comparing teams, tracking likely price changes, and finding transfer options.

The app combines live F1 Fantasy prices, official player-stat feeds, recent race history, and a simple projection model to help answer questions like:

- Which team gives the best expected points under my budget?
- Which drivers or constructors are likely to rise or fall in price?
- What transfers improve my current team?
- How do chips such as 3x, Limitless, and No Negative affect the next race?

> This project is not affiliated with Formula 1, the FIA, Formula One Management, or the official F1 Fantasy game. Outputs are unofficial modelling estimates, not official predictions or guarantees.

## Live app

The app is deployed here:

[https://f1optimise.streamlit.app/](https://f1optimise.streamlit.app/)

---

## Features

### Team optimiser

Builds legal F1 Fantasy teams under a configurable budget and objective.

Supported objectives include:

- points only
- price growth only
- combined points and price growth
- risk-adjusted combined scoring

The optimiser supports locks and exclusions, so you can force assets in or out of team suggestions.

### Current team builder

Enter your current F1 Fantasy squad, bank, budget, and free transfers.

The app then shows:

- team cost
- remaining budget
- expected points
- expected price gain
- projected team value
- fantasy-style cards for each selected driver and constructor

You can also import/export a `current_team.json` file to save your squad setup.

### Price-change budget builder

Shows Canada-style price-change target bands for each driver and constructor.

For each asset, the app estimates how many points are needed in the next race to land in the:

- Terrible
- Poor
- Good
- Great

price-change tiers.

It also computes probabilistic expected price gain using:

- recent true F1 Fantasy points
- expected next-race points
- volatility
- DNF risk
- price floor/ceiling logic

### Transfer recommendations

Compares your current team against possible transfer options.

Recommendations focus on deltas versus your current team:

- change in expected points
- change in expected price gain
- transfer penalty
- remaining budget
- number of transfers used

Transfer suggestions respect locks, exclusions, budget, bank, and free-transfer settings.

### Chip handling

The app supports:

- no chip
- 3x chip
- Limitless
- No Negative

Chip boosts affect expected fantasy points only. They do not affect projected price changes, because price movement is based on the asset’s underlying score, not your chip multiplier.

### Live data

The app uses public F1 Fantasy feeds for:

- live prices
- driver and constructor IDs
- player stats
- recent race-by-race fantasy scores

If those endpoints are unavailable or change structure, the app may show warnings or fall back to cached/manual data.

---

## Project structure

```text
f1_fantasy_optimizer/
  streamlit_app.py          # Streamlit web app

  f1fantasy/
    app_core.py             # Shared app logic
    fantasy_api.py          # F1 Fantasy market feed loading
    player_stats.py         # Player stats endpoint parsing
    model.py                # Expected-points model
    optimize.py             # Team optimiser
    transfers.py            # Transfer recommendation logic
    recommend.py            # CLI entrypoint
    ergast.py               # Historical/schedule data helpers

  data/
    current_team.json       # Optional saved current team
    cache/                  # Local cached data

  tests/
    ...                     # Unit tests
```

---

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/dnpjr/f1_fantasy_optimizer.git
cd f1_fantasy_optimizer

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

On Windows, activate the virtual environment with:

```bash
.venv\Scripts\activate
```

---

## Run locally

From the repo root:

```bash
python -m streamlit run streamlit_app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

---

## Current team JSON format

The app can import and export a simple `current_team.json` file.

Example:

```json
{
  "drivers": [11161, 121, 11051, 118, 11059],
  "constructors": [25, 27],
  "bank": 0.2,
  "free_transfers": 2
}
```

The IDs come from the F1 Fantasy feed. The app displays names and prices in the UI, but stores IDs in the JSON so the file remains stable.

---

## CLI usage

The original command-line workflow is still available:

```bash
python -m f1fantasy.recommend
```

Depending on your environment, console scripts may also be available:

```bash
f1fantasy-recommend
f1fantasy-debug
```

The Streamlit app is the main interface, but the CLI remains useful for quick checks and debugging.

---

## Testing

Run the test suite with:

```bash
python -m pytest
```

Compile-check the app and package with:

```bash
python -m compileall f1fantasy streamlit_app.py tests
```

---

## Notes and limitations

- This is an unofficial modelling tool.
- F1 Fantasy price-change rules are inferred from community observations and live data, not guaranteed official documentation.
- Expected points and transfer recommendations are estimates.
- Public F1 Fantasy endpoints can change, fail, or become temporarily unavailable.
- Chip logic affects expected points only, not price-change projections.
- The app is intended as a decision-support tool, not an automatic guarantee of optimal fantasy performance.

---

## Deployment

The app is deployed on Streamlit Community Cloud at:

[https://f1optimise.streamlit.app/](https://f1optimise.streamlit.app/)

The Streamlit entrypoint is:

```text
streamlit_app.py
```

To redeploy/update the live app, push changes to the connected GitHub branch. Streamlit Community Cloud will rebuild the app from the repo.

The deployed app depends on the same public F1 Fantasy data feeds used locally. If those feeds change or become temporarily unavailable, the live app may show warnings or stale data until the feed is available again.
