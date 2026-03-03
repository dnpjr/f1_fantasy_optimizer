from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD, value


@dataclass
class TeamSolution:
    drivers: pd.DataFrame
    constructors: pd.DataFrame
    boosted_driver: Optional[str]
    no_negative: bool
    limitless: bool
    total_cost: float
    expected_score: float


def _solve_once(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    budget: float | None = 100.0,
    drs_multiplier: float = 2.0,
    allow_no_negative: bool = False,
    exclude: Optional[List[Tuple[List[str], List[str]]]] = None,
) -> TeamSolution:
    """Solve one lineup MILP.

    Linearity note:
    - PuLP/CBC solves linear MILPs. So we model 'No Negative' as a *scenario switch*:
      if allow_no_negative=True and nn_exp_score columns exist, we use those scores in the objective.
      We do NOT create a 'chip active' binary variable that multiplies selection variables (bilinear).
    """
    d = drivers.copy().reset_index(drop=True)
    c = constructors.copy().reset_index(drop=True)

    if "id" not in d.columns:
        d["id"] = d.index.astype(str)
    if "id" not in c.columns:
        c["id"] = c.index.astype(str)

    prob = LpProblem("f1_fantasy", LpMaximize)

    xd = {i: LpVariable(f"d_{i}", cat=LpBinary) for i in range(len(d))}
    xc = {i: LpVariable(f"c_{i}", cat=LpBinary) for i in range(len(c))}

    # boosted driver (DRS/Boost): pick exactly one selected driver
    boost = {i: LpVariable(f"boost_{i}", cat=LpBinary) for i in range(len(d))}
    for i in range(len(d)):
        prob += boost[i] <= xd[i]
    prob += lpSum(boost[i] for i in range(len(d))) == 1

    # Roster constraints
    prob += lpSum(xd.values()) == 5
    prob += lpSum(xc.values()) == 2

    # Budget (None means 'Limitless' scenario)
    limitless = budget is None
    if not limitless:
        prob += (
            lpSum(xd[i] * float(d.loc[i, "price"]) for i in range(len(d)))
            + lpSum(xc[i] * float(c.loc[i, "price"]) for i in range(len(c)))
            <= float(budget)
        )

    # Exclude previously found solutions to get top-k distinct teams
    if exclude:
        for drv_ids, con_ids in exclude:
            drv_set = set(map(str, drv_ids))
            con_set = set(map(str, con_ids))
            lhs = (
                lpSum(xd[i] for i in range(len(d)) if str(d.loc[i, "id"]) in drv_set)
                + lpSum(xc[i] for i in range(len(c)) if str(c.loc[i, "id"]) in con_set)
            )
            # If all 7 picks match, forbid by requiring <=6
            prob += lhs <= 6

    # Objective: use nn_exp_score if scenario enabled and column exists
    use_col_d = "nn_exp_score" if (allow_no_negative and "nn_exp_score" in d.columns) else "exp_score"
    use_col_c = "nn_exp_score" if (allow_no_negative and "nn_exp_score" in c.columns) else "exp_score"

    obj = (
        lpSum(xd[i] * float(d.loc[i, use_col_d]) for i in range(len(d)))
        + lpSum(xc[i] * float(c.loc[i, use_col_c]) for i in range(len(c)))
        + lpSum(boost[i] * float(d.loc[i, use_col_d]) for i in range(len(d))) * float(drs_multiplier - 1.0)
    )
    prob += obj

    prob.solve(PULP_CBC_CMD(msg=False))

    chosen_d = d[[xd[i].value() == 1 for i in range(len(d))]].copy()
    chosen_c = c[[xc[i].value() == 1 for i in range(len(c))]].copy()

    boosted_name = None
    for i in range(len(d)):
        if boost[i].value() == 1:
            boosted_name = str(d.loc[i, "name"])
            break

    total_cost = float(chosen_d["price"].sum() + chosen_c["price"].sum())
    expected_score = float(value(prob.objective))

    return TeamSolution(
        drivers=chosen_d,
        constructors=chosen_c,
        boosted_driver=boosted_name,
        no_negative=bool(allow_no_negative),
        limitless=bool(limitless),
        total_cost=total_cost,
        expected_score=expected_score,
    )


def optimize_top_k(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    budget: float | None = 100.0,
    k: int = 5,
    drs_multiplier: float = 2.0,
    allow_no_negative: bool = False,
) -> List[TeamSolution]:
    solutions: List[TeamSolution] = []
    excludes: List[Tuple[List[str], List[str]]] = []

    d = drivers.copy()
    c = constructors.copy()

    if "id" not in d.columns:
        d["id"] = d.get("playerId", d.get("driverId", d.index)).astype(str)
    if "id" not in c.columns:
        c["id"] = c.get("teamId", c.get("constructorId", c.index)).astype(str)

    for _ in range(k):
        sol = _solve_once(
            d,
            c,
            budget=budget,
            drs_multiplier=drs_multiplier,
            allow_no_negative=allow_no_negative,
            exclude=excludes,
        )
        if sol.drivers.empty or sol.constructors.empty:
            break
        solutions.append(sol)
        excludes.append((sol.drivers["id"].astype(str).tolist(), sol.constructors["id"].astype(str).tolist()))
    return solutions
