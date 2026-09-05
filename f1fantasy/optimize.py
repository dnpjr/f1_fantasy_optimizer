from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpStatus, PULP_CBC_CMD, value


@dataclass
class TeamSolution:
    drivers: pd.DataFrame
    constructors: pd.DataFrame
    boosted_driver: Optional[str]
    no_negative: bool
    limitless: bool
    total_cost: float
    expected_score: float
    triple_driver: Optional[str] = None


def _solve_once(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    budget: float | None = 100.0,
    drs_multiplier: float = 2.0,
    allow_no_negative: bool = False,
    exclude: Optional[List[Tuple[List[str], List[str]]]] = None,
    locked_driver_ids: Optional[List[str]] = None,
    excluded_driver_ids: Optional[List[str]] = None,
    locked_constructor_ids: Optional[List[str]] = None,
    excluded_constructor_ids: Optional[List[str]] = None,
    objective_col: str = "exp_score",
    boost_col: str = "exp_score",
    triple_multiplier: float | None = None,
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

    locked_driver_set = set(map(str, locked_driver_ids or []))
    excluded_driver_set = set(map(str, excluded_driver_ids or []))
    locked_constructor_set = set(map(str, locked_constructor_ids or []))
    excluded_constructor_set = set(map(str, excluded_constructor_ids or []))

    prob = LpProblem("f1_fantasy", LpMaximize)

    xd = {i: LpVariable(f"d_{i}", cat=LpBinary) for i in range(len(d))}
    xc = {i: LpVariable(f"c_{i}", cat=LpBinary) for i in range(len(c))}

    # boosted driver (DRS/Boost): pick exactly one selected driver
    boost = {i: LpVariable(f"boost_{i}", cat=LpBinary) for i in range(len(d))}
    for i in range(len(d)):
        prob += boost[i] <= xd[i]
    prob += lpSum(boost[i] for i in range(len(d))) == 1

    triple = None
    if triple_multiplier is not None:
        triple = {i: LpVariable(f"triple_{i}", cat=LpBinary) for i in range(len(d))}
        for i in range(len(d)):
            prob += triple[i] <= xd[i]
            prob += boost[i] + triple[i] <= 1
        prob += lpSum(triple[i] for i in range(len(d))) == 1

    # Roster constraints
    prob += lpSum(xd.values()) == 5
    prob += lpSum(xc.values()) == 2

    # User constraints
    for i in range(len(d)):
        asset_id = str(d.loc[i, "id"])
        if asset_id in locked_driver_set:
            prob += xd[i] == 1
        if asset_id in excluded_driver_set:
            prob += xd[i] == 0

    for i in range(len(c)):
        asset_id = str(c.loc[i, "id"])
        if asset_id in locked_constructor_set:
            prob += xc[i] == 1
        if asset_id in excluded_constructor_set:
            prob += xc[i] == 0

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

    # Objective: use nn_exp_score if scenario enabled and column exists, unless an explicit objective column is provided.
    use_col_d = objective_col
    use_col_c = objective_col
    use_boost_col = boost_col
    if objective_col == "exp_score" and allow_no_negative:
        use_col_d = "nn_exp_score" if "nn_exp_score" in d.columns else "exp_score"
        use_col_c = "nn_exp_score" if "nn_exp_score" in c.columns else "exp_score"
        use_boost_col = use_col_d

    obj = (
        lpSum(xd[i] * float(d.loc[i, use_col_d]) for i in range(len(d)))
        + lpSum(xc[i] * float(c.loc[i, use_col_c]) for i in range(len(c)))
        + lpSum(boost[i] * float(d.loc[i, use_boost_col]) for i in range(len(d))) * float(drs_multiplier - 1.0)
    )
    if triple is not None:
        obj += lpSum(triple[i] * float(d.loc[i, use_boost_col]) for i in range(len(d))) * float(triple_multiplier - 1.0)
    prob += obj

    prob.solve(PULP_CBC_CMD(msg=False))

    if LpStatus.get(prob.status) != "Optimal":
        return TeamSolution(
            drivers=d.iloc[0:0].copy(),
            constructors=c.iloc[0:0].copy(),
            boosted_driver=None,
            no_negative=bool(allow_no_negative),
            limitless=bool(limitless),
            total_cost=0.0,
            expected_score=0.0,
            triple_driver=None,
        )

    chosen_d = d[[xd[i].value() == 1 for i in range(len(d))]].copy()
    chosen_c = c[[xc[i].value() == 1 for i in range(len(c))]].copy()

    boosted_name = None
    triple_name = None
    for i in range(len(d)):
        if boost[i].value() == 1:
            boosted_name = str(d.loc[i, "name"])
        if triple is not None and triple[i].value() == 1:
            triple_name = str(d.loc[i, "name"])

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
        triple_driver=triple_name,
    )


def optimize_top_k(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    budget: float | None = 100.0,
    k: int = 5,
    drs_multiplier: float = 2.0,
    allow_no_negative: bool = False,
    locked_driver_ids: Optional[List[str]] = None,
    excluded_driver_ids: Optional[List[str]] = None,
    locked_constructor_ids: Optional[List[str]] = None,
    excluded_constructor_ids: Optional[List[str]] = None,
    objective_col: str = "exp_score",
    boost_col: str = "exp_score",
    triple_multiplier: float | None = None,
    excluded_team_combinations: Optional[
        Sequence[Tuple[Sequence[str], Sequence[str]]]
    ] = None,
) -> List[TeamSolution]:
    solutions: List[TeamSolution] = []
    excludes: List[Tuple[List[str], List[str]]] = [
        (list(map(str, driver_ids)), list(map(str, constructor_ids)))
        for driver_ids, constructor_ids in (excluded_team_combinations or ())
    ]

    d = drivers.copy()
    c = constructors.copy()

    if "id" not in d.columns:
        d["id"] = d.get("playerId", d.get("driverId", d.index)).astype(str)
    if "id" not in c.columns:
        c["id"] = c.get("teamId", c.get("constructorId", c.index)).astype(str)

    if objective_col not in d.columns or objective_col not in c.columns:
        raise ValueError(f"Objective column is not available on both asset tables: {objective_col}")
    if boost_col not in d.columns:
        raise ValueError(f"Boost column is not available on driver table: {boost_col}")

    locked_driver_set = set(map(str, locked_driver_ids or []))
    excluded_driver_set = set(map(str, excluded_driver_ids or []))
    locked_constructor_set = set(map(str, locked_constructor_ids or []))
    excluded_constructor_set = set(map(str, excluded_constructor_ids or []))

    driver_ids = set(d["id"].astype(str))
    constructor_ids = set(c["id"].astype(str))

    driver_overlap = locked_driver_set & excluded_driver_set
    constructor_overlap = locked_constructor_set & excluded_constructor_set
    if driver_overlap:
        raise ValueError(f"Driver IDs cannot be both locked and excluded: {sorted(driver_overlap)}")
    if constructor_overlap:
        raise ValueError(f"Constructor IDs cannot be both locked and excluded: {sorted(constructor_overlap)}")
    if not locked_driver_set <= driver_ids:
        missing = sorted(locked_driver_set - driver_ids)
        raise ValueError(f"Locked driver IDs are not available: {missing}")
    if not excluded_driver_set <= driver_ids:
        missing = sorted(excluded_driver_set - driver_ids)
        raise ValueError(f"Excluded driver IDs are not available: {missing}")
    if not locked_constructor_set <= constructor_ids:
        missing = sorted(locked_constructor_set - constructor_ids)
        raise ValueError(f"Locked constructor IDs are not available: {missing}")
    if not excluded_constructor_set <= constructor_ids:
        missing = sorted(excluded_constructor_set - constructor_ids)
        raise ValueError(f"Excluded constructor IDs are not available: {missing}")
    if len(locked_driver_set) > 5:
        raise ValueError("Cannot lock more than 5 drivers.")
    if len(locked_constructor_set) > 2:
        raise ValueError("Cannot lock more than 2 constructors.")
    if len(driver_ids - excluded_driver_set) < 5:
        raise ValueError("Excluding those drivers leaves fewer than 5 available drivers.")
    if len(constructor_ids - excluded_constructor_set) < 2:
        raise ValueError("Excluding those constructors leaves fewer than 2 available constructors.")

    if budget is not None:
        locked_driver_cost = d[d["id"].astype(str).isin(locked_driver_set)]["price"].astype(float).sum()
        locked_constructor_cost = c[c["id"].astype(str).isin(locked_constructor_set)]["price"].astype(float).sum()
        if float(locked_driver_cost + locked_constructor_cost) > float(budget):
            raise ValueError("Locked assets cost more than the selected budget.")

    for _ in range(k):
        sol = _solve_once(
            d,
            c,
            budget=budget,
            drs_multiplier=drs_multiplier,
            allow_no_negative=allow_no_negative,
            exclude=excludes,
            locked_driver_ids=list(locked_driver_set),
            excluded_driver_ids=list(excluded_driver_set),
            locked_constructor_ids=list(locked_constructor_set),
            excluded_constructor_ids=list(excluded_constructor_set),
            objective_col=objective_col,
            boost_col=boost_col,
            triple_multiplier=triple_multiplier,
        )
        if sol.drivers.empty or sol.constructors.empty:
            break
        solutions.append(sol)
        excludes.append((sol.drivers["id"].astype(str).tolist(), sol.constructors["id"].astype(str).tolist()))
    return solutions
