from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Optional, Dict

import pandas as pd

@dataclass
class TransferRecommendation:
    outgoing: List[str]
    incoming: List[str]
    num_transfers: int
    penalty_points: int
    new_cost: float
    new_expected: float
    delta_expected_after_penalty: float

def _count_transfers(old_ids: List[str], new_ids: List[str]) -> int:
    return len(set(old_ids).symmetric_difference(set(new_ids))) // 2 if old_ids and new_ids else len(set(old_ids).symmetric_difference(set(new_ids)))

def best_two_transfer_move(
    current_driver_ids: List[str],
    current_constructor_ids: List[str],
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    budget: float = 100.0,
    free_transfers: int = 2,
    max_transfers_considered: int = 4,
) -> List[TransferRecommendation]:
    """Enumerate best moves up to max_transfers_considered assets changed.
    We treat each changed driver/constructor as 1 transfer; > free_transfers costs -10 each. citeturn4view1
    """
    # Build current team frames
    d_cur = drivers[drivers["id"].astype(str).isin([str(x) for x in current_driver_ids])].copy()
    c_cur = constructors[constructors["id"].astype(str).isin([str(x) for x in current_constructor_ids])].copy()

    if len(d_cur) != 5 or len(c_cur) != 2:
        return []

    base_cost = float(d_cur["price"].sum() + c_cur["price"].sum())
    base_exp = float(d_cur["exp_score"].sum() + c_cur["exp_score"].sum())

    all_driver_ids = drivers["id"].astype(str).tolist()
    all_constructor_ids = constructors["id"].astype(str).tolist()

    recs: List[TransferRecommendation] = []

    # Consider swapping up to the full team:
    # 0..5 drivers and 0..2 constructors.
    # The max_transfers_considered parameter can still cap the search below 7 if desired.
    for d_k in range(0, 6):
        for c_k in range(0, 3):
            if d_k + c_k == 0:
                continue
            if d_k + c_k > max_transfers_considered:
                continue

            for d_out in combinations(current_driver_ids, d_k):
                remaining_d = [x for x in current_driver_ids if x not in d_out]
                # choose incoming drivers not already owned
                candidates_d = [x for x in all_driver_ids if x not in set(current_driver_ids)]
                for d_in in combinations(candidates_d, d_k):
                    new_d_ids = remaining_d + list(d_in)

                    for c_out in combinations(current_constructor_ids, c_k):
                        remaining_c = [x for x in current_constructor_ids if x not in c_out]
                        candidates_c = [x for x in all_constructor_ids if x not in set(current_constructor_ids)]
                        for c_in in combinations(candidates_c, c_k):
                            new_c_ids = remaining_c + list(c_in)

                            new_d = drivers[drivers["id"].astype(str).isin([str(x) for x in new_d_ids])]
                            new_c = constructors[constructors["id"].astype(str).isin([str(x) for x in new_c_ids])]
                            if len(new_d) != 5 or len(new_c) != 2:
                                continue

                            cost = float(new_d["price"].sum() + new_c["price"].sum())
                            if cost > budget:
                                continue

                            exp = float(new_d["exp_score"].sum() + new_c["exp_score"].sum())
                            transfers = d_k + c_k
                            penalty = 10 * max(0, transfers - free_transfers)
                            delta = (exp - base_exp) - penalty

                            recs.append(TransferRecommendation(
                                outgoing=[str(x) for x in list(d_out) + list(c_out)],
                                incoming=[str(x) for x in list(d_in) + list(c_in)],
                                num_transfers=transfers,
                                penalty_points=penalty,
                                new_cost=cost,
                                new_expected=exp,
                                delta_expected_after_penalty=delta,
                            ))

    recs.sort(key=lambda r: r.delta_expected_after_penalty, reverse=True)
    return recs[:10]
