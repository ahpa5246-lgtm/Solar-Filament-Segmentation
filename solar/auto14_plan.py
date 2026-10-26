"""Finite, deterministic 14-day *research* schedule; never a leaderboard optimiser.

Every experiment trains its own checkpoint with a fixed month-grouped split.
Only a dedicated final, independently reviewed submission may use test images.
"""
from __future__ import annotations

import datetime as dt
import math

START = dt.date(2026, 10, 24)
DAYS = 14
UPSTREAM = "https://github.com/FarnooshMemari/solar-filament-segmentation.git"
UPSTREAM_SHA = "ef37d2e93d4cc9e248595e791ee39fe520d392d0"
UPSTREAM_LICENSE = "MIT, Copyright (c) 2026 Farnoosh Memari"
KAGGLE_OWNER = "minahussain111"
COMPETITION = "filament-segmentation-2026"
MAX_GPU_SECONDS = 7200
MIN_REMAINING_GPU_HOURS = 2.25
MAX_DAILY_RUNS = 1

# Explicit, bounded configurations. A 0.70 public score is a goal, not a prior.
# CPU/GPU bounds can still fail; failed experiments are recorded, not silently promoted.
EXPERIMENTS = (
    (0.50, 32, 24, 160, 8, 3.0, 0.70),
    (0.50, 32, 30, 180, 8, 4.0, 0.70),
    (0.50, 32, 30, 180, 8, 2.0, 0.75),
    (0.50, 48, 24, 160, 6, 3.0, 0.70),
    (0.75, 32, 20, 140, 4, 3.0, 0.75),
    (0.50, 32, 32, 180, 8, 3.0, 0.80),
    (0.75, 32, 24, 160, 4, 4.0, 0.70),
    (0.50, 48, 28, 180, 6, 4.0, 0.75),
    (0.75, 48, 20, 120, 3, 3.0, 0.75),
    (0.50, 32, 36, 180, 8, 2.0, 0.70),
    (0.75, 32, 24, 150, 4, 2.0, 0.75),
    (0.50, 48, 30, 180, 6, 3.0, 0.80),
    (0.75, 48, 22, 140, 3, 4.0, 0.70),
    (0.50, 32, 30, 180, 8, 3.0, 0.75),
)


def day_index(today):
    """UTC calendar slots: Oct 24--Nov 6 inclusive. Out-of-window = no run."""
    if isinstance(today, dt.datetime):
        today = today.astimezone(dt.timezone.utc).date()
    if not isinstance(today, dt.date):
        raise TypeError("Expected UTC date")
    index = (today - START).days + 1
    return index if 1 <= index <= DAYS else None


def _valid_history(history):
    out = []
    if not isinstance(history, dict):
        return out
    for key, record in history.items():
        if not isinstance(record, dict) or record.get("status") != "COMPLETE":
            continue
        pq = record.get("local_tuning_pq")
        if type(pq) not in (float, int) or not math.isfinite(pq) or not 0 <= pq <= 1:
            continue
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if index < 1 or index > DAYS:
            continue
        if record.get("day") != index or record.get("source_sha") != UPSTREAM_SHA:
            continue
        out.append((float(pq), -index, index))
    return sorted(out, reverse=True)


def choose_experiment(day, history=None):
    if type(day) is not int or not 1 <= day <= DAYS:
        raise ValueError("Day must be in [1, 14]")
    scale, base, epochs, steps, batch, weight, pos_frac = EXPERIMENTS[day - 1]
    # Exploit only the best previous locally observed training config; each day
    # changes just one predeclared variable. No Kaggle leaderboard feedback.
    rank = _valid_history(history)
    if day in (10, 12, 14) and rank:
        best = history[str(rank[0][2])].get("config", {})
        if isinstance(best, dict) and all(k in best for k in
              ("scale", "base", "epochs", "steps", "batch", "pos_weight", "pos_frac")):
            try:
                scale = float(best["scale"])
                base = int(best["base"])
                batch = int(best["batch"])
                weight = float(best["pos_weight"])
                pos_frac = float(best["pos_frac"])
                epochs = min(36, max(20, int(best["epochs"]) + 3))
                steps = min(185, max(100, int(best["steps"])))
            except (TypeError, ValueError, OverflowError):
                pass
    plan = {
        "day": day, "scale": scale, "base": base, "epochs": epochs,
        "steps": steps, "batch": batch, "pos_weight": weight,
        "pos_frac": pos_frac, "seed": 42 + 19 * day,
        "tuning_seed": 2027, "confirmation_seed": 2026,
        "holdout_fraction": 0.15, "tuning_fraction_of_development": 0.18,
        "model": "Farnoosh-Memari MIT U-Net, modified fixed month-split research challenger",
        "comparison": "tuning-local PQ; NO official score; no hidden test labels",
    }
    # Preserve the exact immutable day-1 experiment JSON on Kaggle: the first
    # version may already be running when this upgrade is merged. Days 1-3
    # provide the unweighted control, while later days test boundary weighting.
    if day >= 4:
        weight_by_day = {4:1.5, 5:3.0, 6:6.0, 7:1.5, 8:3.0, 9:6.0,
                         10:3.0, 11:1.5, 12:6.0, 13:3.0, 14:1.5}
        boundary = weight_by_day[day]
        if day in (10, 12, 14) and rank:
            selected = history[str(rank[0][2])]["config"].get("boundary_weight")
            if type(selected) in (float, int) and math.isfinite(selected) and 0 <= selected <= 8:
                boundary = float(selected)
        plan["boundary_weight"] = boundary
        plan["model"] += "; optional edge-weighted BCE research arm"
    # Connectivity experiments start only after three separate boundary-weight
    # controls. All earlier Kaggle notebook plans remain byte-compatible.
    # clDice encourages thin structures to remain connected. This is a
    # training loss, NOT a local/official PQ measurement or a promise of 0.50.
    if day >= 7:
        topology_by_day = {7: 0.12, 8: 0.25, 9: 0.12, 10: 0.25,
                           11: 0.12, 12: 0.25, 13: 0.12, 14: 0.25}
        plan["topology_weight"] = topology_by_day[day]
        plan["model"] += "; differentiable skeleton-connectivity research arm"
    return plan


def notebook_slug(day):
    if type(day) is not int or not 1 <= day <= DAYS:
        raise ValueError("Invalid day")
    return f"{KAGGLE_OWNER}/solar-autoresearch-d{day:02d}"


def best_result(history):
    ranks = _valid_history(history)
    return str(ranks[0][2]) if ranks else None
