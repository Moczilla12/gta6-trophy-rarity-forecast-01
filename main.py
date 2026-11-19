#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates on Exophase at a case's target time.

Reads one case file (schema makefun/gta6-trophies-case/v1) from
$MAKEFUN_INPUT_DIR (or /input) and writes one forecast file to
$MAKEFUN_OUTPUT_DIR (or /output): for every trophy, five quantiles
(5/25/50/75/95%) of its earn rate at the case's target.

Model: earners-over-players decomposition in log time, dilution-aware.
------------------------------------------------------------------------
An earn rate is a ratio:  rate = earners / players_tracked.  Over launch
week both parts move fast, and they move for different reasons:

* players_tracked grows as more players link their PSN to Exophase.
  Every new tracked player starts with (nearly) no trophies, so a jump
  in tracked players DILUTES the rate of every trophy the newcomers have
  not earned yet.  The house saw exactly this on launch night: one
  refresh took tracked players 638 -> 3,868 and every rate but the 100%
  one fell.
* earners grow as the tracked base plays: story trophies fast (players
  reach each chapter beat in order), grind / collectible / platinum
  trophies slowly.

Fitting the rate directly confuses the two.  So this program fits each
side separately in log space against log time since release, then divides:

    E_i(t) ~ exp(a_i + b_i * ln(1 + t))     earners of trophy i
    P(t)   ~ exp(a_P + b_P * ln(1 + t))     players tracked
    r_i    = 100 * E_i(target) / P(target)

Details that matter:

* Exophase refreshes on its own schedule, so the house's reads repeat
  the same page for hours.  Consecutive identical reads are collapsed
  to one snapshot before fitting; otherwise the fit is dominated by
  frozen plateaus.
* Fits are weighted least squares over the last 24h of distinct
  snapshots, with recency weights (tau = 24h) and a sqrt(P) precision
  weight, so the launch-night freeze and the first refresh do not drag
  the slope.  A 24h window beats 48h in backtest: launch growth
  decelerates, so the recent slope is the one that extrapolates.
* Each trophy's earner slope is shrunk toward its tier's mean slope
  (empirical Bayes, pseudo-count 6): trophies with few distinct reads
  borrow strength from their tier instead of extrapolating noise.
* Extrapolated slopes are damped to 80% around the fit's centre: growth
  keeps decelerating past the fit window, so the undamped line overshoots
  the 4.5-day target.
* The answering read shows the page as of Exophase's last refresh before
  it, so the forecast is evaluated at target minus half the typical
  refresh interval, estimated from the gaps between distinct snapshots.
* The spread is honest about five things: fit residual, the
  players-tracked fit's residual, refresh-lag uncertainty, a refresh-jump
  floor, and a random-walk term that grows with the horizon.  It is
  floored and capped.
* When the history is a single frozen page (the practice cases), there
  is no trend to fit.  The fallback then applies the launch-night
  dilution prior: the next refresh multiplies tracked players several-
  fold while earners grow only a little, so rates fall.  The shift is
  a fixed -1.8 logit.  That number is the observed first refresh
  itself: tracked players went 638 -> 3,868 (6.06x), and for a small
  rate a 6.06x dilution is -ln(6.06) = -1.8 logit.  It is fixed
  because Exophase refreshes on its own clock, hours apart — the
  launch-night page sat unchanged for about four and a half hours
  before that one refresh — so the house's read cadence (every 30
  minutes early on) is not the refresh interval, and counting
  read-cadence intervals as refreshes would pretend several 6x jumps
  happen where only one does.  Later refreshes also dilute far less
  than the first, because the tracked base is already large, so
  stacking more -1.8 shifts would over-dilute.  The fixed -1.8 is
  the one-refresh dilution the practice cases calibrated, and the
  spread stays wide, because the dilution magnitude is still uncertain.
  But the tracked cohort is not frozen: those 638 players are the
  earliest, most active adopters, and they keep playing and earning
  in the ~90 minutes between the last frozen read and the target,
  while the newcomers contribute almost no earners yet.  So the
  earners behind the median grow first — 1.6x for a story/secret
  trophy the cohort races through, 1.4x for another bronze activity,
  1.2x for a silver/gold/platinum grind — and only then does the
  6.06x dilution divide them.  The -1.8 dilution itself is unchanged;
  the growth offsets part of it on the logit scale (+ln(1.6) = +0.47
  for a story trophy).  The stale spread is narrowed 25% in total:
  v2's 90% coverage ran 0.9818 against a 0.90 target, so the whole
  stale sigma is scaled by 0.75 — the median is now close (typical
  miss 0.05pp), so a wide interval only costs pinball loss.
* A trophy at 100% stays (near) 100%: every tracked player earns the
  prologue trophy within minutes, and it survived the 6x refresh at 100%.

Safety: any per-trophy failure falls back to persistence plus a
random-walk spread (the house baseline's shape), and any whole-case
failure falls back the same way for every trophy, so the program always
writes a valid file.  No randomness anywhere: the same case always gives
the same forecast.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so an
# import of a module shipped beside this file would find it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# --- Model constants (fixed in advance, never tuned on the practice) ---
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# Random-walk spread per sqrt(day) on the logit scale.
RW_SIGMA = 0.5
# Refresh-jump floor, at launch (age 0): one Exophase refresh can move
# a rate by this much on the logit scale, whatever the horizon.
REFRESH_SIGMA_LAUNCH = 0.55
# Refresh-jump floor once the player base is mature.
REFRESH_SIGMA_MATURE = 0.25
# The refresh floor decays with the game's age (days) at this rate.
REFRESH_DECAY_DAYS = 2.0
# How much the median falls when the page is stale (launch night): a
# fixed -1.8 logit.  That is the observed first refresh itself —
# 638 -> 3,868 tracked players (6.06x), and for a small rate a 6.06x
# dilution is -ln(6.06) = -1.80 logit.  It stays fixed because
# Exophase refreshes hours apart on its own clock, not every read:
# the house read every 30 minutes while the page sat unchanged for
# ~4.5 hours, so the read cadence is not a refresh interval and
# counting it as one would stack several 6x jumps where only one
# happens.  Later refreshes dilute far less than the first (the base
# is already large), so no extra shift is added for a longer horizon.
STALE_SHIFT = -1.8
# Earner growth in the stale case (v3): the 638 tracked players are
# the earliest, most active adopters and they keep playing and
# earning between the last frozen read and the target, while the
# newcomers contribute almost no earners yet.  Growth by trophy
# kind: a story/secret trophy the cohort races through in the
# opening hours, 1.6x; another bronze activity, 1.4x; a silver,
# gold or platinum grind, 1.2x — those take far longer than ~90
# minutes to earn, so the cohort barely adds earners there.
# (v2 used 1.5x / 1.3x / 1.15x; v2's loss improved only a little,
# so the median was still slightly low and the factors rise a touch.)
STALE_GROWTH_SECRET = 1.6
STALE_GROWTH_BRONZE = 1.4
STALE_GROWTH_HARD = 1.2
# Refresh-jump floor scale in the stale spread (v2): v1's 90%
# coverage was 0.9636 against a 0.90 target, so the floor is
# scaled down 10%.  The random-walk term is unchanged.
STALE_FLOOR_SCALE = 0.9
# Total stale-spread scale (v4): v3's 90% coverage was still 0.9636
# against a 0.90 target — 53 of 55 trophies inside the interval —
# so the total stale sigma is narrowed a further 20% on top of
# v3's 0.75, i.e. 0.60.  The median is roughly right now (typical
# miss 0.06pp), so the spread, not the median, is the main lever
# left.  The median and the earner-growth factors are untouched
# in this iteration, so only the interval width moves.
# Applies to the stale branch only.
STALE_SPREAD_SCALE = 0.60
# Extra uncertainty (logit) about the dilution magnitude on a stale page.
STALE_SIGMA = 0.45
# A stale page only counts while the game is this young (hours).
STALE_MAX_AGE_HOURS = 24.0
# Recency weight half-life for the trend fit (hours).
WEIGHT_TAU_HOURS = 24.0
# Fit window: last this many hours of distinct snapshots.
FIT_WINDOW_HOURS = 24.0
# Empirical-Bayes pseudo-count for shrinking a trophy's earner slope
# toward its tier's mean slope.
TIER_PSEUDO_COUNT = 6.0
# Damping applied to extrapolated slopes around the fit centre.
SLOPE_DAMP = 0.8
# Cap on the logit spread, so a noisy fit cannot blow the quantiles out.
MAX_SIGMA = 2.5
# Floor on the logit spread, so it never collapses to nothing.
MIN_SIGMA = 0.15
# Cap on a fitted log-time slope (earners / players never grow faster).
MAX_SLOPE = 3.0
# Narrow spread for a trophy every tracked player already holds.
FULL_SPREAD = 0.05
# Continuity correction: a rate of 0.00% means fewer than ~0.5 earners
# are visible at the printed precision, not necessarily zero.
EARNER_FLOOR = 0.5
# Default refresh interval (minutes) when it cannot be estimated.
DEFAULT_REFRESH_MIN = 240.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it.  A whole number comes back
    as an int, so json.dumps writes 100 where JavaScript writes 100
    (not 100.0)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(t: datetime) -> float:
    # Clamped so the log-time axis stays defined for reads just before
    # midnight on release day (the first read is 23:50 on 18 Nov).
    return max((t - RELEASE).total_seconds() / 3600.0, -0.9)


def log_time(t: datetime) -> float:
    return math.log1p(hours_since_release(t))


def quantiles_from_logit(lg: float, sigma: float) -> list:
    sigma = min(max(sigma, MIN_SIGMA), MAX_SIGMA)
    qs = [round4(clamp(inv_logit(lg + z * sigma))) for z in Z]
    # Rounding can only make neighbours equal, never decreasing, but
    # enforce monotonicity explicitly so the row is always valid.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def persistence_row(rate: float, horizon_minutes: float) -> list:
    """House-baseline shape: last rate, random-walk spread."""
    spread = RW_SIGMA * math.sqrt(horizon_minutes / 1440.0)
    return quantiles_from_logit(logit(rate if rate is not None else 0), spread)


def collapse_snapshots(history: list) -> list:
    """Collapse consecutive identical reads (same players_tracked and
    same rates) to one snapshot each, keeping the first occurrence's
    time.  Returns a list of dicts with slot (datetime), players and
    rates."""
    snaps = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if snaps and snaps[-1]["key"] == key:
            continue
        snaps.append({
            "key": key,
            "slot": parse_time(read["slot"]),
            "players": read.get("players_tracked"),
            "rates": list(read.get("rates") or ()),
        })
    return snaps


def refresh_interval_minutes(snaps: list) -> float:
    """Median gap between distinct snapshots: an estimate of Exophase's
    refresh cadence.  Falls back to a default when it cannot be seen."""
    if len(snaps) < 2:
        return DEFAULT_REFRESH_MIN
    gaps = []
    for a, b in zip(snaps, snaps[1:]):
        g = (b["slot"] - a["slot"]).total_seconds() / 60.0
        if g > 0:
            gaps.append(g)
    if not gaps:
        return DEFAULT_REFRESH_MIN
    gaps.sort()
    med = gaps[len(gaps) // 2] if len(gaps) % 2 else 0.5 * (gaps[len(gaps) // 2 - 1] + gaps[len(gaps) // 2])
    return min(max(med, 30.0), 720.0)


def wls_line(xs: list, ys: list, ws: list):
    """Weighted least-squares line y ~ a + b*x.  Returns (a, b, x_bar,
    y_bar, resid_rms) or None when the slope is not identified."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    x_bar = sum(w * x for w, x in zip(ws, xs)) / sw
    y_bar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - x_bar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - x_bar) * (y - y_bar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = y_bar - b * x_bar
    resid = math.sqrt(sum(w * (y - a - b * x) ** 2 for w, x, y in zip(ws, xs, ys)) / sw)
    return (a, b, x_bar, y_bar, resid)


def refresh_floor(age_days: float) -> float:
    """One refresh's logit jump: largest at launch, decaying to the
    mature floor as the tracked base grows."""
    return REFRESH_SIGMA_MATURE + (REFRESH_SIGMA_LAUNCH - REFRESH_SIGMA_MATURE) * math.exp(-age_days / REFRESH_DECAY_DAYS)


def stale_growth(trophy: dict) -> float:
    """Earner-growth factor for a trophy in the stale case: how much
    the tracked cohort's earners grow between the last frozen read
    and the target.  Story/secret beats the cohort races through
    grow fastest; a grind grows slowest."""
    if trophy.get("secret"):
        return STALE_GROWTH_SECRET
    if trophy.get("tier", "bronze") == "bronze":
        return STALE_GROWTH_BRONZE
    return STALE_GROWTH_HARD


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_minutes = float(case["horizon_minutes"])
    horizon_days = horizon_minutes / 1440.0
    age_days = max(hours_since_release(at) / 24.0, 0.0)

    snaps = collapse_snapshots(case["history"])
    last = case["history"][-1]
    last_rates = list(last.get("rates") or ())

    # --- Stale page: a single frozen snapshot and a game under a day
    # old.  There is no trend to fit: the page has not refreshed since
    # launch night, the tracked count is about to jump and every rate
    # but the one at 100% is about to fall by dilution.  Shift the
    # median down by the observed first-refresh dilution and keep a
    # wide spread.  This fires on the launch-night practice cases and
    # never once the history shows real movement.
    stale = len(snaps) == 1 and hours_since_release(at) < STALE_MAX_AGE_HOURS
    if stale:
        rw = RW_SIGMA * math.sqrt(horizon_days)
        # v2: the refresh-jump floor is scaled down 10% — v1's 90%
        # coverage ran 0.9636 against a 0.90 target.
        floor = refresh_floor(age_days) * STALE_FLOOR_SCALE
        # v3: the whole stale sigma is scaled by 0.75 — v2's 90%
        # coverage ran 0.9818 against a 0.90 target, so the spread
        # was far too wide and a wide interval only costs pinball
        # loss now that the median is close.
        sigma = math.sqrt(rw ** 2 + floor ** 2 + STALE_SIGMA ** 2) * STALE_SPREAD_SCALE
        # Fixed one-refresh dilution: -1.8 logit, the observed
        # 638 -> 3,868 jump.  Not scaled by the horizon — the read
        # cadence is not the refresh interval, and later refreshes
        # dilute far less than the first.
        shift = STALE_SHIFT
        rows = []
        for i, trophy in enumerate(trophies):
            rate = last_rates[i] if i < len(last_rates) else None
            if rate is not None and rate >= 99.99:
                rows.append({"id": trophy["id"], "q": quantiles_from_logit(logit(99.99), FULL_SPREAD)})
            else:
                # v3: the cohort keeps earning into the refresh, so
                # its earners grow by the trophy's factor before the
                # dilution divides them: earners_target =
                # earners_last * growth, rate = 100 * earners_target
                # / (players_last * 6.06).  On the logit scale the
                # growth offsets part of the -1.8 shift (+ln(1.6)
                # = +0.47 for a story trophy); the -1.8 itself is
                # unchanged.
                g = stale_growth(trophy)
                rows.append({"id": trophy["id"], "q": quantiles_from_logit(logit(rate if rate is not None else 0) + shift + math.log(g), sigma)})
        return {"trophies": rows}

    # --- Moving history: earners/players decomposition. ---
    # Fit window: last 24h of distinct snapshots; widen to all of them
    # when that leaves too few points to identify a slope.
    window_start = at.timestamp() - FIT_WINDOW_HOURS * 3600.0
    fit_snaps = [s for s in snaps if s["slot"].timestamp() >= window_start]
    if len(fit_snaps) < 2:
        fit_snaps = snaps

    # Weights: recency (tau 24h) times sqrt(players) precision.
    def weight(s):
        age_h = max((at - s["slot"]).total_seconds() / 3600.0, 0.0)
        p = s["players"] or 0
        return math.exp(-age_h / WEIGHT_TAU_HOURS) * math.sqrt(max(p, 1))

    xs = [log_time(s["slot"]) for s in fit_snaps]
    ws = [weight(s) for s in fit_snaps]

    # Players fit: ln(P) ~ a + b * ln(1 + hours).
    ys_p = [math.log(max(s["players"] or 1, 1)) for s in fit_snaps]
    fit_p = wls_line(xs, ys_p, ws)
    if fit_p is None:
        raise ValueError(f"{name}: players fit not identified")
    a_p, b_p, x_bar_p, y_bar_p, resid_p = fit_p
    b_p = min(max(b_p, 0.0), MAX_SLOPE)
    a_p = y_bar_p - b_p * x_bar_p

    # Evaluation time: the answering read shows the page as of
    # Exophase's last refresh before the target.
    refresh_min = refresh_interval_minutes(snaps)
    from datetime import timedelta
    eval_time = target - timedelta(minutes=0.5 * refresh_min)
    x_eval = log_time(eval_time)

    def predict(fit, x_bar, y_bar, b, damp=True):
        """Log-space prediction at x_eval, damping the extrapolated
        slope to 80% around the fit centre."""
        dx = x_eval - x_bar
        if damp and dx > 0:
            dx = SLOPE_DAMP * dx
        return y_bar + b * dx

    log_p_pred = predict(fit_p, x_bar_p, y_bar_p, b_p)
    p_pred = math.exp(log_p_pred)

    # Earner fits, one per trophy, shrunk toward the tier mean slope.
    tier_slopes = {}
    tier_counts = {}
    raw_fits = []
    for i in range(n):
        ys_e = []
        ok = True
        for s in fit_snaps:
            r = s["rates"][i] if i < len(s["rates"]) else None
            p = s["players"] or 0
            e = (r * p / 100.0) if r is not None else 0.0
            ys_e.append(math.log(max(e, EARNER_FLOOR)))
        fit_e = wls_line(xs, ys_e, ws)
        if fit_e is None:
            raw_fits.append(None)
            continue
        a_e, b_e, x_bar_e, y_bar_e, resid_e = fit_e
        b_e = min(max(b_e, 0.0), MAX_SLOPE)
        raw_fits.append((a_e, b_e, x_bar_e, y_bar_e, resid_e))
        tier = trophies[i].get("tier", "bronze")
        tier_slopes[tier] = tier_slopes.get(tier, 0.0) + b_e
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    tier_mean = {t: tier_slopes[t] / tier_counts[t] for t in tier_slopes}
    # Fallback tier mean over all trophies, for a tier with no fit.
    all_mean = (sum(tier_slopes.values()) / sum(tier_counts.values())) if tier_counts else 0.0

    rw = RW_SIGMA * math.sqrt(horizon_days)
    floor = refresh_floor(age_days)
    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i] if i < len(last_rates) else None
        if rate is not None and rate >= 99.99:
            rows.append({"id": trophy["id"], "q": quantiles_from_logit(logit(99.99), FULL_SPREAD)})
            continue
        fit_e = raw_fits[i]
        if fit_e is None:
            # No earner trend for this trophy: persistence + random walk.
            rows.append({"id": trophy["id"], "q": persistence_row(rate, horizon_minutes)})
            continue
        a_e, b_e, x_bar_e, y_bar_e, resid_e = fit_e
        tier = trophy.get("tier", "bronze")
        tm = tier_mean.get(tier, all_mean)
        k = len(fit_snaps)
        b_shrunk = (k * b_e + TIER_PSEUDO_COUNT * tm) / (k + TIER_PSEUDO_COUNT)
        b_shrunk = min(max(b_shrunk, 0.0), MAX_SLOPE)
        log_e_pred = y_bar_e + b_shrunk * (SLOPE_DAMP * (x_eval - x_bar_e) if x_eval > x_bar_e else (x_eval - x_bar_e))
        e_pred = math.exp(log_e_pred)
        rate_pred = clamp(100.0 * e_pred / max(p_pred, 1.0))
        # Refresh-lag uncertainty: the rate moves while the page sits
        # unrefreshed, at the fitted logit speed of the ratio.
        hours_eval = max(hours_since_release(eval_time), 0.0)
        dlogit_dt = abs(b_shrunk - b_p) / (1.0 + hours_eval)
        lag_sigma = dlogit_dt * (0.5 * refresh_min / 60.0)
        sigma = math.sqrt(
            rw ** 2
            + max(resid_e, 0.1) ** 2
            + max(resid_p, 0.1) ** 2
            + lag_sigma ** 2
            + floor ** 2
        )
        rows.append({"id": trophy["id"], "q": quantiles_from_logit(logit(rate_pred), sigma)})
    return {"trophies": rows}


def forecast_case_safe(case: dict, name: str) -> dict:
    """Whole-case fallback: any failure forecasts persistence plus a
    random-walk spread for every trophy, so a valid file is always
    written."""
    try:
        return forecast_case(case, name)
    except Exception:
        horizon_minutes = float(case.get("horizon_minutes", 0) or 0)
        last = case["history"][-1] if case.get("history") else {"rates": []}
        last_rates = list(last.get("rates") or ())
        rows = []
        for i, trophy in enumerate(case["trophies"]):
            rate = last_rates[i] if i < len(last_rates) else None
            rows.append({"id": trophy["id"], "q": persistence_row(rate, horizon_minutes)})
        return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case_safe(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
