# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Moczilla12/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Moczilla12/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI trophy's earn rate on [Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/) at a case's target time, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy. Built for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast the rates at 2026-11-26 00:00 UTC, scored by pinball loss on the logit scale.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. No network, no randomness: the same case always gives the same forecast. Reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory.

## The model

An earn rate is a ratio: `rate = earners / players_tracked`. Over launch week both parts move fast, for different reasons — and fitting the rate directly confuses them. So `main.py` fits each side separately in log space against log time since release, then divides:

```
E_i(t) ~ exp(a_i + b_i * ln(1 + t))     earners of trophy i
P(t)   ~ exp(a_P + b_P * ln(1 + t))     players tracked
r_i    = 100 * E_i(target) / P(target)
```

**Dilution.** Exophase refreshes a game's counts on its own schedule, not with every player's trophy. On launch night the page sat unchanged — 638 players tracked, every rate the same — from 23:48 UTC through 04:00 UTC, then one refresh took it to 3,868 players and every rate but the one at 100% fell. Every new tracked player starts with (nearly) no trophies, so a jump in tracked players dilutes every rate. Modelling earners and players separately is what makes the forecast survive a refresh instead of chasing the last rate.

**Stale page.** When the whole history is a single frozen page and the game is under a day old, there is no trend to fit. The program then applies the launch-night dilution prior: the next refresh multiplies tracked players several-fold while earners grow only a little, so rates fall. The observed first refresh was 638 → 3,868 players (6.06x), which for a small rate is a logit shift of about −ln(6.06) = −1.8 — so the median is shifted down 1.8 logit with a wide spread. This fires on the launch-night practice cases and never once the history shows real movement.

**The rest.** Consecutive identical reads are collapsed to one snapshot before fitting, so frozen plateaus cannot dominate. Fits are recency-weighted least squares over the last 24 hours of distinct snapshots (recency tau 24h, plus a sqrt(players) precision weight). Each trophy's earner slope is shrunk toward its tier's mean slope (empirical Bayes, pseudo-count 6), and extrapolated slopes are damped to 80% around the fit's centre, because launch growth decelerates. The forecast is evaluated at the target minus half the typical refresh interval, since the answering read shows the page as of Exophase's last refresh before the target. The spread (logit sigma, parts added in quadrature, floored and capped) is honest about five things: the fit residual, the players fit's residual, refresh-lag uncertainty, a refresh-jump floor that decays from 0.55 at launch to 0.25 as the base matures, and a random-walk term growing with the horizon. A trophy every tracked player already holds (100%) stays at 99.99 with a narrow spread. Any per-trophy failure falls back to persistence plus a random-walk spread (the house baseline's shape), so the program always writes a valid file.
