# Discovery Phase — Finding a Real Volatility Phenomenon

**Research programme, Parts 1–17** · broker CFD M5, 2022-01 → 2026-07, six instruments
Run 2026-08-10 · `.`
330 registered tests · full log in `results/research_log.json`

---

## Bottom line

**One phenomenon is real, large, and survives everything: clock-standardised
volatility scale is strongly forecastable at 1–3 hour horizons.** It holds on
6/6 instruments, in every year, under a causal threshold, with effect sizes up
to a 55-percentage-point swing in the odds of a volatility spike. 67 of 102
tests in the volatility family survive Benjamini-Hochberg at FDR 0.10.

**No directional phenomenon survived.** The directional family ran 132
pre-specified tests. Nine had raw p < 0.05 against a chance expectation of 6.6.
**Zero survived FDR correction.** Not one.

The single most useful thing this programme found is *why* those two facts sit
together, and it is not obvious:

> Volatility **scale** is forecastable. Path **shape** is not. When a session is
> going to be quiet, everything shrinks together — the opening range, the
> excursions, the reversal distances — in the same proportion. So a forecast
> that is excellent in absolute terms conveys almost nothing about the *shape*
> of what happens, and shape is where directional edge lives.

That is a structural reason to stop hunting for a directional trigger in this
family of ideas, and it is worth more than another twenty indicator tests.

---

## Part 1 — What survived from the first study

| Finding from study 1 | Status after this programme |
|---|---|
| Compression → immediate expansion is false | **Confirmed and sharpened.** With a proper seasonal yardstick, activity in the 2h after a compression episode ends is 1.14× the seasonal norm against an unconditional baseline of 1.15×. Exactly normal. |
| Breakout from compression → continuation is false | **Confirmed decisively.** Signed return after an expansion out of compression straddles zero at k = 1, 3, 6, 12, 24 on all six instruments. |
| Compression is confounded with the Asian session | **Confirmed and then fixed.** 91% of raw compression bars on US100 are Asian-session. After clock standardisation that falls to 36% — the session share of the tape. |
| Compression shows volatility persistence | **Confirmed and this is the real phenomenon.** See below. |
| Costs are large relative to ATR in compression | **Confirmed.** Spread runs 3.6%–12.3% of the impulse-bar range at event time. |
| XAUUSD and EURUSD looked more interesting | **Partly.** They are cleaner (compression less clock-driven), but the headline phenomenon is equally strong on the US indices. |

### The measurement that changed everything

Study 1's compression detector was largely a clock. This programme replaced it:

```
S_tod(t) = median ATR14 in the SAME 5-minute slot over the previous 60
           sessions, shifted one session (causal, no self-contribution)
abn_t    = ATR14_t / S_tod(t)      "how quiet is this, for this time of day"
```

`abn` and raw ATR percentile select largely **different bars** — Jaccard overlap
is only 0.22–0.35. This is a genuinely new variable, not a rescaling, and every
result below is built on it.

---

## Part 2 — Ranked conditional phenomena

### ① Clock-standardised volatility persistence — **STRONG**

The cleanest expression is the session-handover test (Part 5), where time of day
is controlled *exactly* by construction and each observation is one independent
day. An abnormally quiet hour going into a handover forecasts an abnormally
quiet session after it:

**P(a bar ≥2× its seasonal norm in the 2h after the handover):**

| Instrument | Boundary | after a quiet hour | otherwise | difference |
|---|---|---|---|---|
| US500.cash | Asia→London | **0.254** | 0.804 | −0.549 |
| US30.cash | Asia→London | **0.302** | 0.822 | −0.520 |
| US100.cash | Asia→London | **0.293** | 0.805 | −0.512 |
| XAUUSD | Asia→London | **0.185** | 0.654 | −0.468 |
| EURUSD | Asia→London | **0.369** | 0.722 | −0.353 |
| XAUUSD | NY cash open | **0.397** | 0.709 | −0.312 |
| GER40.cash | NY cash open | — | — | −0.336 |

All p < 0.001 on ~1,150 independent days each.

The continuous version (Part 6) is monotone across the whole distribution, not
just the tail. Quintiles of `abn`, outcome = P(clock-referenced spike within 2h),
each measured against a 5-minute-slot × calendar-month matched baseline:

| Instrument | q1 (quietest) | q2 | q3 | q4 | q5 (busiest) |
|---|---|---|---|---|---|
| US100.cash | **0.61** | 0.77 | 0.95 | 1.15 | **1.34** |
| US500.cash | 0.62 | 0.79 | 0.93 | 1.15 | 1.34 |
| EURUSD | 0.71 | 0.84 | 0.93 | 1.08 | 1.30 |
| GER40.cash | 0.71 | 0.80 | 0.92 | 1.10 | 1.27 |
| XAUUSD | 0.66 | 0.85 | 0.97 | 1.12 | 1.20 |

Monotone on 6/6, with the extreme buckets more than a factor of two apart.

**Robustness (Parts 11, 12).** Re-run with a *causal* trailing 250-session
threshold — a flag a trader could actually have set that morning — the effect
holds on every instrument, every boundary and **every single year**:

| US100 Asia→London | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| P(spike) quiet | 0.42 | 0.33 | 0.46 | 0.38 |
| P(spike) rest | 0.80 | 0.82 | 0.89 | 0.82 |

Roughly 50 year × instrument × boundary cells, all the same sign. No year
carries the result.

### ② Volatility regimes are non-Markovian — **MODERATE, and interesting**

Comparing observed k-step persistence against a first-order Markov chain fitted
to the same data (if the process were memoryless given the current state, these
would match):

**Excess over Markov P^k, P(still compressed), clock-standardised state:**

| Instrument | k=3 | k=6 | k=12 | k=24 |
|---|---|---|---|---|
| EURUSD | +0.080 | +0.151 | **+0.176** | +0.091 |
| US30.cash | +0.074 | +0.142 | +0.174 | +0.106 |
| US500.cash | +0.071 | +0.138 | +0.173 | +0.113 |
| US100.cash | +0.072 | +0.140 | +0.171 | +0.108 |
| GER40.cash | +0.073 | +0.140 | +0.172 | +0.108 |
| XAUUSD | +0.073 | +0.139 | +0.164 | +0.086 |

Remarkably uniform across instruments and peaking at ~1 hour. **How long you
have been in a state carries information the state itself does not.** Note the
clock-standardised state shows *more* excess memory than the raw state (+0.171
vs +0.095 on US100) despite lower raw persistence — deseasonalising strips out
mechanical autocorrelation and leaves genuine regime structure.

### ③ Compression depth matters, but only in the extreme tail — **MODERATE**

P(clock-referenced spike) after an episode ends, matched lift by the minimum
percentile the episode reached:

| Instrument | 0–5th | 5–10th | 10–15th | 15–20th |
|---|---|---|---|---|
| EURUSD | **1.101** | 1.066 | 1.029 | 0.985 |
| XAUUSD | **1.087** | 1.090 | 0.988 | 0.985 |
| US100.cash | **1.086** | 1.015 | 0.997 | 0.986 |
| GER40.cash | **1.070** | 0.992 | 0.988 | 0.962 |
| US500.cash | 1.046 | 1.013 | 1.045 | 1.000 |

Monotone on 5/6, significant only in the deepest bucket. Mild compression is
indistinguishable from nothing — so "compression" as a binary at the 20th
percentile is the wrong construct; only the extreme tail behaves differently.

### ④ Compression duration matters, non-monotonically — **WEAK/MIXED**

| Instrument | 1–3 bars | 4–6 | 7–12 | 13–24 | 25+ |
|---|---|---|---|---|---|
| EURUSD | 1.009 | 1.088 | **1.126** | 1.108 | 1.053 |
| XAUUSD | 0.992 | 1.078 | **1.108** | 1.098 | 1.060 |
| US100.cash | 0.996 | 1.040 | **1.092** | 1.072 | 1.070 |
| US500.cash | 1.026 | 1.010 | 1.022 | 1.055 | 1.041 |
| GER40.cash | 0.989 | 1.023 | 1.002 | 1.058 | 1.069 |

Clear hump at 7–24 bars on the three priority instruments, absent on US500 and
US30. Consistent with ② but only 3/6 instruments — treat as a lead.

---

## Part 3 — What failed, and how thoroughly

### The directional family: 132 tests, 0 survivors

**Part 7 — continuation vs reversal after an expansion out of compression.**
Signed return in units of the impulse bar's own range, entry at the next bar's
open:

| Instrument | k=1 | k=3 | k=6 | k=12 | k=24 |
|---|---|---|---|---|---|
| XAUUSD | +0.006 | +0.015 | +0.023 | +0.014 | +0.015 |
| EURUSD | −0.006 | −0.019 | −0.028 | −0.049 | −0.028 |
| US100.cash | −0.000 | −0.018 | −0.021 | −0.041 | −0.018 |
| US500.cash | +0.004 | −0.000 | +0.008 | +0.014 | −0.014 |
| US30.cash | +0.008 | −0.010 | −0.000 | −0.015 | −0.045 |
| GER40.cash | +0.013 | −0.007 | — | — | — |

Every confidence interval contains zero. MFE and MAE are symmetric to two
decimals (e.g. US100 k=12: MFE 1.375, MAE 1.409). It is neither continuation nor
reversal — it is noise.

**Parts 8–10 — the 17-variable independent screen.** Momentum at three horizons,
higher-timeframe distance and slope, VWAP distance, session-open distance,
PDH/PDL distance, position in the compression box, ADX, aligned RSI, ATR slope,
prior abnormality, impulse size, body fraction, close location. Quintile split,
q5−q1 difference in signed 12-bar return, day-block bootstrapped, one test per
variable × instrument.

- 102 tests, 9 with raw p < 0.05, chance expectation 6.6
- **0 survive BH at FDR 0.10**
- The "hits" do not agree across instruments

**Part 5 directional carry-over.** Whether direction established before a
session handover carries into the next session: every confidence interval
contains zero, on every boundary and instrument.

**The exploratory opening-range fade — a false positive I caught.** While
looking at a control arm, fading the first break of the opening range appeared
to have positive net expectancy on five of six instruments (+0.05 to +0.07
range-units, CI above zero). It was an artifact: unresolved trades were being
booked at exactly **zero** rather than marked to market, and that covered ~49%
of cases. Marking them to market flips it negative on essentially every
instrument and parameter setting:

| Instrument | 1h range, NY open | 3h range, NY open |
|---|---|---|
| XAUUSD | −0.035 R | −0.023 R |
| EURUSD | −0.011 | −0.012 |
| US100.cash | −0.010 | −0.022 |
| US500.cash | −0.038 | −0.029 |

Recorded in the log as exploratory and **dead**.

### The one weak directional lead worth pre-registering

`v_htf_slope` — higher-timeframe trend slope, signed by the breakout direction —
is negative on **6/6** instruments (q5−q1: −0.47 US100, −0.37 GER40, −0.27
US500, −0.18 XAUUSD, −0.05 EURUSD, −0.05 US30), i.e. expansions that fire
*against* the prevailing higher-timeframe move outperform those that fire with
it. Two instruments reach raw significance; none survive FDR; EURUSD breaks the
pattern at the q1 bucket. Sign agreement of 6/6 has ~3% probability per variable
under the null, but 17 correlated variables were screened and three showed it,
against a chance expectation of ~0.5 — suggestive, not established.

**This is a lead, not an edge.** It earns a pre-registered test, nothing more.

---

## Part 4 — The structural insight (Parts 14, 16)

Given how strong ① is, the obvious next step is to monetise it. The natural
consequence of "this session will be quiet" is that the opening range should
contain price better than usual. It does not:

Quiet-flagged sessions versus the rest, at the Asia→London and NY-open
boundaries — differences in the *normalised* path statistics:

| Instrument | Δ P(leaves the opening range) | p | Δ P(reverts to mid before extending) | p |
|---|---|---|---|---|
| XAUUSD | +0.018 / +0.015 | 0.117 / 0.524 | +0.055 / +0.046 | 0.147 / 0.237 |
| EURUSD | +0.020 / +0.013 | 0.167 / 0.486 | +0.059 / −0.008 | 0.121 / 0.845 |
| US100.cash | −0.000 / −0.006 | 0.999 / 0.800 | +0.018 / +0.062 | 0.671 / 0.147 |
| US500.cash | +0.018 / +0.009 | 0.229 / 0.525 | +0.021 / −0.010 | 0.613 / 0.763 |
| US30.cash | +0.026 / −0.019 | 0.036 / 0.494 | −0.002 / +0.006 | 0.943 / 0.868 |
| GER40.cash | −0.013 / +0.009 | 0.554 / 0.707 | +0.005 / −0.005 | 0.923 / 0.869 |

Twenty-four comparisons, one at p < 0.05 against a chance expectation of 1.2.
Nothing.

On a quiet session the opening range is *smaller*, the excursions are *smaller*,
and the reversal distances are *smaller* — all by the same factor. The
normalised path statistics are unchanged. **The forecast tells you the size of
the ruler, not the shape of the curve.**

This is why ① is genuinely valuable and simultaneously produces no entry signal.
Its value is in risk, not in direction:

- position sizing and stop distance that anticipate the session's scale
- switching a breakout system off on days it will not pay for its spread
- expected time-to-target and expected holding period
- **drawdown-constrained risk budgeting**, which is scale management by definition

### Costs

| Instrument | round-trip spread as share of impulse-bar range |
|---|---|
| EURUSD | 3.6% |
| XAUUSD | 3.8% |
| US30.cash | 4.7% |
| US100.cash | 5.7% |
| US500.cash | 9.1% |
| GER40.cash | 12.3% |

No directional effect measured anywhere in this programme exceeded its
instrument's spread with a confidence interval clear of zero.

---

## Part 5 — The strongest hypotheses to carry forward

### H1 — Session-scale volatility forecast **(ready to build on)**

**Hypothesis.** The abnormality of realised range in the hour before a session
handover predicts the abnormality of realised range in the two hours after it,
with an odds ratio on spike occurrence of 2–4×, independently of time of day.

**Why it might exist.** Volatility is driven by information arrival, and
information arrival is autocorrelated at the daily scale — macro calendars,
positioning, overnight news flow. A quiet Asian session means nothing has
happened; the absence of news persists into London. This is the deseasonalised
core of the ARCH/GARCH result, isolated from the intraday clock that usually
contaminates it.

**Exact definition.** `S_tod(t)` = median ATR14 in the same 5-minute slot over
the previous 60 sessions, shifted one session. `pre_abn` = Σrange over the 12
bars before the boundary ÷ Σ`S_tod` over the same bars. Flag = `pre_abn` below
its own trailing 250-session 20th percentile. Outcome = max(range/`S_tod`) over
the 24 bars after the boundary.

**Evidence.** Differences of −0.09 to −0.55 in spike probability, p ≤ 0.001,
6/6 instruments, 2 boundaries, ~1,150 independent days each, stable in all of
2022–2026 under a causal threshold. 67/102 volatility-family tests survive FDR.

**Instruments / timeframes / states.** All six. M5 input, session-scale output.
Strongest at Asia→London; weaker but present at the NY cash open.

**Costs.** Not applicable directly — it is a risk input, not a trade.

**Fails when.** It does not tell you direction, and it does not change the
normalised shape of the path. Do not expect a fade or a breakout edge from it.

**Falsification.** On data after 2026-07-29: if the flagged group's spike
probability is not at least 0.10 below the unflagged group's on ≥4 of 6
instruments at Asia→London, the hypothesis is wrong.

**Strategy potential.** High, as an overlay: volatility-targeted position sizing
and a session-level on/off filter. Directly relevant to daily-loss limits.

### H2 — Duration-dependent regime memory

**Hypothesis.** Time already spent in a volatility state predicts the transition
hazard beyond what the current state predicts.

**Why.** Volatility has long memory (fractional integration / HAR structure); a
first-order state machine cannot represent it.

**Definition.** Excess of observed P(same state at t+k) over the k-th power of
the fitted one-step transition matrix.

**Evidence.** +0.14 to +0.18 at k = 6–12, uniform across 6/6 instruments.

**Falsification.** Fit a HAR-RV model on the deseasonalised series; if it does
not beat the first-order Markov chain out of sample on the same horizons, drop it.

**Strategy potential.** Medium — it improves H1's forecast, it is not a strategy.

### H3 — Extreme-tail compression is a different object

**Hypothesis.** Only the bottom ~5% of clock-standardised volatility behaves
differently; the 5th–20th percentile band is indistinguishable from normal.

**Evidence.** Matched spike lift 1.05–1.10 in the 0–5th bucket (5/6 significant)
versus 0.96–1.00 in the 15th–20th (0/6 significant).

**Falsification.** If a pre-registered test on new data shows the 15th–20th
bucket lifting as much as the 0–5th, the "depth matters" claim is dead.

**Strategy potential.** Low on its own; it is a definition correction that
should be applied to H1.

### H4 — Counter-trend expansions outperform **(pre-register only)**

**Hypothesis.** An expansion bar firing against the prevailing higher-timeframe
slope is followed by better returns in its own direction than one firing with it.

**Why it might exist.** Exhaustion — an extended move that produces one more
impulse is more likely to be terminal than continuational.

**Evidence.** 6/6 sign agreement, 2/6 raw p < 0.05, **0/6 after FDR**. Effect at
the extreme quintile ~0.1–0.2 R against spreads of 0.04–0.12 R.

**Falsification.** Pre-register the q1-versus-q5 test on post-2026-07 data. If
the sign is not negative on ≥5 of 6 instruments, abandon it.

**Strategy potential.** Unknown. Do not build on this yet.

---

## Part 6 — Recommended next step

Do **not** build a directional strategy from this programme. Two options, in
order of expected value:

1. **Build H1 as a risk overlay and test it in a drawdown-constrained portfolio
   simulation.** The question is precise and answerable: does volatility-targeted
   sizing driven by `pre_abn` improve risk-adjusted return and drawdown versus fixed
   sizing, holding the entry logic constant? This uses the strongest result in the
   study for the thing it is actually good at.

2. **Go where shape information lives.** This programme establishes that bar-level
   volatility state cannot supply direction. The two remaining candidates are
   (a) an economic calendar, so the pre-announcement lull can be separated from
   ordinary quiet — genuinely untested, and the one part of the original
   literature never examined; and (b) order-flow/microstructure data, which the
   MT5 bar export cannot provide but the tick export could.

If neither appeals, the honest position is that this line of research has been
worked out and the next hypothesis should come from somewhere else.

---

## Reproducing

```bash
cd scripts
for s in r00_prep r0_features r1_persistence r2_transitions r3_conditional \
         r4_sessions r5_directional r6_consequence r7_orfade r8_summary; do
  python -u $s.py
done
```

Full test tree, including every failure, in `results/research_log.json` (330 entries,
each flagged pre-specified or exploratory). Seeds fixed throughout.
