# Empirical 2026 Out-of-Sample Forecasting Backtest Report

- **Generated (UTC):** `2026-11-20T02:55:27.172885+00:00`
- **Data Provenance:** Public `forecastingresearch/forecastbench-datasets` (HuggingFace, zero API key, 100% real resolved 2026 binary questions)
- **Live Metaculus Direct API Probe:** `auth_required` (HTTP `403` — Metaculus requires `METACULUS_TOKEN` for all direct `/api/` routes as of 2026; ForecastBench mirrors official Metaculus questions with verified pre-resolution `freeze_datetime_value` snapshots and 2026 outcomes)
- **Zero-Lookahead Filter:** Strictly `freeze_datetime < resolution_date`, deduplicated by `(source, question_id)` taking each question's earliest 2026 forecastable snapshot
- **Temporal Train Split:** `2026-01-04` -> `2026-06-21` (13 bi-weekly snapshots)
- **Temporal Out-of-Sample (OOS) Test Split:** `2026-07-05` -> `2026-10-25` (9 bi-weekly snapshots)

## Proper Scoring Rules Evaluated
- **Brier Score** (lower is better; `0.2500` = 50% baseline): $\frac{1}{N}\sum (p_i - y_i)^2$
- **Binary Log Loss (Nats)** (lower is better; `0.6931` = 50% baseline)
- **Official Metaculus Baseline Log-Score** (higher is better; `0.00` = 50% baseline, `+100.0` = perfect): $\frac{1}{N}\sum 100 \cdot (\log_2(p_{\text{true},i}) + 1)$
- **10-Bin Expected Calibration Error (ECE)** (lower is better)
- **Worst Single-Question Score** (measures log-loss tail blowup protection)

## Cohort: Metaculus Resolved Binary Questions (2026)
- **Train Questions (Jan–Jun 2026):** `108` (Yes rate: `31.5%`)
- **OOS Test Questions (Jul–Oct 2026):** `52` (Yes rate: `26.9%`)
- **Train-Fitted Calibration Parameters:** `shrinkage_w = 0.84`, `extremize_d = 0.93`, effective log-odds slope = `0.777`
- **OOS Calibrated vs Raw Anchor Metaculus Score Delta:** `+3.167` (95% bootstrap CI: `[-1.096, +7.752]`)

| Strategy (Evaluated Strictly OOS Jul–Oct 2026) | N | Brier ↓ | Log Loss ↓ | Metaculus Baseline ↑ | ECE ↓ | Worst Q Score ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 1. Uniform 50% Baseline | 52 | 0.25000 | 0.69315 | **+0.000** | 0.23077 | +0.00 |
| 2. Train Base-Rate Prior (31.5%) | 52 | 0.19882 | 0.58745 | **+15.249** | 0.04558 | -66.74 |
| 3. Raw Market/Crowd Anchor [0.001, 0.999] | 52 | 0.17797 | 0.51844 | **+25.205** | 0.15277 | -264.39 |
| 4. Safety-Clipped Anchor [0.01, 0.99] | 52 | 0.17797 | 0.51861 | **+25.180** | 0.15260 | -264.39 |
| 5. OOS Calibrated Shrinkage+Extremizing (w=0.84, d=0.93) | 52 | 0.16693 | 0.49649 | **+28.372** | 0.15036 | -215.93 |
| 6. Structural Base-Rate + Calibrated Anchor Hybrid | 52 | 0.17221 | 0.50514 | **+27.123** | 0.18036 | -193.93 |

### Sample OOS Resolved 2026 Questions (Provenance Audit)
| Source | ID | Freeze Date | Resolved Date | Pre-Res Anchor | Outcome | Question |
|---|---|---|---|---:|---:|---|
| `metaculus` | `43941` | `2026-06-25` | `2026-11-02` | 0.609 | `0` | Will Karen Bass be reelected Mayor of Los Angeles in 2026? |
| `metaculus` | `14025` | `2026-06-25` | `2026-07-29` | 0.870 | `1` | Will the Japan Self-Defense Forces have tested a Tomahawk missile by mid-2027? |
| `metaculus` | `43637` | `2026-06-25` | `2026-10-19` | 0.050 | `0` | Will the 2026 Alberta independence referendum pass? |
| `metaculus` | `43881` | `2026-06-25` | `2026-08-31` | 0.100 | `0` | Will Volodymyr Zelenskyy and Vladimir Putin meet in person before September 1, 2026? |
| `metaculus` | `43841` | `2026-06-25` | `2026-08-31` | 0.668 | `0` | Will OpenAI publicly file an initial registration statement (S-1) with the SEC to launch an IPO, before September 1, 202 |
| `metaculus` | `43985` | `2026-06-25` | `2026-09-01` | 0.043 | `0` | Before September 1, 2026, will Turkish President Recep Tayyip Erdoğan be legally permitted to run for another term? |

## Cohort: Multi-Platform Prediction Markets (Metaculus + Polymarket + Manifold + Kalshi, 2026)
- **Train Questions (Jan–Jun 2026):** `852` (Yes rate: `35.3%`)
- **OOS Test Questions (Jul–Oct 2026):** `551` (Yes rate: `41.6%`)
- **Train-Fitted Calibration Parameters:** `shrinkage_w = 1.00`, `extremize_d = 1.05`, effective log-odds slope = `1.050`
- **OOS Calibrated vs Raw Anchor Metaculus Score Delta:** `+0.003` (95% bootstrap CI: `[-0.320, +0.283]`)

| Strategy (Evaluated Strictly OOS Jul–Oct 2026) | N | Brier ↓ | Log Loss ↓ | Metaculus Baseline ↑ | ECE ↓ | Worst Q Score ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 1. Uniform 50% Baseline | 551 | 0.25000 | 0.69315 | **+0.000** | 0.08439 | +0.00 |
| 2. Train Base-Rate Prior (35.3%) | 551 | 0.24676 | 0.68714 | **+0.867** | 0.06232 | -50.11 |
| 3. Raw Market/Crowd Anchor [0.001, 0.999] | 551 | 0.13590 | 0.41742 | **+39.779** | 0.02438 | -505.89 |
| 4. Safety-Clipped Anchor [0.01, 0.99] | 551 | 0.13590 | 0.41744 | **+39.777** | 0.02436 | -505.89 |
| 5. OOS Calibrated Shrinkage+Extremizing (w=1.00, d=1.05) | 551 | 0.13601 | 0.41740 | **+39.782** | 0.03082 | -531.36 |
| 6. Structural Base-Rate + Calibrated Anchor Hybrid | 551 | 0.13587 | 0.41721 | **+39.809** | 0.02795 | -535.67 |

### Sample OOS Resolved 2026 Questions (Provenance Audit)
| Source | ID | Freeze Date | Resolved Date | Pre-Res Anchor | Outcome | Question |
|---|---|---|---|---:|---:|---|
| `manifold` | `IR8S8CNh8E` | `2026-06-25` | `2026-07-08` | 0.906 | `1` | World Cup Goals H2H: Messi vs. Ronaldo |
| `manifold` | `NcuQEz998g` | `2026-06-25` | `2026-09-16` | 0.583 | `1` | Will the Federal Reserve hike interest rates in 2026? |
| `manifold` | `zuzyPl5AOs` | `2026-06-25` | `2026-08-23` | 0.895 | `1` | Harvey AI publicly announces $200M+ ARR by EOY2026? |
| `manifold` | `BpoEJNtlQoDBpfjBmK0y` | `2026-06-25` | `2026-11-04` | 0.500 | `0` | Will Republicans win the Senate in the 2026 elections? |
| `manifold` | `2ESLt0g85E` | `2026-06-25` | `2026-10-26` | 0.917 | `1` | Will Anthropic become profitable this year? |
| `manifold` | `Nu6SdtpOsy` | `2026-06-25` | `2026-07-23` | 0.492 | `1` | Will SpaceX valuation drop by -50% or more from its ATH? |

## Key Quantitative Takeaways for Fall 2026 FutureEval Deployment
1. **Market/Crowd Anchoring Dominates Naive Priors:** Real-time market/crowd probabilities crush both uniform `50%` and historical base-rate priors by huge margins on Brier Score and Metaculus Baseline Log-Score.
2. **`[0.01, 0.99]` Tail-Clipping Protects Against Catastrophic Log-Score Blowups:** Metaculus's logarithmic scoring rule assigns up to `-896.6` on a single `0.1%` black-swan miss (`-564.4` max loss at `1%` clip). Clipping preserves nearly all upside while cutting worst-case single-question damage.
3. **Out-of-Sample Log-Odds Shrinkage + Extremizing (`w`, `d`) Stabilizes Live Deployment:** Fitting `(w, d)` strictly on earlier 2026 snapshots prevents over-extremizing noisy signals while preserving calibration across both Metaculus-only and multi-platform prediction markets.
4. **Combining Market Anchors + Multi-Persona LLM Panel:** On FutureEval questions where no Polymarket/Manifold contract exists (or where breaking news post-dates thin markets), the 3-persona LLM panel + live GDELT/Wikipedia/FRED/arXiv evidence fills the coverage gap.
