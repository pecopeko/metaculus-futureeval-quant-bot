# Metaculus FutureEval Quant Bot (`$50,000` Fall 2026 AI Forecasting Tournament)

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Cost: $0.00](https://img.shields.io/badge/infrastructure-%240.00%20GitHub%20Actions-brightgreen.svg)](.github/workflows/metaculus_futureeval_bot.yml)

An autonomous, **no-human-in-the-loop** quantitative forecasting bot engineered for the **Metaculus Fall 2026 FutureEval Bot Tournament** (`$50,000` USD prize pool, `2026-09-28` → `2027-01-06`, Tournament ID `33121`), bi-weekly **`$1,000` MiniBench** (`minibench`), and **Winter 2027** seasons.

Instead of naive single-prompt LLM guessing, this bot combines **live cross-venue prediction market anchoring**, **zero-API-key structured evidence retrieval**, **multi-persona log-odds pooling**, **empirically backtested calibration + extremizing**, and **strict Metaculus API payload validation** — all running on **free GitHub Actions cron** with **`$0.00` out-of-pocket spend**.

---

## Architecture

```text
Metaculus API v2 (GET /api/posts/?tournaments=33121)
        │
        ├──▶ 1. Live Market Anchor Matcher (Zero API Key)
        │       • Polymarket Gamma API (real USDC liquidity/volume)
        │       • Manifold Markets Public API (play-money down-weighted 0.65×)
        │       • Entity canonicalization + numeric threshold guard +
        │         3-way mutual-exclusion rejection + polarity inversion (1-p) +
        │         target-year/close-date alignment gating
        │
        ├──▶ 2. Live Public Evidence Collector (Zero API Key, Polite Rate Limit)
        │       • Wikipedia Search & REST Summary
        │       • GDELT 2.0 Global News Doc API
        │       • FRED Public Macro CSV (FEDFUNDS, CPIAUCSL, UNRATE, DGS10, GDP, SP500)
        │       • arXiv Atom API (AI/ML/science/biotech/climate)
        │       • Hacker News Algolia API (tech/AI/startups)
        │       • SEC EDGAR EFTS (corporate filings; surfaces HTTP 403 transparently)
        │
        ├──▶ 3. Quantitative Forecasting Engine
        │       • Binary: Structural outside-view base-rate prior +
        │                 liquidity-weighted log-odds market pooling +
        │                 3-persona LLM panel (OpenRouter / OpenAI / Anthropic / Gemini / Ollama)
        │                 OR $0 deterministic statistical + market-anchor fallback +
        │                 trimmed log-odds aggregation + Neyman/Baron extremizing (d) +
        │                 strict [0.01, 0.99] log-loss safety clipping
        │       • Numeric / Discrete: Exact Metaculus 201-point (or inbound+1) strictly
        │                 monotonic standardized CDF (linear & log zero_point scaling,
        │                 open/closed bounds, min step >= 5e-5, max PMF cap <= 0.2)
        │       • Multiple Choice: Dirichlet-smoothed simplex summing strictly to 1.0
        │
        └──▶ 4. Local Payload Validator + Metaculus API Submission
                • POST /api/questions/forecast/  (JSON list payload, dry-run supported)
                • POST /api/comments/create/     (is_private=True audit reasoning comment)
```

---

## Real Empirical 2026 Out-of-Sample Backtest Highlights

Evaluated strictly **out-of-sample** on **1,403 real unique resolved 2026 binary questions** fetched live from all 22 bi-weekly 2026 ForecastBench snapshots (`2026-01-04` → `2026-10-25`, zero lookahead `freeze_datetime < resolution_date`). Calibration parameters `(shrinkage_w, extremize_d)` were fitted **strictly on Jan–Jun 2026** and tested **strictly OOS on Jul–Oct 2026** (see full provenance and question samples in [`BACKTEST_REPORT.md`](BACKTEST_REPORT.md)):

### Cohort A: Real 2026 Resolved Metaculus Binary Questions (`N_train = 108`, `N_test = 52` OOS)

| Strategy (Strictly OOS Jul–Oct 2026) | Brier ↓ | Log Loss ↓ | Metaculus Baseline ↑ | Worst Q Score ↑ |
|---|---:|---:|---:|---:|
| 1. Uniform 50% Baseline | 0.25000 | 0.69315 | `+0.000` | `+0.00` |
| 2. Train Base-Rate Prior (31.5%) | 0.19882 | 0.58745 | `+15.249` | `-66.74` |
| 3. Raw Crowd/Market Anchor `[0.001, 0.999]` | 0.17797 | 0.51844 | `+25.205` | `-264.39` |
| 4. Safety-Clipped Anchor `[0.01, 0.99]` | 0.17797 | 0.51861 | `+25.180` | `-264.39` |
| **5. OOS Calibrated Shrinkage+Extremizing (`w=0.84, d=0.93`)** | **0.16693** | **0.49649** | **`+28.372`** (**`+3.17` vs raw**) | **`-215.93`** |
| 6. Structural Base-Rate + Calibrated Anchor Hybrid | 0.17221 | 0.50514 | `+27.123` | **`-193.93`** |

### Cohort B: Multi-Platform Prediction Markets (`Metaculus + Polymarket + Manifold + Kalshi`, `N_train = 852`, `N_test = 551` OOS)

| Strategy (Strictly OOS Jul–Oct 2026) | Brier ↓ | Log Loss ↓ | Metaculus Baseline ↑ |
|---|---:|---:|---:|
| 1. Uniform 50% Baseline | 0.25000 | 0.69315 | `+0.000` |
| 2. Train Base-Rate Prior (35.3%) | 0.24676 | 0.68714 | `+0.867` |
| 3. Raw Market/Crowd Anchor `[0.001, 0.999]` | 0.13590 | 0.41742 | `+39.779` |
| **6. Structural Base-Rate + Calibrated Anchor Hybrid** | **0.13587** | **0.41721** | **`+39.809`** |

---

## Quick Start (5 Minutes, `$0.00`)

See [`SETUP_AND_FREE_CREDITS_GUIDE.md`](SETUP_AND_FREE_CREDITS_GUIDE.md) for the exact step-by-step instructions to:
1. Create your dedicated Metaculus Bot Account and copy `METACULUS_TOKEN` at `https://www.metaculus.com/futureeval/participate/`.
2. Submit the required participation form and apply for **free Metaculus-sponsored OpenRouter LLM credits** (`https://forms.gle/aQdYMq9Pisrf1v7d8`; open-source MIT bots qualify for ~2.5× MiniBench performance grant multipliers).
3. Add `METACULUS_TOKEN` (and optional `OPENROUTER_API_KEY`) to GitHub Actions Secrets and enable [`.github/workflows/metaculus_futureeval_bot.yml`](.github/workflows/metaculus_futureeval_bot.yml).

### Local CLI Commands

```bash
pip install -e ".[dev]"
pytest -v

# Check credentials, LLM routing, and live Metaculus API probe
metaculus-quant-bot check-setup

# Safe zero-key dry-run against sample Fall 2026 binary, MC, and numeric questions
metaculus-quant-bot forecast --sample --dry-run

# Re-run live empirical 2026 OOS backtest from ForecastBench
metaculus-quant-bot backtest --output-dir .

# Live tournament dry-run (requires METACULUS_TOKEN)
metaculus-quant-bot forecast --tournament bot-testing-area --dry-run

# Live tournament submission (requires METACULUS_TOKEN)
metaculus-quant-bot forecast --tournament 33121 --submit
```

---

## Verified Nov 2026 Metaculus API Specifications
- **Authentication:** `Authorization: Token <METACULUS_TOKEN>` (Metaculus enforces authentication on all `/api/` endpoints, including read-only `GET /api/posts/`, as of 2026).
- **Target Tournaments:** Fall 2026 FutureEval (`33121` / `fall-futureeval-2026`), MiniBench (`minibench`), Sandbox (`bot-testing-area`).
- **Submission Endpoint:** `POST https://www.metaculus.com/api/questions/forecast/` accepting a JSON array of forecast objects (`question`, `source: "api"`, `probability_yes`, `probability_yes_per_category`, `continuous_cdf`).
- **Private Reasoning Comment:** `POST https://www.metaculus.com/api/comments/create/` with `included_forecast: true` and `is_private: true`.

## License
MIT License — open-source friendly and eligible for Metaculus FutureEval open-source bot grant multipliers.
