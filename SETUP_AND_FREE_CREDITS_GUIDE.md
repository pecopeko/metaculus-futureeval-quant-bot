# 5-Minute Turnkey Setup & Free Metaculus Credits Guide (`$0.00` Out-of-Pocket)

This guide takes you from zero to a fully autonomous bot competing in the **Metaculus Fall 2026 FutureEval Bot Tournament** (`$50,000` USD cash pool, `2026-09-28` -> `2027-01-06`, Tournament ID `33121`) and bi-weekly **`$1,000` MiniBench** (`minibench`), running every 6 hours on **free GitHub Actions** with **zero local Mac CPU usage** and **`$0.00` out-of-pocket cost**.

---

## Step 1: Create Your Metaculus Bot Account & Copy `METACULUS_TOKEN` (2 minutes)

> **Important 2026 API Policy:** Metaculus requires an authenticated bot token (`Authorization: Token <METACULUS_TOKEN>`) for **all** `/api/` requests (both reading open tournament questions and submitting predictions).

1. Sign in to [Metaculus](https://www.metaculus.com/) and open the official FutureEval participation portal:
   - **https://www.metaculus.com/futureeval/participate/**
   - Reference docs: [Metaculus AI Benchmark Resources](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#creating-your-bot-account-and-metaculus-token)
2. Create a dedicated **Bot Account** (required under FutureEval's strict **no human-in-the-loop** rule — never submit bot forecasts from your personal human account).
3. Copy the generated **`METACULUS_TOKEN`**.

---

## Step 2: Submit Required Participation Form & Apply for Free Metaculus LLM Credits (2 minutes)

Metaculus sponsors participant LLM usage through donated **OpenRouter** credits (supporting **OpenAI, Anthropic, and Google** models):

1. Open the official **FutureEval Participation & Free Credits Form**:
   - **https://forms.gle/aQdYMq9Pisrf1v7d8**
2. Complete the required 3-question participation section (mandatory before scoring in FutureEval).
3. Apply for donated **`OPENROUTER_API_KEY`** credits on the same form:
   - **Fall 2026 Incremental Funding Model:** Selected bots receive an initial allocation (~`$100` in OpenRouter credits) for MiniBench + Fall 2026 FutureEval.
   - **Performance Grants + Open-Source Multiplier:** Above-average MiniBench performers automatically unlock top-up grants, and **open-source bots (like this MIT-licensed repo!) qualify for ~2.5× higher grant multipliers**.
   - **Zero-Key Fallback Ready Immediately:** Even while waiting for credit approval, this bot **already runs at `$0.00`** using live Polymarket Gamma + Manifold Markets semantic anchoring + Wikipedia/GDELT/FRED/arXiv evidence + calibrated log-odds shrinkage.
   - Optional extra search credits: AskNews (`ASKNEWS_CLIENT_ID` / `ASKNEWS_SECRET`) and Exa (`EXA_API_KEY`) also periodically partner with FutureEval (check the [Metaculus Discord `#build-a-forecasting-bot`](https://discord.com/invite/NJgCC2nDfh) channel for current codes).

---

## Step 3: Add Secrets to GitHub Actions (1 minute)

In your GitHub repository (`https://github.com/pecopeko/metaculus-futureeval-quant-bot`):

1. Go to **`Settings → Secrets and variables → Actions`**.
2. Under **Secrets → New repository secret**, add:
   - **`METACULUS_TOKEN`** = *(your bot token from Step 1)*
   - **`OPENROUTER_API_KEY`** = *(your free Metaculus-sponsored OpenRouter key once approved — optional; you can also provide `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`)*
3. Under **Variables → New repository variable**, optionally configure:
   - **`METACULUS_TOURNAMENT`** = `33121` *(Fall 2026 FutureEval; use `bot-testing-area` for your first smoke test, or `minibench` for bi-weekly `$1,000` rounds!)*
   - **`FORECAST_MODEL`** = `anthropic/claude-sonnet-4.5` *(or `openai/gpt-4o` / `google/gemini-2.5-pro`; note Google donated credits share a ~150 RPM global pool, which this bot handles automatically via exponential HTTP 429 backoff)*
   - **`SUBMIT_FORECASTS`** = `false` initially, then flip to **`true`** once your test run looks clean!

---

## Step 4: Verify End-to-End on `bot-testing-area` Before Going Live

1. Open the **`Actions`** tab -> select **`Metaculus FutureEval Quant Bot`** -> click **`Run workflow`**:
   - `tournament`: `bot-testing-area`
   - `submit`: `true`
   - `max_questions`: `5`
2. Wait ~2 minutes for the green checkmark.
3. Switch to your **Bot Account** on [metaculus.com/tournament/bot-testing-area/](https://www.metaculus.com/tournament/bot-testing-area/) and confirm your bot's forecasts + private reasoning comments landed cleanly.
4. Flip repository variable **`SUBMIT_FORECASTS`** to **`true`** and **`METACULUS_TOURNAMENT`** to **`33121`** (Fall 2026 FutureEval).

From that moment on, the GitHub Actions cron (`17 */6 * * *`) wakes up every 6 hours automatically, skips already-forecasted questions, fetches live prediction-market anchors + public evidence, runs the calibrated quantitative forecaster, validates every payload locally, and submits both the forecast and private reasoning comment with **zero human intervention** and **`$0.00` cloud spend**!

---

## Local Verification Commands (Optional)

```bash
pip install -e ".[dev]"
pytest -v
metaculus-quant-bot check-setup
metaculus-quant-bot forecast --sample --dry-run
metaculus-quant-bot backtest --output-dir .
```

---

## Tournament Compliance Checklist
- [x] **No Human-in-the-Loop:** Fully autonomous scheduled execution with deterministic code & prompt templates; zero manual forecast overrides.
- [x] **Private Reasoning Comments (`is_private=True`):** Posts structured audit-trail rationale privately on each forecasted post without leaking signals to rival bots while markets are open.
- [x] **Exact API Payload Validation:** Verifies `[0.01, 0.99]` binary safety clipping, exact `1.0` multiple-choice simplex sums, and 201-point strictly monotonic standardized numeric CDFs (`min_step >= 5e-5`, `max_pmf <= 0.2`) before every POST.
- [x] **Strict `$0.00` Rule:** Zero paid infrastructure, zero paid search keys required, runs on GitHub Actions free tier.
