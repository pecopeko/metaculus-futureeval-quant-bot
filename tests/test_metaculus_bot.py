"""Comprehensive unit & integration tests for metaculus-futureeval-quant-bot."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

from metaculus_quant_bot.backtest import (
    brier_score,
    evaluate_predictions,
    metaculus_binary_baseline_scores,
    transform_calibrated_log_odds,
)
from metaculus_quant_bot.forecaster import (
    HybridQuantForecaster,
    build_metaculus_cdf,
    clip_prob,
    extremize_log_odds,
    normalize_multiple_choice_probs,
    pool_binary_signals,
    trimmed_log_odds_mean,
)
from metaculus_quant_bot.market_anchor import (
    detect_polarity_inversion,
    has_mutually_exclusive_action_conflict,
    score_candidate_match,
    score_date_alignment,
)
from metaculus_quant_bot.metaculus_client import (
    MetaculusAuthError,
    MetaculusClient,
    create_forecast_payload,
    parse_post_to_questions,
    validate_forecast_payload,
)
from metaculus_quant_bot.models import MarketAnchorMatch, MetaculusQuestion, NumericScaling

PKG_ROOT = Path(__file__).resolve().parents[1]


def test_trimmed_log_odds_and_extremizing_calibration() -> None:
    # Outlier-resistant trimmed log-odds mean ignores single hallucinated extreme (0.99)
    probs = [0.60, 0.62, 0.65, 0.63, 0.99]
    trimmed = trimmed_log_odds_mean(probs, trim_fraction=0.20)
    assert 0.61 <= trimmed <= 0.65

    # Extremizing with d > 1 pushes away from 0.50 prior; d < 1 shrinks toward prior
    p_ext = extremize_log_odds(0.70, d=1.25, prior=0.50)
    p_shr = extremize_log_odds(0.70, d=0.75, prior=0.50)
    p_id = extremize_log_odds(0.70, d=1.00, prior=0.50)
    assert p_shr < p_id < p_ext
    assert math.isclose(p_id, 0.70, abs_tol=1e-6)

    # Strict [0.01, 0.99] safety clipping prevents catastrophic log-loss blowups
    assert extremize_log_odds(0.9999, d=2.0, prior=0.50) == 0.99
    assert extremize_log_odds(0.0001, d=2.0, prior=0.50) == 0.01
    assert clip_prob(float("nan")) == 0.5


def test_binary_log_odds_pooling_market_and_llm() -> None:
    pm_anchor = MarketAnchorMatch(
        platform="polymarket",
        market_id="pm-1",
        question="US recession by end of 2026?",
        url="https://polymarket.com/event/us-recession-2026",
        raw_probability=0.10,
        aligned_probability=0.10,
        polarity_inverted=False,
        similarity_score=0.95,
        liquidity_usd=80_000.0,
        volume_usd=400_000.0,
        date_alignment_score=1.0,
        quality_weight=0.96,
    )
    mf_anchor = MarketAnchorMatch(
        platform="manifold",
        market_id="mf-1",
        question="US recession in 2026?",
        url="https://manifold.markets/market/us-recession-2026",
        raw_probability=0.15,
        aligned_probability=0.15,
        polarity_inverted=False,
        similarity_score=0.90,
        liquidity_usd=500.0,
        volume_usd=2_000.0,
        date_alignment_score=1.0,
        quality_weight=0.55,
    )

    p_market_only, diag1 = pool_binary_signals(
        base_rate_prior=0.35,
        market_anchors=[pm_anchor, mf_anchor],
        llm_probabilities=[],
        extremize_d=1.05,
    )
    assert 0.05 < p_market_only < 0.25
    assert diag1["weights"]["market"] > diag1["weights"]["prior"]

    p_hybrid, diag2 = pool_binary_signals(
        base_rate_prior=0.35,
        market_anchors=[pm_anchor],
        llm_probabilities=[0.18, 0.20, 0.22],
        extremize_d=1.10,
    )
    assert 0.01 <= p_hybrid <= 0.99
    assert diag2["llm_trimmed_prob"] is not None
    assert diag2["market_pooled_prob"] is not None


@pytest.mark.parametrize(
    ("open_lo", "open_hi", "zero_point", "cdf_size"),
    [
        (False, False, None, 201),
        (True, True, None, 201),
        (True, False, None, 201),
        (False, True, 0.0, 201),
        (False, False, None, 21),  # discrete question with 20 inbound outcomes
    ],
)
def test_numeric_and_discrete_cdf_strict_metaculus_compliance(
    open_lo: bool,
    open_hi: bool,
    zero_point: float | None,
    cdf_size: int,
) -> None:
    qtype = "discrete" if cdf_size != 201 else "numeric"
    q = MetaculusQuestion(
        question_id=12345,
        post_id=12345,
        title="Test numeric question",
        question_type=qtype,
        open_lower_bound=open_lo,
        open_upper_bound=open_hi,
        scaling=NumericScaling(
            range_min=10.0,
            range_max=100.0,
            zero_point=zero_point,
            inbound_outcome_count=(cdf_size - 1) if qtype == "discrete" else None,
        ),
    )
    # Intentionally sharp percentiles to stress-test PMF cap and min-step enforcement
    cdf = build_metaculus_cdf(
        percentiles={10: 48.0, 25: 49.5, 50: 50.0, 75: 50.5, 90: 52.0},
        range_min=10.0,
        range_max=100.0,
        open_lower_bound=open_lo,
        open_upper_bound=open_hi,
        zero_point=zero_point,
        cdf_size=cdf_size,
    )
    assert len(cdf) == cdf_size
    payload = create_forecast_payload(qtype, continuous_cdf=cdf)
    errors = validate_forecast_payload(q, payload)
    assert errors == [], f"Unexpected CDF validation errors: {errors}"

    # Verify strict monotonicity & step bounds directly
    diffs = np.diff(cdf)
    assert float(np.min(diffs)) >= 5e-5 - 1e-12
    assert float(np.max(diffs)) <= 0.2 * (200.0 / (cdf_size - 1)) + 1e-6


def test_multiple_choice_dirichlet_simplex_and_validation() -> None:
    options = ["Google", "OpenAI", "Anthropic", "xAI", "Other"]
    probs = normalize_multiple_choice_probs(
        {"Google": 0.99, "OpenAI": 0.0, "Anthropic": 0.01, "xAI": 0.0, "Other": 0.0},
        options=options,
        dirichlet_alpha=0.35,
    )
    assert set(probs.keys()) == set(options)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert all(v >= 0.005 for v in probs.values())
    assert probs["Google"] > probs["Anthropic"] > probs["xAI"]

    q = MetaculusQuestion(
        question_id=555,
        post_id=555,
        title="Leading AI lab",
        question_type="multiple_choice",
        options=options,
    )
    payload = create_forecast_payload("multiple_choice", multiple_choice_probabilities=probs)
    assert validate_forecast_payload(q, payload) == []

    bad_payload = create_forecast_payload(
        "multiple_choice",
        multiple_choice_probabilities={"Google": 0.6, "OpenAI": 0.6},
    )
    bad_errors = validate_forecast_payload(q, bad_payload)
    assert len(bad_errors) >= 1


def test_market_anchor_polarity_threshold_date_and_entity_guards() -> None:
    # 1. Clean negation/complement polarity inversion
    inv, _ = detect_polarity_inversion(
        "Will Bitcoin close above $100k in 2026?",
        "Will Bitcoin close below $100k in 2026?",
    )
    assert inv is True

    # 2. Mutually exclusive 3-way actions must REJECT instead of flipping 1 - p
    assert has_mutually_exclusive_action_conflict(
        "Will the Fed decrease interest rates by 25 bps at the December 2026 meeting?",
        "Will the Fed increase interest rates by 25 bps at the December 2026 meeting?",
    )
    assert has_mutually_exclusive_action_conflict(
        "Will the Fed decrease interest rates by 25 bps at the December 2026 meeting?",
        "Will there be no change in Fed interest rates after the December 2026 meeting?",
    )

    q_fed = MetaculusQuestion(
        question_id=1,
        post_id=1,
        title="Will the US Federal Reserve decrease interest rates by 25 bps at the December 2026 meeting?",
        question_type="binary",
        scheduled_resolve_time="2026-12-31T23:59:00Z",
    )
    # Exact canonical match ("Federal Reserve" <-> "Fed") passes
    ok_match = score_candidate_match(
        question=q_fed,
        platform="polymarket",
        market_id="m1",
        market_question="Will the Fed decrease interest rates by 25 bps after the December 2026 meeting?",
        market_url="https://polymarket.com/event/fed-dec-2026",
        raw_probability=0.05,
        liquidity_usd=500_000.0,
        volume_usd=1_000_000.0,
        close_date="2026-12-18",
    )
    assert ok_match is not None
    assert ok_match.polarity_inverted is False
    assert ok_match.aligned_probability == 0.05

    # Mutually exclusive hike rejected
    assert (
        score_candidate_match(
            question=q_fed,
            platform="polymarket",
            market_id="m2",
            market_question="Will the Fed increase interest rates by 25 bps after the December 2026 meeting?",
            market_url="https://polymarket.com/event/fed-dec-2026",
            raw_probability=0.05,
            liquidity_usd=500_000.0,
            volume_usd=1_000_000.0,
            close_date="2026-12-18",
        )
        is None
    )

    # Wrong central bank rejected by entity guard
    assert (
        score_candidate_match(
            question=q_fed,
            platform="polymarket",
            market_id="m3",
            market_question="Bank of Mexico decreases interest rates by 25 bps after December 2026 meeting?",
            market_url="https://polymarket.com/event/banxico",
            raw_probability=0.30,
            liquidity_usd=50_000.0,
            volume_usd=100_000.0,
            close_date="2026-12-18",
        )
        is None
    )

    # Different threshold ($100k vs $150k) rejected
    q_btc = MetaculusQuestion(
        question_id=2,
        post_id=2,
        title="Will Bitcoin reach $100k before the end of 2026?",
        question_type="binary",
        scheduled_resolve_time="2026-12-31T23:59:00Z",
    )
    assert (
        score_candidate_match(
            question=q_btc,
            platform="polymarket",
            market_id="m4",
            market_question="Will Bitcoin reach $150k before the end of 2026?",
            market_url="https://polymarket.com/event/btc-150k",
            raw_probability=0.05,
            liquidity_usd=100_000.0,
            volume_usd=200_000.0,
            close_date="2026-12-31",
        )
        is None
    )

    # Target year mismatch (2026 vs 2028) heavily penalized
    date_score, _ = score_date_alignment(
        "Will a Democrat win in 2026?",
        "2026-11-15",
        "Will a Democrat win in 2028?",
        "2028-11-15",
    )
    assert date_score <= 0.15


def test_metaculus_client_parsing_dry_run_and_403_handling() -> None:
    sample_post = {
        "id": 44001,
        "title": "Sample Binary",
        "status": "open",
        "question": {
            "id": 44002,
            "title": "Will X happen in 2026?",
            "type": "binary",
            "status": "open",
            "description": "Background",
            "resolution_criteria": "Criteria",
            "fine_print": "",
            "my_forecasts": {"latest": None},
        },
    }
    parsed = parse_post_to_questions(sample_post)
    assert len(parsed) == 1
    assert parsed[0].question_id == 44002
    assert parsed[0].already_forecasted is False

    # Dry-run submission validates and never sends HTTP POST
    fc = HybridQuantForecaster().forecast(parsed[0], enable_network=False)
    mc = MetaculusClient(token="")
    receipt = mc.submit_forecast(parsed[0], fc, dry_run=True)
    assert receipt.dry_run is True
    assert receipt.submitted is False
    assert receipt.validation_errors == []

    # Mocked HTTP 403 raises explicit MetaculusAuthError (never silently swallowed)
    def _mock_403(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Permission Error: authenticated users only")

    mock_http = httpx.Client(transport=httpx.MockTransport(_mock_403))
    mc_mock = MetaculusClient(token="", http_client=mock_http)
    with pytest.raises(MetaculusAuthError):
        mc_mock.list_open_tournament_questions(tournament_id=33121)


def test_metaculus_scoring_rule_math() -> None:
    # Official Metaculus Binary Baseline Score: 100 * (log2(p_true) + 1)
    s50 = metaculus_binary_baseline_scores(np.array([0.5]), np.array([1.0]))[0]
    s99_hit = metaculus_binary_baseline_scores(np.array([0.99]), np.array([1.0]))[0]
    s01_miss = metaculus_binary_baseline_scores(np.array([0.99]), np.array([0.0]))[0]
    s001_miss = metaculus_binary_baseline_scores(np.array([0.999]), np.array([0.0]))[0]

    assert math.isclose(s50, 0.0, abs_tol=1e-6)
    assert math.isclose(s99_hit, 98.55, abs_tol=0.05)
    assert math.isclose(s01_miss, -564.39, abs_tol=0.05)
    assert math.isclose(s001_miss, -896.58, abs_tol=0.05)
    assert math.isclose(brier_score(np.array([0.5, 0.5]), np.array([1.0, 0.0])), 0.25)

    summary = evaluate_predictions("test", [0.8, 0.2], [1.0, 0.0])
    assert summary.brier_score == 0.04
    assert summary.metaculus_baseline_score > 65.0

    cal = transform_calibrated_log_odds(np.array([0.8, 0.2]), np.array([0.5, 0.5]), 0.85, 0.95)
    assert 0.5 < cal[0] < 0.8
    assert 0.2 < cal[1] < 0.5


def test_cli_offline_sample_dry_run() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "metaculus_quant_bot.cli", "forecast", "--sample", "--dry-run", "--offline"],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": str(PKG_ROOT / "src")},
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["sample_mode"] is True
    assert data["submit_live"] is False
    assert data["processed_count"] == 3
    for row in data["forecasts"]:
        assert row["dry_run"] is True
        assert row["submitted"] is False
        assert row["validation_errors"] == []


def test_openrouter_free_fallback_and_reasoning_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    from metaculus_quant_bot.forecaster import OPENROUTER_FREE_ENSEMBLE, PluggableLLMRouter

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.delenv("FORECAST_MODEL", raising=False)

    calls: list[str] = []

    def _mock_or(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        model = payload["model"]
        calls.append(model)
        assert payload["max_tokens"] >= 1800
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited", "code": 429}})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning": '{"probability": 0.31, "rationale": "Recovered from reasoning field"}',
                        }
                    }
                ]
            },
        )

    router = PluggableLLMRouter(http_client=httpx.Client(transport=httpx.MockTransport(_mock_or)))
    text, used_model = router.complete_with_model("sys", "user")
    assert used_model == OPENROUTER_FREE_ENSEMBLE[1]
    assert calls == [OPENROUTER_FREE_ENSEMBLE[0], OPENROUTER_FREE_ENSEMBLE[1]]
    assert '"probability": 0.31' in text


def test_google_news_rss_and_kalshi_and_group_questions() -> None:
    from metaculus_quant_bot.evidence_retriever import EvidenceRetriever
    from metaculus_quant_bot.market_anchor import MarketAnchorMatcher

    rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>Senegal lawmakers elect new speaker - Reuters</title>
        <link>https://example.com/news1</link>
        <pubDate>Tue, 26 May 2026 07:00:00 GMT</pubDate>
        <source>Reuters</source>
        <description>Parliamentary update</description>
      </item>
    </channel></rss>"""

    def _mock_rss(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=rss_xml)

    ev = EvidenceRetriever(
        min_request_interval_s=0.0,
        http_client=httpx.Client(transport=httpx.MockTransport(_mock_rss)),
    )
    items, status = ev.fetch_google_news_rss("Senegal National Assembly")
    assert status.startswith("ok")
    assert len(items) == 1
    assert items[0].source == "google_news"
    assert "Senegal lawmakers" in items[0].title

    # Reject unsafe DOCTYPE/ENTITY XML
    unsafe_ev = EvidenceRetriever(
        min_request_interval_s=0.0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, text="<!DOCTYPE foo><rss/>"))
        ),
    )
    bad_items, bad_status = unsafe_ev.fetch_google_news_rss("Senegal")
    assert bad_items == []
    assert "rejected_unsafe_xml" in bad_status

    # Kalshi binary contract parsing & matching
    kalshi_payload = {
        "events": [
            {
                "series_ticker": "KXFEDDEC26",
                "title": "Fed December 2026 rate decision",
                "markets": [
                    {
                        "ticker": "KXFEDDEC26-CUT25",
                        "market_type": "binary",
                        "status": "active",
                        "title": "Will the Fed decrease interest rates by 25 bps at the December 2026 meeting?",
                        "yes_bid_dollars": "0.2200",
                        "yes_ask_dollars": "0.2600",
                        "open_interest_fp": "45000.00",
                        "volume_fp": "120000.00",
                        "expected_expiration_time": "2026-12-18T19:00:00Z",
                    }
                ],
            }
        ]
    }
    matcher = MarketAnchorMatcher(
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json=kalshi_payload))
        )
    )
    cands, k_stat = matcher.search_kalshi("fed decrease interest rate")
    assert k_stat.startswith("ok")
    assert len(cands) == 1
    assert math.isclose(cands[0]["raw_probability"], 0.24, abs_tol=1e-6)

    # Group of questions unpacking
    grp_post = {
        "id": 46015,
        "title": "What will be the maximum intraday value of the VIX?",
        "status": "open",
        "group_of_questions": {
            "description": "Group description text",
            "resolution_criteria": "Group criteria text",
            "questions": [
                {
                    "id": 46246,
                    "title": "What will be the maximum intraday value of the VIX?",
                    "label": "Dec 7 - Dec 18",
                    "type": "numeric",
                    "status": "open",
                    "open_lower_bound": None,
                    "open_upper_bound": None,
                    "scaling": {
                        "range_min": 10.0,
                        "range_max": 50.0,
                        "open_lower_bound": True,
                        "open_upper_bound": True,
                        "inbound_outcome_count": 200,
                    },
                    "my_forecasts": {"latest": None},
                }
            ],
        },
    }
    g_qs = parse_post_to_questions(grp_post)
    assert len(g_qs) == 1
    assert "Dec 7 - Dec 18" in g_qs[0].title
    assert g_qs[0].open_lower_bound is True
    assert g_qs[0].open_upper_bound is True
    assert g_qs[0].description == "Group description text"
    assert g_qs[0].resolution_criteria == "Group criteria text"
