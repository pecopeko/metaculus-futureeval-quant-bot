"""Command-line interface for Metaculus FutureEval Quant Bot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from .backtest import run_empirical_2026_backtest
from .forecaster import HybridQuantForecaster, PluggableLLMRouter
from .metaculus_client import (
    BOT_TESTING_AREA_SLUG,
    FALL_2026_FUTUREEVAL_ID,
    MINIBENCH_SLUG,
    MetaculusAuthError,
    MetaculusClient,
)
from .models import MetaculusQuestion, NumericScaling


def get_sample_dry_run_questions() -> list[MetaculusQuestion]:
    """Representative Fall 2026 FutureEval binary, multiple-choice, and numeric sample questions."""
    return [
        MetaculusQuestion(
            question_id=990001,
            post_id=990001,
            title="Will there be a US recession by the end of 2026?",
            description="Tracks whether NBER or two consecutive negative real US GDP quarters confirm a US recession in 2026.",
            resolution_criteria="Resolves Yes if official NBER/BEA criteria confirm a US recession before 2027-01-01.",
            question_type="binary",
            scheduled_close_time="2026-12-28T23:59:00Z",
            scheduled_resolve_time="2027-01-31T00:00:00Z",
        ),
        MetaculusQuestion(
            question_id=990002,
            post_id=990002,
            title="Which lab will hold #1 on Chatbot Arena / LMSYS overall leaderboard on December 31, 2026?",
            description="Compares frontier model labs on public arena rankings at year-end 2026.",
            resolution_criteria="Resolves to the organization owning the highest-rated model on Dec 31, 2026.",
            question_type="multiple_choice",
            options=["Google", "OpenAI", "Anthropic", "xAI", "Other"],
            scheduled_close_time="2026-12-28T23:59:00Z",
            scheduled_resolve_time="2027-01-05T00:00:00Z",
        ),
        MetaculusQuestion(
            question_id=990003,
            post_id=990003,
            title="What will the US effective Federal Funds Rate (FRED: FEDFUNDS) be for December 2026?",
            description="Continuous numeric forecast anchored on official Federal Reserve H.15 / FRED FEDFUNDS release.",
            resolution_criteria="Resolves to the December 2026 monthly FEDFUNDS value reported by FRED.",
            question_type="numeric",
            open_lower_bound=True,
            open_upper_bound=True,
            unit="%",
            scaling=NumericScaling(range_min=1.0, range_max=7.0, zero_point=None),
            scheduled_close_time="2026-12-28T23:59:00Z",
            scheduled_resolve_time="2027-01-10T00:00:00Z",
        ),
    ]


def cmd_check_setup() -> int:
    """Check credentials, LLM provider detection, and live Metaculus API probe."""
    mc = MetaculusClient()
    llm = PluggableLLMRouter()
    probe = mc.check_connectivity()
    status = {
        "metaculus_token_configured": mc.has_token,
        "metaculus_api_probe": probe,
        "llm_provider": llm.provider,
        "llm_model": llm.model or "zero_key_statistical_anchor_fallback",
        "submit_forecasts_env": os.getenv("SUBMIT_FORECASTS", "false"),
        "default_tournament": os.getenv("METACULUS_TOURNAMENT", str(FALL_2026_FUTUREEVAL_ID)),
        "free_credits_application_url": "https://forms.gle/aQdYMq9Pisrf1v7d8",
        "bot_token_setup_url": "https://www.metaculus.com/futureeval/participate/",
    }
    print(json.dumps(status, indent=2))
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    """Run forecasting loop against a live Metaculus tournament or local sample dry-run."""
    mc = MetaculusClient()
    forecaster = HybridQuantForecaster()

    tournament: int | str = args.tournament
    if isinstance(tournament, str) and tournament.isdigit():
        tournament = int(tournament)

    submit_live = bool(args.submit) and not bool(args.dry_run)
    if os.getenv("SUBMIT_FORECASTS", "").strip().lower() == "true" and not args.dry_run:
        submit_live = True

    questions: list[MetaculusQuestion] = []
    used_sample_mode = False

    if args.sample:
        questions = get_sample_dry_run_questions()
        used_sample_mode = True
        submit_live = False
    else:
        if not mc.has_token:
            msg = (
                "METACULUS_TOKEN is not set. Metaculus requires authentication for /api/posts/ as of 2026. "
                "Running safe dry-run on sample Fall 2026 questions instead (zero writes)."
            )
            print(f"[INFO] {msg}", file=sys.stderr)
            if submit_live:
                print("[ERROR] Refusing --submit without METACULUS_TOKEN.", file=sys.stderr)
                return 2
            questions = get_sample_dry_run_questions()
            used_sample_mode = True
        else:
            try:
                questions = mc.list_open_tournament_questions(
                    tournament_id=tournament,
                    limit=args.max_questions,
                    skip_already_forecasted=not args.include_forecasted,
                )
            except MetaculusAuthError as exc:
                print(f"[AUTH ERROR] {exc}", file=sys.stderr)
                return 2

    results_summary = []
    for q in questions[: args.max_questions]:
        forecast = forecaster.forecast(q, enable_network=not args.offline)
        receipt = mc.submit_forecast(
            question=q,
            forecast=forecast,
            dry_run=(not submit_live) or used_sample_mode,
            post_comment=True,
            private_comment=True,
        )
        results_summary.append(
            {
                "question_id": q.question_id,
                "post_id": q.post_id,
                "type": q.question_type,
                "title": q.title,
                "mode": forecast.mode,
                "binary_probability": forecast.binary_probability,
                "multiple_choice_probabilities": forecast.multiple_choice_probabilities,
                "cdf_points": len(forecast.continuous_cdf) if forecast.continuous_cdf else None,
                "market_anchors_matched": len(forecast.market_anchors),
                "evidence_status": forecast.evidence_status,
                "dry_run": receipt.dry_run,
                "submitted": receipt.submitted,
                "validation_errors": receipt.validation_errors,
                "message": receipt.message,
            }
        )

    output = {
        "tournament": tournament,
        "sample_mode": used_sample_mode,
        "submit_live": submit_live and not used_sample_mode,
        "processed_count": len(results_summary),
        "forecasts": results_summary,
    }
    print(json.dumps(output, indent=2))
    has_validation_errors = any(r["validation_errors"] for r in results_summary)
    return 1 if has_validation_errors else 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run real empirical 2026 OOS backtest and write report + JSON."""
    out_dir = Path(args.output_dir).resolve() if args.output_dir else None
    report = run_empirical_2026_backtest(output_dir=out_dir)
    compact = {
        "generated_at": report["generated_at"],
        "total_unique_records": report["total_unique_records"],
        "snapshots_loaded": len(report["provenance"]["snapshots_loaded"]),
        "cohorts": [
            {
                "cohort": c["cohort"],
                "train_n": c["train_n"],
                "test_n": c["test_n"],
                "oos_fit": c["oos_fit"],
                "delta_vs_raw": c["calibrated_vs_raw_metaculus_delta"],
                "oos_metrics": c["oos_metrics"],
            }
            for c in report["cohorts"]
        ],
    }
    print(json.dumps(compact, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metaculus-quant-bot",
        description="Metaculus Fall 2026 FutureEval Hybrid Quant + Market-Anchor Forecasting Bot",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-setup", help="Inspect tokens, LLM provider detection, and Metaculus API status")

    p_fc = sub.add_parser("forecast", help="Run forecasting pipeline (dry-run by default)")
    p_fc.add_argument(
        "--tournament",
        default=os.getenv("METACULUS_TOURNAMENT", str(FALL_2026_FUTUREEVAL_ID)),
        help=f"Tournament ID or slug (default: {FALL_2026_FUTUREEVAL_ID}, or '{MINIBENCH_SLUG}', '{BOT_TESTING_AREA_SLUG}')",
    )
    p_fc.add_argument("--max-questions", type=int, default=15, help="Max questions per run")
    p_fc.add_argument("--dry-run", action="store_true", help="Force dry-run (never POST to Metaculus)")
    p_fc.add_argument("--submit", action="store_true", help="Actually POST forecasts + private comments")
    p_fc.add_argument("--include-forecasted", action="store_true", help="Re-forecast already forecasted questions")
    p_fc.add_argument("--sample", action="store_true", help="Run dry-run against 3 representative Fall 2026 questions")
    p_fc.add_argument("--offline", action="store_true", help="Disable external Polymarket/Manifold/evidence HTTP fetches")

    p_bt = sub.add_parser("backtest", help="Run real empirical 2026 out-of-sample backtest")
    p_bt.add_argument(
        "--output-dir",
        default=None,
        help="Directory where BACKTEST_REPORT.md and data/backtest_2026_results.json will be written",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check-setup":
        return cmd_check_setup()
    if args.command == "forecast":
        return cmd_forecast(args)
    if args.command == "backtest":
        return cmd_backtest(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
