"""Hybrid Quantitative + Market-Anchor + Multi-Persona Forecasting Engine.

Supports:
1. Binary questions:
   - Structural outside-view base-rate prior
   - Liquidity- & alignment-weighted Polymarket/Manifold log-odds anchor pooling
   - Trimmed log-odds aggregation across a 3-persona LLM panel (or $0 deterministic fallback)
   - Neyman / Baron et al. (2014) log-odds extremizing (`d` calibration factor)
   - Strict [0.01, 0.99] log-loss safety clipping
2. Numeric & Discrete questions:
   - Exact Metaculus-compliant 201-point (or `inbound_outcome_count + 1`) strictly monotonic CDF
   - Linear & log (`zero_point`) scaling, open/closed tail mass, min step >= 5e-5, max PMF cap
3. Multiple-Choice questions:
   - Dirichlet-smoothed probability simplex summing strictly to 1.0 with log-loss floor protection
4. Pluggable multi-provider LLM router (OpenRouter / OpenAI / Anthropic / Gemini / Ollama)
   via pure `httpx` with HTTP 429 exponential backoff (handles shared Google/OpenRouter limits)
   AND zero-key pure-statistical + market-anchor deterministic fallback when no LLM key is set.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import numpy as np

from .evidence_retriever import EvidenceRetriever, format_evidence_digest
from .market_anchor import MarketAnchorMatcher
from .metaculus_client import create_forecast_payload
from .models import (
    EvidenceBundle,
    ForecastResult,
    MarketAnchorMatch,
    MetaculusQuestion,
    PersonaEstimate,
)

DEFAULT_PROB_MIN = 0.01
DEFAULT_PROB_MAX = 0.99
DEFAULT_EXTREMIZE_D = 1.08

OPENROUTER_FREE_ENSEMBLE: list[str] = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "dots-studio/dots-3-note-preview:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "recursive/flint-v1.0:free",
    "particle-ai/particle-1:free",
]

PERSONA_SPECS: list[tuple[str, str]] = [
    (
        "Outside-View Base-Rate Quant",
        (
            "You are a quantitative superforecaster specializing in outside-view reference classes, "
            "base rates, time-horizon decay, and status-quo persistence. Anchor first on historical "
            "frequencies and remaining time before updating conservatively on current news."
        ),
    ),
    (
        "Inside-View Evidence & Market Arbitrageur",
        (
            "You are an analytical forecaster who dissects exact resolution criteria, recent live "
            "news/macro observations, and prediction-market pricing. Identify whether live market "
            "odds already incorporate the newest facts or lag recent developments."
        ),
    ),
    (
        "Red-Team Tail-Risk Auditor",
        (
            "You are an adversarial risk-control forecaster whose job is to falsify the consensus "
            "narrative, check resolution fine-print traps, and prevent log-score blowups from "
            "overconfidence. Explicitly weigh surprising reversal scenarios."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Quantitative Probability & Calibration Primitives
# ---------------------------------------------------------------------------


def clip_prob(p: float, lo: float = DEFAULT_PROB_MIN, hi: float = DEFAULT_PROB_MAX) -> float:
    """Clip probability safely inside [lo, hi] to prevent catastrophic log-loss blowups."""
    if math.isnan(p) or math.isinf(p):
        return 0.5
    return float(min(hi, max(lo, p)))


def logit(p: float) -> float:
    """Numerically safe log-odds transform."""
    pc = clip_prob(p, 1e-6, 1.0 - 1e-6)
    return math.log(pc / (1.0 - pc))


def expit(x: float) -> float:
    """Numerically safe inverse logit (logistic sigmoid)."""
    if x >= 35.0:
        return 1.0 - 1e-15
    if x <= -35.0:
        return 1e-15
    return 1.0 / (1.0 + math.exp(-x))


def trimmed_log_odds_mean(
    probabilities: list[float],
    trim_fraction: float = 0.20,
) -> float:
    """Compute outlier-resistant trimmed mean in log-odds space.

    When >= 4 estimates are supplied, symmetrically trims the lowest and highest
    `floor(n * trim_fraction)` log-odds values before averaging, protecting against
    single-model hallucinations while preserving Bayesian log-odds pooling.
    """
    if not probabilities:
        raise ValueError("probabilities must be non-empty")
    vals = sorted(logit(clip_prob(p, 0.002, 0.998)) for p in probabilities)
    n = len(vals)
    k = int(math.floor(n * trim_fraction)) if n >= 4 else 0
    if k > 0 and (2 * k) < n:
        trimmed = vals[k : n - k]
    else:
        trimmed = vals
    return expit(sum(trimmed) / len(trimmed))


def extremize_log_odds(
    probability: float,
    d: float = DEFAULT_EXTREMIZE_D,
    prior: float = 0.50,
    lo: float = DEFAULT_PROB_MIN,
    hi: float = DEFAULT_PROB_MAX,
) -> float:
    """Neyman / Baron et al. (2014) log-odds extremizing (or shrinkage when d < 1).

    logit(p_out) = logit(prior) + d * (logit(p) - logit(prior))
    """
    lp = logit(clip_prob(probability, 1e-4, 1.0 - 1e-4))
    l0 = logit(clip_prob(prior, 0.05, 0.95))
    return clip_prob(expit(l0 + d * (lp - l0)), lo, hi)


def estimate_structural_base_rate(
    question: MetaculusQuestion,
    evidence: Optional[EvidenceBundle] = None,
) -> tuple[float, list[str]]:
    """Compute a transparent outside-view structural base-rate prior for a binary question.

    Uses resolution horizon cues, rare-event wording, status-quo framing, and FRED macro
    trends (when available) — strictly bounded in [0.18, 0.72] so the prior stays humble.
    """
    text = f"{question.title} {question.resolution_criteria}".lower()
    prior_lo = 0.0  # logit(0.50) = 0
    reasons: list[str] = ["neutral_start=0.50"]

    # Rare/extreme milestone or dramatic event cues historically resolve No > 50%
    rare_cues = [
        "invade", "war", "nuclear", "recession", "default", "impeach", "resign",
        "bankruptcy", "agi", "superintelligence", "crash", "collapse", "assassinate",
        "out before", "all-time high", "record high", "double", "triple",
    ]
    if any(cue in text for cue in rare_cues):
        prior_lo -= 0.45
        reasons.append("rare_disruptive_event_cue(-0.45_logit)")

    # Status-quo continuation cues historically resolve Yes > 50%
    status_quo_cues = ["remain", "stay", "continue", "keep", "hold", "maintain"]
    if any(cue in text for cue in status_quo_cues):
        prior_lo += 0.35
        reasons.append("status_quo_persistence_cue(+0.35_logit)")

    # Short remaining horizon reduces probability of dramatic state transitions
    if question.scheduled_resolve_time:
        try:
            iso = question.scheduled_resolve_time.replace("Z", "+00:00")
            resolve_dt = datetime.fromisoformat(iso)
            if resolve_dt.tzinfo is None:
                resolve_dt = resolve_dt.replace(tzinfo=timezone.utc)
            days_left = (resolve_dt - datetime.now(timezone.utc)).total_seconds() / 86400.0
            if 0 < days_left < 30 and prior_lo < 0:
                prior_lo -= 0.25
                reasons.append(f"short_horizon_{int(days_left)}d(-0.25_logit)")
            elif days_left > 365 and prior_lo < 0:
                prior_lo += 0.15
                reasons.append(f"long_horizon_{int(days_left)}d(+0.15_logit)")
        except ValueError:
            pass

    # Optional FRED macro threshold check for simple "above/below X%" binary questions
    if evidence is not None and evidence.macro_series:
        for sid, meta in evidence.macro_series.items():
            latest = meta.get("latest_value")
            if isinstance(latest, (int, float)):
                m_above = re.search(r"(?:above|over|exceed|higher than)\s+(\d+(?:\.\d+)?)\s*%?", text)
                m_below = re.search(r"(?:below|under|lower than|less than)\s+(\d+(?:\.\d+)?)\s*%?", text)
                if m_above:
                    thresh = float(m_above.group(1))
                    diff = latest - thresh
                    adj = max(-0.8, min(0.8, diff * 0.4))
                    prior_lo += adj
                    reasons.append(f"fred_{sid}_vs_{thresh}({adj:+.2f}_logit)")
                    break
                if m_below:
                    thresh = float(m_below.group(1))
                    diff = thresh - latest
                    adj = max(-0.8, min(0.8, diff * 0.4))
                    prior_lo += adj
                    reasons.append(f"fred_{sid}_vs_{thresh}({adj:+.2f}_logit)")
                    break

    prior_p = clip_prob(expit(prior_lo), 0.18, 0.72)
    return round(prior_p, 4), reasons


def pool_binary_signals(
    base_rate_prior: float,
    market_anchors: list[MarketAnchorMatch],
    llm_probabilities: list[float],
    extremize_d: float = DEFAULT_EXTREMIZE_D,
    prob_min: float = DEFAULT_PROB_MIN,
    prob_max: float = DEFAULT_PROB_MAX,
    market_weight_cap: float = 0.72,
    evidence_log_odds_nudge: float = 0.0,
) -> tuple[float, dict[str, Any]]:
    """Combine base-rate prior, liquidity-weighted market anchors, and trimmed LLM panel in log-odds space.

    Returns (final_clipped_probability, diagnostics_dict).
    """
    prior_p = clip_prob(base_rate_prior, 0.05, 0.95)
    prior_lo = logit(prior_p)

    # 1. Pool market anchors by quality_weight in log-odds space
    market_lo: Optional[float] = None
    market_strength = 0.0
    if market_anchors:
        total_qw = sum(max(0.01, m.quality_weight) for m in market_anchors)
        weighted_lo = sum(
            max(0.01, m.quality_weight) * logit(clip_prob(m.aligned_probability, 0.005, 0.995))
            for m in market_anchors
        ) / total_qw
        market_lo = weighted_lo
        # Strength scales smoothly with best anchor quality & number of confirming venues
        best_qw = max(m.quality_weight for m in market_anchors)
        venue_bonus = 0.08 if len({m.platform for m in market_anchors}) > 1 else 0.0
        market_strength = min(market_weight_cap, max(0.15, best_qw * 0.85 + venue_bonus))

    # 2. Trimmed log-odds mean of LLM panel
    llm_lo: Optional[float] = None
    llm_pooled_p: Optional[float] = None
    if llm_probabilities:
        llm_pooled_p = trimmed_log_odds_mean(llm_probabilities, trim_fraction=0.20)
        llm_lo = logit(llm_pooled_p)

    # 3. Adaptive weighting across available signal pillars
    bounded_nudge = max(-0.35, min(0.35, float(evidence_log_odds_nudge)))
    if market_lo is not None and llm_lo is not None:
        w_market = min(0.55, market_strength * 0.75)
        w_llm = min(0.55, 0.85 - w_market)
        w_prior = max(0.08, 1.0 - w_market - w_llm)
        norm = w_market + w_llm + w_prior
        w_market, w_llm, w_prior = w_market / norm, w_llm / norm, w_prior / norm
        combined_lo = w_market * market_lo + w_llm * llm_lo + w_prior * prior_lo
        effective_d = extremize_d
    elif market_lo is not None:
        # Zero-key market-anchor mode: shrink slightly toward structural prior
        w_market = market_strength
        w_prior = 1.0 - w_market
        w_llm = 0.0
        combined_lo = w_market * market_lo + w_prior * prior_lo + 0.5 * bounded_nudge
        # Markets are already sharp crowds; mild calibration only
        effective_d = min(extremize_d, 1.08)
    elif llm_lo is not None:
        w_market = 0.0
        w_llm = 0.84
        w_prior = 0.16
        combined_lo = w_llm * llm_lo + w_prior * prior_lo
        effective_d = extremize_d
    else:
        w_market = 0.0
        w_llm = 0.0
        w_prior = 1.0
        combined_lo = prior_lo + bounded_nudge
        effective_d = 1.0

    pre_ext_p = expit(combined_lo)
    final_p = extremize_log_odds(
        pre_ext_p,
        d=effective_d,
        prior=prior_p,
        lo=prob_min,
        hi=prob_max,
    )

    diagnostics = {
        "base_rate_prior": round(prior_p, 4),
        "market_pooled_prob": round(expit(market_lo), 4) if market_lo is not None else None,
        "llm_trimmed_prob": round(llm_pooled_p, 4) if llm_pooled_p is not None else None,
        "weights": {
            "market": round(w_market, 4),
            "llm": round(w_llm, 4),
            "prior": round(w_prior, 4),
        },
        "pre_extremize_prob": round(pre_ext_p, 4),
        "extremize_d": round(effective_d, 4),
        "final_prob": round(final_p, 6),
    }
    return round(final_p, 6), diagnostics


# ---------------------------------------------------------------------------
# Numeric & Discrete CDF Builder (Exact Metaculus API Specification)
# ---------------------------------------------------------------------------


def _nominal_to_cdf_location(
    value: float,
    range_min: float,
    range_max: float,
    zero_point: Optional[float],
) -> float:
    """Map a real-world numeric value onto Metaculus's [0, 1] scaled CDF x-axis."""
    if zero_point is None:
        span = range_max - range_min
        if span <= 0:
            return 0.5
        return (value - range_min) / span

    # Log-scaled question per official Metaculus formula
    if value <= zero_point:
        return -1e-6
    deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
    if deriv_ratio <= 0 or abs(deriv_ratio - 1.0) < 1e-12:
        return (value - range_min) / max(1e-12, range_max - range_min)
    arg = 1.0 + (value - range_min) * (deriv_ratio - 1.0) / (range_max - range_min)
    if arg <= 1e-12:
        return -1e-6
    return math.log(arg) / math.log(deriv_ratio)


def _cdf_location_to_nominal(
    loc: float,
    range_min: float,
    range_max: float,
    zero_point: Optional[float],
) -> float:
    """Inverse mapping from [0, 1] CDF location back to nominal numeric scale."""
    if zero_point is None:
        return range_min + (range_max - range_min) * loc
    deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
    return range_min + (range_max - range_min) * (deriv_ratio**loc - 1.0) / (deriv_ratio - 1.0)


def standardize_metaculus_cdf(
    raw_cdf: list[float],
    open_lower_bound: bool,
    open_upper_bound: bool,
) -> list[float]:
    """Standardize a CDF to satisfy ALL Metaculus API constraints.

    Guarantees:
    - Closed lower bound -> cdf[0] == 0.0; open lower bound -> cdf[0] >= 0.001
    - Closed upper bound -> cdf[-1] == 1.0; open upper bound -> cdf[-1] <= 0.999
    - Strictly monotonically increasing with every step >= 5e-5
    - Maximum PMF bin mass capped strictly below 0.2 * (200 / (len(cdf) - 1))
    """
    n = len(raw_cdf)
    if n < 2:
        raise ValueError("CDF must have at least 2 points")

    cdf = np.asarray(raw_cdf, dtype=np.float64)
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))

    scale_lower_to = 0.0 if open_lower_bound else float(cdf[0])
    scale_upper_to = 1.0 if open_upper_bound else float(cdf[-1])
    inbound_mass = scale_upper_to - scale_lower_to

    locations = np.linspace(0.0, 1.0, n)
    if inbound_mass <= 1e-12:
        rescaled = locations.copy()
    else:
        rescaled = np.clip((cdf - scale_lower_to) / inbound_mass, 0.0, 1.0)

    if open_lower_bound and open_upper_bound:
        cdf_std = 0.988 * rescaled + 0.01 * locations + 0.001
    elif open_lower_bound:
        cdf_std = 0.989 * rescaled + 0.01 * locations + 0.001
    elif open_upper_bound:
        cdf_std = 0.989 * rescaled + 0.01 * locations
    else:
        cdf_std = 0.990 * rescaled + 0.01 * locations

    # Operate in PMF space to cap max bin height per Metaculus NumericDefaults
    inbound_count = n - 1
    cap = 0.95 * 0.2 * (200.0 / inbound_count)
    min_step = max(5.05e-5, 0.005 / inbound_count)
    if min_step >= cap:
        min_step = 0.5 / inbound_count

    pmf = np.diff(cdf_std, prepend=0.0, append=1.0)
    lower_tail = float(pmf[0])
    upper_tail = float(pmf[-1])
    interior = np.clip(pmf[1:-1], min_step, cap)
    target_interior_sum = max(1e-6, 1.0 - lower_tail - upper_tail)

    # Exact bisection for multiplier `scale` so sum(clip(scale * interior, min_step, cap)) == target
    def _scaled_interior(scale: float) -> np.ndarray:
        return np.clip(scale * pmf[1:-1], min_step, cap)

    lo_s, hi_s = 1e-4, 1.0
    while float(_scaled_interior(hi_s).sum()) < target_interior_sum and hi_s < 1e6:
        hi_s *= 1.5
    while float(_scaled_interior(lo_s).sum()) > target_interior_sum and lo_s > 1e-9:
        lo_s *= 0.5

    for _ in range(80):
        mid = 0.5 * (lo_s + hi_s)
        s_val = float(_scaled_interior(mid).sum())
        if s_val < target_interior_sum:
            lo_s = mid
        else:
            hi_s = mid
        if abs(s_val - target_interior_sum) < 1e-10:
            break

    interior = _scaled_interior(0.5 * (lo_s + hi_s))
    # Tiny final exact residual distribution across non-saturated bins
    residual = target_interior_sum - float(interior.sum())
    if abs(residual) > 1e-12:
        headroom = (cap - interior) if residual > 0 else (interior - min_step)
        mask = headroom > 1e-9
        if np.any(mask):
            interior[mask] += residual * (headroom[mask] / headroom[mask].sum())

    pmf_final = np.concatenate([[lower_tail], interior, [upper_tail]])
    cdf_final = np.cumsum(pmf_final)[:-1]

    # Exact closed-bound endpoint anchors
    if not open_lower_bound:
        cdf_final[0] = 0.0
    else:
        cdf_final[0] = max(0.001, float(cdf_final[0]))
    if not open_upper_bound:
        cdf_final[-1] = 1.0
    else:
        cdf_final[-1] = min(0.999, float(cdf_final[-1]))

    return [round(float(x), 10) for x in cdf_final.tolist()]


def build_metaculus_cdf(
    percentiles: dict[int, float],
    range_min: float,
    range_max: float,
    open_lower_bound: bool = False,
    open_upper_bound: bool = False,
    zero_point: Optional[float] = None,
    cdf_size: int = 201,
) -> list[float]:
    """Build a strictly valid Metaculus continuous CDF from sparse percentile estimates.

    Args:
        percentiles: Mapping of integer percentile (1..99, e.g. {10: ..., 50: ..., 90: ...})
                     to nominal numeric values.
        range_min, range_max, zero_point: Metaculus question scaling fields.
        open_lower_bound, open_upper_bound: Bound openness flags.
        cdf_size: 201 for standard numeric/date questions, `inbound_outcome_count + 1` for discrete.
    """
    if not percentiles:
        raise ValueError("percentiles mapping must be non-empty")

    sorted_pts = sorted((int(k) / 100.0, float(v)) for k, v in percentiles.items())
    # Enforce monotone values across increasing percentiles
    xs_loc: list[float] = []
    ys_cdf: list[float] = []
    for prob, val in sorted_pts:
        loc = _nominal_to_cdf_location(val, range_min, range_max, zero_point)
        if xs_loc and loc <= xs_loc[-1]:
            loc = xs_loc[-1] + 1e-4
        xs_loc.append(loc)
        ys_cdf.append(max(0.001, min(0.999, prob)))

    # Smooth tail anchors outside [0, 1] using logistic/linear slope continuation
    lower_x = min(0.0, xs_loc[0] - 0.15)
    upper_x = max(1.0, xs_loc[-1] + 0.15)
    lower_y = 0.0 if not open_lower_bound else max(0.001, ys_cdf[0] * 0.25)
    upper_y = 1.0 if not open_upper_bound else min(0.999, 1.0 - (1.0 - ys_cdf[-1]) * 0.25)

    anchor_x = [lower_x] + xs_loc + [upper_x]
    anchor_y = [lower_y] + ys_cdf + [upper_y]

    # Ensure strictly non-decreasing anchor_x & anchor_y
    for i in range(1, len(anchor_x)):
        if anchor_x[i] <= anchor_x[i - 1]:
            anchor_x[i] = anchor_x[i - 1] + 1e-5
        if anchor_y[i] < anchor_y[i - 1]:
            anchor_y[i] = anchor_y[i - 1]

    eval_locs = np.linspace(0.0, 1.0, cdf_size)
    raw_cdf = np.interp(eval_locs, anchor_x, anchor_y).tolist()
    return standardize_metaculus_cdf(
        raw_cdf,
        open_lower_bound=open_lower_bound,
        open_upper_bound=open_upper_bound,
    )


def default_numeric_percentiles(
    question: MetaculusQuestion,
    evidence: Optional[EvidenceBundle] = None,
) -> dict[int, float]:
    """Construct a calibrated statistical percentile fallback for numeric/discrete questions."""
    if question.scaling is None:
        return {10: 10.0, 25: 25.0, 50: 50.0, 75: 75.0, 90: 90.0}
    rmin = question.scaling.range_min
    rmax = question.scaling.range_max
    zp = question.scaling.zero_point

    # Anchor center on latest FRED value if inside question range
    center_loc = 0.50
    spread = 0.22
    if evidence is not None and evidence.macro_series:
        first_meta = next(iter(evidence.macro_series.values()))
        latest = first_meta.get("latest_value")
        if isinstance(latest, (int, float)) and rmin <= latest <= rmax:
            center_loc = max(0.15, min(0.85, _nominal_to_cdf_location(latest, rmin, rmax, zp)))
            spread = 0.14

    offsets = {10: -1.2816, 25: -0.6745, 50: 0.0, 75: 0.6745, 90: 1.2816}
    out: dict[int, float] = {}
    for pct, z in offsets.items():
        loc = max(0.02, min(0.98, center_loc + z * spread))
        out[pct] = round(_cdf_location_to_nominal(loc, rmin, rmax, zp), 6)
    return out


# ---------------------------------------------------------------------------
# Multiple-Choice Dirichlet-Smoothed Simplex
# ---------------------------------------------------------------------------


def normalize_multiple_choice_probs(
    raw_scores: dict[str, float],
    options: list[str],
    dirichlet_alpha: float = 0.35,
    min_prob: float = 0.005,
) -> dict[str, float]:
    """Produce a Dirichlet-smoothed normalized probability simplex matching `options` exactly."""
    if not options:
        raise ValueError("options must be non-empty")
    k = len(options)
    effective_floor = min(min_prob, 0.25 / k)

    weights: list[float] = []
    raw_sum = sum(max(0.0, float(raw_scores.get(opt, 0.0))) for opt in options)
    for opt in options:
        val = max(0.0, float(raw_scores.get(opt, 0.0)))
        norm_val = (val / raw_sum) if raw_sum > 0 else (1.0 / k)
        # Dirichlet shrinkage toward uniform 1/K prevents overconfident 0% blowups
        smoothed = (norm_val + dirichlet_alpha * (1.0 / k)) / (1.0 + dirichlet_alpha)
        weights.append(max(effective_floor, smoothed))

    total = sum(weights)
    probs = [w / total for w in weights]
    rounded = [round(p, 6) for p in probs]

    # Fix floating-point rounding drift on argmax index so sum == 1.000000
    drift = round(1.0 - sum(rounded), 6)
    max_idx = max(range(k), key=lambda idx: rounded[idx])
    rounded[max_idx] = round(rounded[max_idx] + drift, 6)
    return {opt: rounded[i] for i, opt in enumerate(options)}


# ---------------------------------------------------------------------------
# Pluggable Zero-Heavy-Dependency LLM Router
# ---------------------------------------------------------------------------


class PluggableLLMRouter:
    """Pure-httpx multi-provider LLM router with automatic env detection & HTTP 429 retry."""

    def __init__(
        self,
        timeout: float = 45.0,
        max_retries: int = 3,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self._external_client = http_client
        self.provider, self.api_key, self.base_url, self.model = self._detect_provider()

    @staticmethod
    def _detect_provider() -> tuple[str, str, str, str]:
        override_model = os.getenv("FORECAST_MODEL", "").strip()

        def _real(name: str) -> str:
            val = os.getenv(name, "").strip()
            return "" if not val or val == "REPLACE_ME" else val

        if key := _real("OPENROUTER_API_KEY"):
            return (
                "openrouter",
                key,
                "https://openrouter.ai/api/v1/chat/completions",
                override_model or OPENROUTER_FREE_ENSEMBLE[0],
            )
        if key := _real("OPENAI_API_KEY"):
            return (
                "openai",
                key,
                "https://api.openai.com/v1/chat/completions",
                override_model or "gpt-4o",
            )
        if key := _real("ANTHROPIC_API_KEY"):
            return (
                "anthropic",
                key,
                "https://api.anthropic.com/v1/messages",
                override_model or "claude-sonnet-4-5",
            )
        if key := _real("GEMINI_API_KEY"):
            model = override_model or "gemini-2.5-pro"
            return (
                "gemini",
                key,
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                model,
            )
        if ollama_url := _real("OLLAMA_BASE_URL"):
            return (
                "ollama",
                "ollama",
                f"{ollama_url.rstrip('/')}/v1/chat/completions",
                override_model or "llama3.1:8b",
            )
        return ("none", "", "", "")

    @property
    def is_available(self) -> bool:
        return self.provider != "none"

    def _candidate_models_for_call(self, preferred_model: Optional[str] = None) -> list[str]:
        """Build ordered model fallback chain (enables zero-cost OpenRouter :free resilience)."""
        primary = (preferred_model or self.model).strip()
        if self.provider != "openrouter":
            return [primary]
        # If caller/user explicitly pinned a paid non-:free OpenRouter model, keep primary first then :free safety net
        chain: list[str] = []
        if primary and primary != "openrouter-free-ensemble":
            chain.append(primary)
        for m in OPENROUTER_FREE_ENSEMBLE:
            if m not in chain:
                chain.append(m)
        return chain

    def complete_with_model(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.25,
        preferred_model: Optional[str] = None,
    ) -> tuple[str, str]:
        """Send a chat completion and return `(response_text, actual_model_used)`."""
        if not self.is_available:
            raise RuntimeError("No LLM provider configured")

        candidates = self._candidate_models_for_call(preferred_model=preferred_model)
        last_err = ""

        for cand_idx, active_model in enumerate(candidates):
            if self.provider in ("openrouter", "openai", "ollama"):
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                if self.provider == "openrouter":
                    headers["HTTP-Referer"] = "https://github.com/pecopeko/metaculus-futureeval-quant-bot"
                    headers["X-Title"] = "Metaculus FutureEval Quant Bot"
                body: dict[str, Any] = {
                    "model": active_model,
                    "max_tokens": 2048,
                    "temperature": temperature,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                url = self.base_url
                params = None
            elif self.provider == "anthropic":
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                }
                body = {
                    "model": active_model,
                    "max_tokens": 1600,
                    "temperature": temperature,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                }
                url = self.base_url
                params = None
            elif self.provider == "gemini":
                headers = {"Content-Type": "application/json"}
                body = {
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"parts": [{"text": user_prompt}]}],
                    "generationConfig": {"temperature": temperature},
                }
                url = self.base_url
                params = {"key": self.api_key}
            else:
                raise RuntimeError(f"Unsupported provider: {self.provider}")

            # When multiple fallback models exist on OpenRouter, hop immediately on 429/404/5xx
            attempts_for_model = 1 if len(candidates) > 1 else self.max_retries
            for attempt in range(attempts_for_model):
                try:
                    if self._external_client is not None:
                        resp = self._external_client.post(
                            url, params=params, headers=headers, json=body, timeout=self.timeout
                        )
                    else:
                        with httpx.Client(timeout=self.timeout) as client:
                            resp = client.post(url, params=params, headers=headers, json=body)
                except httpx.HTTPError as exc:
                    last_err = f"{active_model}: {exc}"
                    if attempt + 1 < attempts_for_model:
                        time.sleep(2.0 * (2**attempt))
                        continue
                    break

                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = f"{active_model} HTTP {resp.status_code}: {resp.text[:180]}"
                    if attempt + 1 < attempts_for_model:
                        delay = min(15.0, 2.0 * (2**attempt))
                        time.sleep(delay)
                        continue
                    break

                if resp.status_code >= 400:
                    last_err = f"{active_model} HTTP {resp.status_code}: {resp.text[:220]}"
                    if cand_idx + 1 < len(candidates):
                        break
                    raise RuntimeError(f"LLM API error ({self.provider} {last_err})")

                data = resp.json()
                if self.provider in ("openrouter", "openai", "ollama"):
                    choices = data.get("choices") or []
                    if not choices:
                        last_err = f"{active_model}: empty choices"
                        break
                    msg_obj = choices[0].get("message") or {}
                    text_out = str(msg_obj.get("content") or "").strip()
                    if not text_out and msg_obj.get("reasoning"):
                        text_out = str(msg_obj.get("reasoning") or "").strip()
                    if not text_out:
                        last_err = f"{active_model}: empty content"
                        break
                    return text_out, active_model
                if self.provider == "anthropic":
                    parts = data.get("content") or []
                    return "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict)), active_model
                if self.provider == "gemini":
                    cands = data.get("candidates") or []
                    parts = ((cands[0].get("content") or {}).get("parts")) if cands else []
                    return "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict)), active_model

        raise RuntimeError(f"LLM retries exhausted ({self.provider}): {last_err}")

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.25) -> str:
        """Send a single chat completion with exponential backoff / model fallback."""
        text, _ = self.complete_with_model(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        return text


def _extract_json_block(text: str) -> Optional[dict[str, Any]]:
    """Extract first JSON object from model output (tolerates markdown fences)."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        brace = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if brace:
            cleaned = brace.group(1)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_binary_prob_from_text(text: str) -> Optional[float]:
    parsed = _extract_json_block(text)
    if parsed and "probability" in parsed:
        val = float(parsed["probability"])
        return val / 100.0 if val > 1.0 else val
    m = re.search(r"(?:probability|forecast)\s*[:=]\s*(\d+(?:\.\d+)?)\s*%?", text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val / 100.0 if val > 1.0 else val
    return None


# ---------------------------------------------------------------------------
# End-to-End Hybrid Forecaster
# ---------------------------------------------------------------------------


class HybridQuantForecaster:
    """Multi-stage quantitative + market-anchor + live-evidence forecasting engine."""

    def __init__(
        self,
        market_matcher: Optional[MarketAnchorMatcher] = None,
        evidence_retriever: Optional[EvidenceRetriever] = None,
        llm_router: Optional[PluggableLLMRouter] = None,
        extremize_d: Optional[float] = None,
        prob_min: Optional[float] = None,
        prob_max: Optional[float] = None,
    ) -> None:
        self.market_matcher = market_matcher or MarketAnchorMatcher()
        self.evidence_retriever = evidence_retriever or EvidenceRetriever()
        self.llm_router = llm_router or PluggableLLMRouter()
        self.extremize_d = (
            extremize_d
            if extremize_d is not None
            else float(os.getenv("EXTREMIZE_D", str(DEFAULT_EXTREMIZE_D)))
        )
        self.prob_min = (
            prob_min
            if prob_min is not None
            else float(os.getenv("PROB_MIN", str(DEFAULT_PROB_MIN)))
        )
        self.prob_max = (
            prob_max
            if prob_max is not None
            else float(os.getenv("PROB_MAX", str(DEFAULT_PROB_MAX)))
        )

    def _build_question_prompt(
        self,
        question: MetaculusQuestion,
        anchors: list[MarketAnchorMatch],
        evidence: EvidenceBundle,
        base_rate: Optional[float],
    ) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        anchor_lines = []
        for a in anchors:
            inv_tag = " [INVERTED->1-p]" if a.polarity_inverted else ""
            anchor_lines.append(
                f"- {a.platform.upper()} ({a.aligned_probability:.1%}{inv_tag}, "
                f"sim={a.similarity_score:.2f}, liq=${a.liquidity_usd:,.0f}): {a.question}"
            )
        anchor_block = "\n".join(anchor_lines) if anchor_lines else "No high-confidence external market anchor matched."
        evidence_block = format_evidence_digest(evidence, max_items=8)

        header = (
            f"Today (UTC): {today}\n"
            f"Question: {question.title}\n"
            f"Question Type: {question.question_type}\n"
            f"Scheduled Close: {question.scheduled_close_time or 'unspecified'}\n"
            f"Scheduled Resolve: {question.scheduled_resolve_time or 'unspecified'}\n\n"
            f"Background:\n{question.description[:1200]}\n\n"
            f"Resolution Criteria:\n{question.resolution_criteria[:1000]}\n\n"
            f"Fine Print:\n{question.fine_print[:600]}\n\n"
            f"Live Prediction Market Anchors:\n{anchor_block}\n\n"
            f"Live Public Evidence:\n{evidence_block}\n"
        )

        if question.question_type == "binary":
            br_txt = f"{base_rate:.1%}" if base_rate is not None else "50.0%"
            return (
                f"{header}\nStructural Base-Rate Prior: {br_txt}\n\n"
                'Respond ONLY with valid JSON: {"probability": <float between 0.01 and 0.99>, '
                '"rationale": "<concise 3-5 sentence quantitative justification>"}'
            )

        if question.question_type == "multiple_choice":
            opts_json = json.dumps(question.options)
            return (
                f"{header}\nExact Options: {opts_json}\n\n"
                'Respond ONLY with valid JSON: {"option_probabilities": {<exact option string>: <float>, ...}, '
                '"rationale": "<concise 3-5 sentence justification>"} where probabilities sum to 1.0.'
            )

        scale_desc = "unspecified"
        if question.scaling is not None:
            scale_desc = (
                f"min={question.scaling.range_min}, max={question.scaling.range_max}, "
                f"open_lower={question.open_lower_bound}, open_upper={question.open_upper_bound}, "
                f"unit='{question.unit}'"
            )
        return (
            f"{header}\nNumeric Scale: {scale_desc}\n\n"
            'Respond ONLY with valid JSON: {"percentiles": {"10": <num>, "25": <num>, '
            '"50": <num>, "75": <num>, "90": <num>}, '
            '"rationale": "<concise 3-5 sentence justification>"} with strictly increasing values.'
        )

    def _run_llm_panel(
        self,
        question: MetaculusQuestion,
        anchors: list[MarketAnchorMatch],
        evidence: EvidenceBundle,
        base_rate: Optional[float],
    ) -> list[PersonaEstimate]:
        if not self.llm_router.is_available:
            return []

        user_prompt = self._build_question_prompt(question, anchors, evidence, base_rate)
        estimates: list[PersonaEstimate] = []
        use_openrouter_diversity = (
            self.llm_router.provider == "openrouter"
            and (
                self.llm_router.model in OPENROUTER_FREE_ENSEMBLE
                or self.llm_router.model == "openrouter-free-ensemble"
            )
        )
        for p_idx, (persona_name, sys_prompt) in enumerate(PERSONA_SPECS):
            preferred_model: Optional[str] = None
            if use_openrouter_diversity and p_idx < len(OPENROUTER_FREE_ENSEMBLE):
                preferred_model = OPENROUTER_FREE_ENSEMBLE[p_idx]
            used_model = preferred_model or self.llm_router.model
            try:
                raw, used_model = self.llm_router.complete_with_model(
                    sys_prompt,
                    user_prompt,
                    temperature=0.25,
                    preferred_model=preferred_model,
                )
                parsed = _extract_json_block(raw) or {}
                rationale = str(parsed.get("rationale") or raw[:400]).strip()
                if question.question_type == "binary":
                    p_val = _extract_binary_prob_from_text(raw)
                    if p_val is None:
                        raise ValueError("Could not parse binary probability from LLM response")
                    estimates.append(
                        PersonaEstimate(
                            persona=persona_name,
                            provider=self.llm_router.provider,
                            model=used_model,
                            probability=clip_prob(p_val, self.prob_min, self.prob_max),
                            rationale=rationale,
                        )
                    )
                elif question.question_type == "multiple_choice":
                    opt_map_raw = parsed.get("option_probabilities")
                    if not isinstance(opt_map_raw, dict):
                        raise ValueError("Missing option_probabilities dict in LLM response")
                    opt_map = {str(k): float(v) for k, v in opt_map_raw.items()}
                    estimates.append(
                        PersonaEstimate(
                            persona=persona_name,
                            provider=self.llm_router.provider,
                            model=used_model,
                            option_probabilities=opt_map,
                            rationale=rationale,
                        )
                    )
                else:
                    pct_raw = parsed.get("percentiles")
                    if not isinstance(pct_raw, dict):
                        raise ValueError("Missing percentiles dict in LLM response")
                    pct_map = {int(k): float(v) for k, v in pct_raw.items()}
                    estimates.append(
                        PersonaEstimate(
                            persona=persona_name,
                            provider=self.llm_router.provider,
                            model=used_model,
                            percentiles=pct_map,
                            rationale=rationale,
                        )
                    )
            except Exception as exc:
                estimates.append(
                    PersonaEstimate(
                        persona=persona_name,
                        provider=self.llm_router.provider,
                        model=used_model,
                        error=str(exc),
                        rationale="",
                    )
                )
        return estimates

    def forecast(
        self,
        question: MetaculusQuestion,
        enable_network: bool = True,
    ) -> ForecastResult:
        """Execute full forecasting pipeline for binary, multiple_choice, or numeric/discrete questions."""
        created_at = datetime.now(timezone.utc).isoformat()

        # Stage 1: Live market anchor discovery (binary questions benefit most)
        anchors: list[MarketAnchorMatch] = []
        market_status: dict[str, str] = {}
        if enable_network and question.question_type == "binary":
            anchors, market_status = self.market_matcher.find_anchors(question)

        # Stage 2: Zero-key live public evidence collection
        evidence = self.evidence_retriever.collect(question, enable_network=enable_network)
        merged_status = {**market_status, **evidence.source_status}

        # Stage 3: Outside-view structural base rate
        base_rate: Optional[float] = None
        base_rate_reasons: list[str] = []
        if question.question_type == "binary":
            base_rate, base_rate_reasons = estimate_structural_base_rate(question, evidence)

        # Stage 4: Multi-persona LLM panel (if key configured) OR zero-key statistical engine
        persona_estimates = self._run_llm_panel(question, anchors, evidence, base_rate)
        valid_llm_count = sum(1 for e in persona_estimates if e.error is None)
        mode = "hybrid_llm_quant" if valid_llm_count > 0 else "zero_key_statistical_anchor"

        if question.question_type == "binary":
            llm_probs = [
                float(e.probability)
                for e in persona_estimates
                if e.error is None and e.probability is not None
            ]
            final_p, diag = pool_binary_signals(
                base_rate_prior=base_rate if base_rate is not None else 0.50,
                market_anchors=anchors,
                llm_probabilities=llm_probs,
                extremize_d=self.extremize_d,
                prob_min=self.prob_min,
                prob_max=self.prob_max,
            )
            if not persona_estimates:
                persona_estimates.append(
                    PersonaEstimate(
                        persona="Deterministic Quant + Market-Anchor Fallback",
                        provider="local_statistical",
                        model="log_odds_shrinkage_v1",
                        probability=final_p,
                        rationale=(
                            f"Base rate={diag['base_rate_prior']:.1%} ({', '.join(base_rate_reasons)}); "
                            f"market_anchors={len(anchors)}; weights={diag['weights']}; "
                            f"extremize_d={diag['extremize_d']} -> {final_p:.2%}"
                        ),
                    )
                )
            api_payload = create_forecast_payload("binary", binary_probability=final_p)
            reasoning = self._format_reasoning_comment(
                question=question,
                mode=mode,
                summary_line=f"**Final Binary Forecast: `{final_p:.2%}`** (clipped to [{self.prob_min:.0%}, {self.prob_max:.0%}])",
                diagnostics=diag,
                anchors=anchors,
                evidence=evidence,
                personas=persona_estimates,
            )
            return ForecastResult(
                question_id=question.question_id,
                post_id=question.post_id,
                question_type="binary",
                title=question.title,
                binary_probability=final_p,
                api_payload=api_payload,
                reasoning_markdown=reasoning,
                base_rate_prior=base_rate,
                market_anchors=anchors,
                persona_estimates=persona_estimates,
                evidence_status=merged_status,
                mode=mode,
                created_at=created_at,
            )

        if question.question_type == "multiple_choice":
            options = question.options or ["Option A", "Option B"]
            valid_maps = [
                e.option_probabilities
                for e in persona_estimates
                if e.error is None and e.option_probabilities
            ]
            if valid_maps:
                agg_scores = {
                    opt: sum(max(0.0, m.get(opt, 0.0)) for m in valid_maps) / len(valid_maps)
                    for opt in options
                }
            else:
                # Zero-key evidence keyword overlap scoring + strong Dirichlet smoothing
                ev_text = " ".join(f"{it.title} {it.snippet}" for it in evidence.items).lower()
                agg_scores = {}
                for opt in options:
                    opt_tokens = [w for w in re.findall(r"[a-z0-9]+", opt.lower()) if len(w) >= 3]
                    hits = sum(1 for tok in opt_tokens if tok in ev_text) if opt_tokens else 0
                    agg_scores[opt] = 1.0 + 0.25 * hits
                persona_estimates.append(
                    PersonaEstimate(
                        persona="Deterministic Dirichlet Simplex Fallback",
                        provider="local_statistical",
                        model="dirichlet_simplex_v1",
                        option_probabilities=agg_scores,
                        rationale="Dirichlet-smoothed uniform/evidence prior across all declared options.",
                    )
                )

            mc_probs = normalize_multiple_choice_probs(
                agg_scores,
                options=options,
                dirichlet_alpha=0.35 if valid_maps else 1.25,
            )
            api_payload = create_forecast_payload(
                "multiple_choice", multiple_choice_probabilities=mc_probs
            )
            top_opt = max(mc_probs.items(), key=lambda kv: kv[1])
            reasoning = self._format_reasoning_comment(
                question=question,
                mode=mode,
                summary_line=(
                    f"**Final Multiple-Choice Distribution** (leading option: `{top_opt[0]}` at `{top_opt[1]:.2%}`)"
                ),
                diagnostics={"option_probabilities": mc_probs},
                anchors=anchors,
                evidence=evidence,
                personas=persona_estimates,
            )
            return ForecastResult(
                question_id=question.question_id,
                post_id=question.post_id,
                question_type="multiple_choice",
                title=question.title,
                multiple_choice_probabilities=mc_probs,
                api_payload=api_payload,
                reasoning_markdown=reasoning,
                market_anchors=anchors,
                persona_estimates=persona_estimates,
                evidence_status=merged_status,
                mode=mode,
                created_at=created_at,
            )

        # Numeric / Discrete / Date CDF pipeline
        scaling = question.scaling
        rmin = scaling.range_min if scaling is not None else 0.0
        rmax = scaling.range_max if scaling is not None else 100.0
        zp = scaling.zero_point if scaling is not None else None
        cdf_size = 201
        if (
            question.question_type == "discrete"
            and scaling is not None
            and scaling.inbound_outcome_count is not None
        ):
            cdf_size = int(scaling.inbound_outcome_count) + 1

        valid_pcts = [
            e.percentiles
            for e in persona_estimates
            if e.error is None and e.percentiles and len(e.percentiles) >= 3
        ]
        if valid_pcts:
            common_keys = sorted({k for d in valid_pcts for k in d.keys()})
            merged_pcts: dict[int, float] = {}
            for k in common_keys:
                vals = [d[k] for d in valid_pcts if k in d]
                if vals:
                    merged_pcts[int(k)] = float(np.median(vals))
        else:
            merged_pcts = default_numeric_percentiles(question, evidence)
            persona_estimates.append(
                PersonaEstimate(
                    persona="Deterministic Standardized CDF Fallback",
                    provider="local_statistical",
                    model="metaculus_cdf_v1",
                    percentiles=merged_pcts,
                    rationale=(
                        f"Generated calibrated {cdf_size}-point Metaculus CDF over [{rmin}, {rmax}] "
                        f"(open_lower={question.open_lower_bound}, open_upper={question.open_upper_bound})."
                    ),
                )
            )

        cdf = build_metaculus_cdf(
            percentiles=merged_pcts,
            range_min=rmin,
            range_max=rmax,
            open_lower_bound=question.open_lower_bound,
            open_upper_bound=question.open_upper_bound,
            zero_point=zp,
            cdf_size=cdf_size,
        )
        api_payload = create_forecast_payload(question.question_type, continuous_cdf=cdf)
        p50 = merged_pcts.get(50, sorted(merged_pcts.values())[len(merged_pcts) // 2])
        reasoning = self._format_reasoning_comment(
            question=question,
            mode=mode,
            summary_line=(
                f"**Final {question.question_type.title()} CDF ({len(cdf)} points)** "
                f"(P50 ≈ `{p50}`, bounds=`[{rmin}, {rmax}]`)"
            ),
            diagnostics={
                "percentiles": merged_pcts,
                "cdf_size": len(cdf),
                "cdf_start": cdf[0],
                "cdf_end": cdf[-1],
            },
            anchors=anchors,
            evidence=evidence,
            personas=persona_estimates,
        )
        return ForecastResult(
            question_id=question.question_id,
            post_id=question.post_id,
            question_type=question.question_type,
            title=question.title,
            continuous_cdf=cdf,
            api_payload=api_payload,
            reasoning_markdown=reasoning,
            market_anchors=anchors,
            persona_estimates=persona_estimates,
            evidence_status=merged_status,
            mode=mode,
            created_at=created_at,
        )

    @staticmethod
    def _format_reasoning_comment(
        question: MetaculusQuestion,
        mode: str,
        summary_line: str,
        diagnostics: dict[str, Any],
        anchors: list[MarketAnchorMatch],
        evidence: EvidenceBundle,
        personas: list[PersonaEstimate],
    ) -> str:
        lines = [
            f"### Automated Forecast Summary (`{mode}`)",
            summary_line,
            "",
            "#### 1. External Prediction Market Anchors",
        ]
        if anchors:
            for a in anchors:
                inv = " (polarity inverted -> 1-p)" if a.polarity_inverted else ""
                lines.append(
                    f"- **{a.platform.title()}**: `{a.aligned_probability:.2%}`{inv} | "
                    f"sim=`{a.similarity_score:.2f}`, date_align=`{a.date_alignment_score:.2f}`, "
                    f"liq=`${a.liquidity_usd:,.0f}` — [{a.question}]({a.url})"
                )
        else:
            lines.append("- None matched above strict similarity/date/liquidity gates.")

        lines.extend(["", "#### 2. Live Public Evidence Snapshot", format_evidence_digest(evidence, max_items=6), ""])
        lines.append("#### 3. Panel & Aggregation Diagnostics")
        for p in personas:
            if p.error:
                lines.append(f"- **{p.persona}** (`{p.provider}/{p.model}`): ERROR `{p.error[:140]}`")
            elif p.probability is not None:
                lines.append(f"- **{p.persona}** (`{p.provider}/{p.model}`): `{p.probability:.2%}` — {p.rationale[:240]}")
            else:
                lines.append(f"- **{p.persona}** (`{p.provider}/{p.model}`): {p.rationale[:240]}")

        lines.extend(
            [
                "",
                "```json",
                json.dumps(diagnostics, indent=2, sort_keys=True),
                "```",
                "",
                "_Generated autonomously (no human-in-the-loop) by `metaculus-futureeval-quant-bot`._",
            ]
        )
        return "\n".join(lines)
