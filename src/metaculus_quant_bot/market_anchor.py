"""Zero-API-key live prediction market matcher (Polymarket Gamma API + Manifold Markets API).

Matches open Metaculus questions against live binary prediction markets with:
1. Lexical + named-entity + numeric-threshold overlap scoring
2. Strict polarity inversion detection (flips 1 - p when predicate is cleanly inverted)
3. Close-date / target-year alignment scoring (prevents anchoring 2026 questions to 2028 markets)
4. Platform-differentiated liquidity & volume quality weighting (USDC Polymarket > play-money Manifold)
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from .models import MarketAnchorMatch, MetaculusQuestion

POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
MANIFOLD_API_BASE = "https://api.manifold.markets/v0"
USER_AGENT = (
    "MetaculusFutureEvalQuantBot/1.0 "
    "(https://github.com/pecopeko/metaculus-futureeval-quant-bot)"
)

STOPWORDS = {
    "will",
    "the",
    "be",
    "in",
    "of",
    "to",
    "a",
    "an",
    "on",
    "at",
    "by",
    "for",
    "from",
    "or",
    "and",
    "is",
    "are",
    "was",
    "were",
    "as",
    "this",
    "that",
    "these",
    "those",
    "before",
    "after",
    "during",
    "end",
    "endof",
    "any",
    "than",
    "more",
    "less",
    "over",
    "under",
    "above",
    "below",
    "yes",
    "no",
    "not",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "if",
    "it",
    "its",
    "their",
    "there",
    "what",
    "which",
    "who",
    "when",
    "where",
    "how",
    "between",
    "into",
    "out",
    "up",
    "down",
    "new",
    "first",
    "next",
    "least",
    "most",
    "year",
    "month",
    "day",
    "date",
    "time",
    "question",
    "resolves",
    "resolve",
}

NEGATION_TOKENS = {"not", "never", "fail", "fails", "failed", "without", "avoid", "avoids"}

# Strict binary complement pairs ONLY (never include 3-way mutually exclusive pairs like increase/decrease/no-change)
ANTONYM_PAIRS: list[tuple[set[str], set[str]]] = [
    ({"above", "over", "exceed", "exceeds", "higher", "greater", "more"}, {"below", "under", "lower", "fewer", "less"}),
    ({"pass", "passes", "approve", "approved"}, {"reject", "rejected", "fail", "fails"}),
]

# Mutually exclusive directional actions where a 3rd state ("no change" / "hold") exists:
# mismatching these must REJECT the candidate rather than flipping 1 - p!
MUTUALLY_EXCLUSIVE_ACTIONS: list[set[str]] = [
    {"increase", "increases", "rise", "rises", "hike", "hikes", "raise", "raises"},
    {"decrease", "decreases", "fall", "falls", "cut", "cuts", "drop", "drops", "lower", "lowers"},
    {"unchanged", "hold", "holds", "pause", "pauses"},
]

YEAR_RE = re.compile(r"\b(202[4-9]|203[0-9])\b")
THRESHOLD_RE = re.compile(
    r"(?:\$|€|£)?\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:%|percent|bp|bps|k|m|b|t|million|billion|trillion))?\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-z0-9]+")

CANONICAL_PHRASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bfederal\s+reserve\b", re.IGNORECASE), "fed"),
    (re.compile(r"\bu\.?\s*s\.?\s*a\.?\b|\bunited\s+states\b", re.IGNORECASE), "us"),
    (re.compile(r"\bbasis\s+points?\b|\bbps\b", re.IGNORECASE), "bps"),
    (re.compile(r"\binterest\s+rates?\b", re.IGNORECASE), "interest rate"),
    (re.compile(r"\bartificial\s+intelligence\b", re.IGNORECASE), "ai"),
    (re.compile(r"\bs\s*&\s*p\s*500\b", re.IGNORECASE), "sp500"),
    (re.compile(r"\bconsumer\s+price\s+index\b", re.IGNORECASE), "cpi"),
    (re.compile(r"\bgross\s+domestic\s+product\b", re.IGNORECASE), "gdp"),
]


def _canonicalize_text(text: str) -> str:
    out = text or ""
    for pat, repl in CANONICAL_PHRASES:
        out = pat.sub(repl, out)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        v = float(value)
        return default if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return default


def _parse_json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def extract_years(text: str) -> set[int]:
    """Extract 4-digit forecast years (2024-2039) mentioned in text."""
    return {int(m) for m in YEAR_RE.findall(text or "")}


def extract_numeric_thresholds(text: str) -> set[str]:
    """Extract normalized non-year numeric thresholds (e.g., '$100k', '5%', '4.5')."""
    out: set[str] = set()
    for match in THRESHOLD_RE.findall(text or ""):
        cleaned = match.strip().lower().replace(",", "").replace(" ", "")
        # Skip standalone 4-digit years handled separately by extract_years
        if re.fullmatch(r"20[2-3]\d", cleaned):
            continue
        # Normalize shorthand suffixes
        cleaned = cleaned.replace("percent", "%")
        out.add(cleaned)
    return out


def extract_content_tokens(text: str) -> set[str]:
    """Extract normalized non-stopword lexical tokens with forecasting entity canonicalization."""
    tokens = TOKEN_RE.findall(_canonicalize_text(text).lower())
    return {t for t in tokens if len(t) >= 2 and t not in STOPWORDS and not re.fullmatch(r"20[2-3]\d", t)}


def extract_entity_tokens(text: str) -> set[str]:
    """Extract capitalized entity words / acronyms from original-case text."""
    entities: set[str] = set()
    canon = _canonicalize_text(text)
    for idx, word in enumerate(re.findall(r"[A-Za-z][A-Za-z0-9&'-]*", canon)):
        low = word.lower()
        if low in STOPWORDS or len(low) < 2:
            continue
        if low in {"fed", "us", "ai", "cpi", "gdp", "sp500", "fomc", "nato", "eu", "uk", "sec", "fda"}:
            entities.add(low)
        elif word.isupper() and len(word) >= 2:
            entities.add(low)
        elif idx > 0 and word[0].isupper():
            entities.add(low)
    return entities


def build_search_queries(title: str) -> list[str]:
    """Generate 1-3 compact high-signal search queries from a Metaculus question title.

    Polymarket Gamma public-search uses strict AND matching, so a concise 2-3 word core
    entity/topic query dramatically increases recall while the scorer filters false positives.
    """
    canon = _canonicalize_text(title)
    cleaned = re.sub(r"[?!.,:;()\[\]\"']", " ", canon).strip()
    entities = sorted(extract_entity_tokens(title))
    content = [
        t for t in TOKEN_RE.findall(cleaned.lower())
        if len(t) >= 2 and t not in STOPWORDS and not re.fullmatch(r"20[2-3]\d", t)
    ]
    queries: list[str] = []
    if len(content) >= 2:
        q_short = " ".join(content[:2]).strip()
        if q_short:
            queries.append(q_short)
    if entities:
        # Prefer specific non-country entity first if available
        specific_ent = [e for e in entities if e not in {"us", "uk", "eu"}] or entities
        extra = [t for t in content if t not in set(specific_ent[:2]) and t not in {"us", "uk", "eu"}][:1]
        q_ent = " ".join(specific_ent[:1] + extra).strip()
        if q_ent and q_ent not in queries:
            queries.append(q_ent)
    q_med = " ".join(content[:4]).strip()
    if q_med and q_med not in queries:
        queries.append(q_med)
    return queries[:3] or [cleaned[:64]]


def has_mutually_exclusive_action_conflict(source_text: str, candidate_text: str) -> bool:
    """Return True if source and candidate specify mutually exclusive 3-way actions
    (e.g., Fed 'decrease' vs 'increase' vs 'no change'), which are NOT complements."""
    s_low = (source_text or "").lower()
    c_low = (candidate_text or "").lower()
    s_toks = set(TOKEN_RE.findall(s_low))
    c_toks = set(TOKEN_RE.findall(c_low))

    s_states: set[int] = set()
    c_states: set[int] = set()
    for idx, group in enumerate(MUTUALLY_EXCLUSIVE_ACTIONS):
        if s_toks & group:
            s_states.add(idx)
        if c_toks & group:
            c_states.add(idx)
    if "no change" in s_low:
        s_states.add(2)
    if "no change" in c_low:
        c_states.add(2)
    return bool(s_states and c_states and not (s_states & c_states))


def detect_polarity_inversion(source_text: str, candidate_text: str) -> tuple[bool, str]:
    """Detect whether candidate market states the logical negation of source question.

    Returns (is_inverted, reason). Conservative by design to avoid erroneous flips.
    """
    src_low = (source_text or "").lower()
    cand_low = (candidate_text or "").lower()
    src_tokens = set(TOKEN_RE.findall(src_low))
    cand_tokens = set(TOKEN_RE.findall(cand_low))

    src_neg = bool(src_tokens & NEGATION_TOKENS) or ("no " in src_low and "no change" not in src_low)
    cand_neg = bool(cand_tokens & NEGATION_TOKENS) or ("no " in cand_low and "no change" not in cand_low)

    antonym_flips = 0
    antonym_desc = ""
    for pos_set, neg_set in ANTONYM_PAIRS:
        src_pos = bool(src_tokens & pos_set)
        src_opp = bool(src_tokens & neg_set)
        cand_pos = bool(cand_tokens & pos_set)
        cand_opp = bool(cand_tokens & neg_set)
        if (src_pos and not src_opp and cand_opp and not cand_pos) or (
            src_opp and not src_pos and cand_pos and not cand_opp
        ):
            antonym_flips += 1
            antonym_desc = f"antonym_pair({sorted(src_tokens & (pos_set | neg_set))} vs {sorted(cand_tokens & (pos_set | neg_set))})"

    # XOR logic: negation flip XOR antonym flip => net polarity inversion
    net_neg_flip = src_neg != cand_neg
    net_antonym_flip = (antonym_flips % 2) == 1

    if net_neg_flip and not net_antonym_flip:
        return True, "negation_asymmetry"
    if net_antonym_flip and not net_neg_flip:
        return True, antonym_desc or "directional_antonym"
    return False, "aligned"


def _parse_iso_date(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    if isinstance(raw, (int, float)):
        # Manifold uses epoch milliseconds
        ts = float(raw) / 1000.0 if float(raw) > 1e11 else float(raw)
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if m:
            try:
                return datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
            except ValueError:
                return None
    return None


def score_date_alignment(
    question_title: str,
    question_resolve_time: Optional[str],
    market_question: str,
    market_close_date: Optional[str],
) -> tuple[float, str]:
    """Compute [0, 1] temporal alignment between a Metaculus question and a candidate market."""
    q_years = extract_years(question_title)
    m_years = extract_years(market_question)
    q_dt = _parse_iso_date(question_resolve_time)
    m_dt = _parse_iso_date(market_close_date)

    if q_years and m_years:
        if q_years & m_years:
            year_score = 1.0
            reason = f"target_year_match({sorted(q_years & m_years)})"
        else:
            min_gap = min(abs(qy - my) for qy in q_years for my in m_years)
            year_score = 0.12 if min_gap == 1 else 0.05
            reason = f"target_year_mismatch({sorted(q_years)} vs {sorted(m_years)})"
        return year_score, reason

    if q_dt is not None and m_dt is not None:
        gap_days = abs((q_dt - m_dt).total_seconds()) / 86400.0
        if gap_days <= 30:
            return 1.0, f"close_date_within_{int(gap_days)}d"
        if gap_days <= 90:
            return 0.85, f"close_date_within_{int(gap_days)}d"
        if gap_days <= 180:
            return 0.65, f"close_date_gap_{int(gap_days)}d"
        if gap_days <= 365:
            return 0.35, f"close_date_gap_{int(gap_days)}d"
        return 0.15, f"close_date_far_{int(gap_days)}d"

    if q_years and m_dt is not None:
        if m_dt.year in q_years:
            return 0.90, f"market_close_year_{m_dt.year}_matches"
        return 0.25, f"market_close_year_{m_dt.year}_differs_{sorted(q_years)}"

    return 0.75, "date_unspecified_neutral"


def score_candidate_match(
    question: MetaculusQuestion,
    platform: str,
    market_id: str,
    market_question: str,
    market_url: str,
    raw_probability: float,
    liquidity_usd: float,
    volume_usd: float,
    close_date: Optional[str],
    min_similarity: float = 0.40,
    min_date_alignment: float = 0.50,
    min_liquidity_usd: float = 100.0,
) -> Optional[MarketAnchorMatch]:
    """Score and gate a candidate prediction market contract against a Metaculus question."""
    if not (0.001 <= raw_probability <= 0.999):
        return None
    if platform in ("polymarket", "kalshi") and liquidity_usd < min_liquidity_usd and volume_usd < min_liquidity_usd * 3:
        return None
    if platform == "manifold" and liquidity_usd < min(50.0, min_liquidity_usd) and volume_usd < 100.0:
        return None

    q_tokens = extract_content_tokens(question.title)
    m_tokens = extract_content_tokens(market_question)
    if not q_tokens or not m_tokens:
        return None

    # Reject mutually exclusive 3-way action conflicts (e.g. rate cut vs rate hike vs no change)
    if has_mutually_exclusive_action_conflict(question.title, market_question):
        return None

    overlap = q_tokens & m_tokens
    union = q_tokens | m_tokens
    if len(overlap) < 2:
        return None
    jaccard = len(overlap) / len(union)
    if jaccard < 0.25:
        return None
    containment = len(overlap) / max(1, min(len(q_tokens), len(m_tokens)))

    q_entities = extract_entity_tokens(question.title)
    m_entities = extract_entity_tokens(market_question)
    if q_entities and m_entities:
        # Guard against substituting a different named organization/country (e.g. Fed vs Bank of Mexico)
        generic_months = {
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "q1", "q2", "q3", "q4",
        }
        q_core_ent = q_entities - generic_months
        m_core_ent = m_entities - generic_months
        if q_core_ent and m_core_ent and not (q_core_ent & m_core_ent):
            return None
        ent_overlap = (2.0 * len(q_entities & m_entities)) / (len(q_entities) + len(m_entities))
    else:
        ent_overlap = jaccard

    similarity = 0.50 * jaccard + 0.25 * containment + 0.25 * ent_overlap
    reasons: list[str] = [
        f"tokens_overlap={len(overlap)}",
        f"jaccard={jaccard:.2f}",
        f"entity_overlap={ent_overlap:.2f}",
    ]

    # Strict numeric threshold conflict guard (e.g. BTC > $100k vs BTC > $150k)
    q_nums = extract_numeric_thresholds(question.title)
    m_nums = extract_numeric_thresholds(market_question)
    if q_nums and m_nums and not (q_nums & m_nums):
        similarity *= 0.35
        reasons.append(f"threshold_mismatch({sorted(q_nums)} vs {sorted(m_nums)})")
    elif q_nums and m_nums and (q_nums & m_nums):
        similarity = min(1.0, similarity + 0.12)
        reasons.append(f"threshold_match({sorted(q_nums & m_nums)})")

    if similarity < min_similarity:
        return None

    date_score, date_reason = score_date_alignment(
        question.title,
        question.scheduled_resolve_time or question.scheduled_close_time,
        market_question,
        close_date,
    )
    reasons.append(date_reason)
    if date_score < min_date_alignment:
        return None

    inverted, pol_reason = detect_polarity_inversion(question.title, market_question)
    aligned_prob = (1.0 - raw_probability) if inverted else raw_probability
    if inverted:
        reasons.append(f"polarity_inverted({pol_reason})")

    if platform == "polymarket":
        liq_score = min(1.0, math.log1p(max(0.0, liquidity_usd)) / math.log1p(100_000.0))
        vol_score = min(1.0, math.log1p(max(0.0, volume_usd)) / math.log1p(250_000.0))
        platform_mult = 1.0
    elif platform == "kalshi":
        liq_score = min(1.0, math.log1p(max(0.0, liquidity_usd)) / math.log1p(80_000.0))
        vol_score = min(1.0, math.log1p(max(0.0, volume_usd)) / math.log1p(200_000.0))
        platform_mult = 0.95
    else:
        liq_score = min(1.0, math.log1p(max(0.0, liquidity_usd)) / math.log1p(20_000.0))
        vol_score = min(1.0, math.log1p(max(0.0, volume_usd)) / math.log1p(50_000.0))
        platform_mult = 0.65

    depth_score = 0.60 * liq_score + 0.40 * vol_score
    quality_weight = round(
        platform_mult * (0.55 * similarity + 0.25 * date_score + 0.20 * depth_score),
        4,
    )

    return MarketAnchorMatch(
        platform=platform,  # type: ignore[arg-type]
        market_id=str(market_id),
        question=market_question,
        url=market_url,
        raw_probability=round(raw_probability, 6),
        aligned_probability=round(aligned_prob, 6),
        polarity_inverted=inverted,
        similarity_score=round(similarity, 4),
        liquidity_usd=round(liquidity_usd, 2),
        volume_usd=round(volume_usd, 2),
        close_date=close_date,
        date_alignment_score=round(date_score, 4),
        quality_weight=quality_weight,
        match_reasons=reasons,
    )


class MarketAnchorMatcher:
    """Queries live Polymarket Gamma, Kalshi, & Manifold Markets public APIs without any API key."""

    def __init__(
        self,
        timeout: float = 10.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.timeout = timeout
        self._external_client = http_client
        self._kalshi_cache: Optional[tuple[list[dict[str, Any]], str]] = None

    def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self._external_client is not None:
            resp = self._external_client.get(url, params=params, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        with httpx.Client(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    def search_polymarket(self, query: str, limit: int = 8) -> tuple[list[dict[str, Any]], str]:
        """Search active binary Polymarket contracts via Gamma public-search."""
        try:
            data = self._get_json(
                f"{POLYMARKET_GAMMA_BASE}/public-search",
                {"q": query, "events_status": "active", "limit_per_type": limit},
            )
        except Exception as exc:
            return [], f"error:{type(exc).__name__}"

        events = data.get("events") if isinstance(data, dict) else []
        candidates: list[dict[str, Any]] = []
        for event in events or []:
            if not isinstance(event, dict):
                continue
            event_liq = _safe_float(event.get("liquidity") or event.get("liquidityClob"))
            for m in event.get("markets") or []:
                if not isinstance(m, dict):
                    continue
                if m.get("closed") is True or m.get("active") is False:
                    continue
                outcomes = [str(o).strip().lower() for o in _parse_json_list(m.get("outcomes"))]
                prices = [_safe_float(p, -1.0) for p in _parse_json_list(m.get("outcomePrices"))]
                if len(outcomes) != 2 or len(prices) != 2 or "yes" not in outcomes:
                    continue
                yes_idx = outcomes.index("yes")
                yes_prob = prices[yes_idx]
                if not (0.001 <= yes_prob <= 0.999):
                    continue
                liq = _safe_float(
                    m.get("liquidityNum")
                    or m.get("liquidity")
                    or m.get("liquidityClob")
                    or event_liq
                )
                vol = _safe_float(m.get("volumeNum") or m.get("volume") or event.get("volume"))
                slug = m.get("slug") or event.get("slug") or ""
                candidates.append(
                    {
                        "platform": "polymarket",
                        "market_id": str(m.get("id") or ""),
                        "question": str(m.get("question") or event.get("title") or ""),
                        "url": f"https://polymarket.com/event/{event.get('slug') or slug}",
                        "raw_probability": yes_prob,
                        "liquidity_usd": liq,
                        "volume_usd": vol,
                        "close_date": m.get("endDate") or event.get("endDate"),
                    }
                )
        return candidates, f"ok ({len(candidates)} active binary markets)"

    def search_manifold(self, query: str, limit: int = 12) -> tuple[list[dict[str, Any]], str]:
        """Search open binary Manifold Markets contracts."""
        try:
            data = self._get_json(
                f"{MANIFOLD_API_BASE}/search-markets",
                {
                    "term": query,
                    "filter": "open",
                    "contractType": "BINARY",
                    "sort": "liquidity",
                    "limit": limit,
                },
            )
        except Exception as exc:
            return [], f"error:{type(exc).__name__}"

        if not isinstance(data, list):
            return [], "unexpected_payload"

        candidates: list[dict[str, Any]] = []
        for m in data:
            if not isinstance(m, dict) or m.get("isResolved"):
                continue
            if m.get("outcomeType") != "BINARY":
                continue
            prob = _safe_float(m.get("probability"), -1.0)
            if not (0.005 <= prob <= 0.995):
                continue
            close_raw = m.get("closeTime")
            close_iso: Optional[str] = None
            dt = _parse_iso_date(close_raw)
            if dt is not None:
                close_iso = dt.date().isoformat()
            candidates.append(
                {
                    "platform": "manifold",
                    "market_id": str(m.get("id") or ""),
                    "question": str(m.get("question") or ""),
                    "url": str(m.get("url") or f"https://manifold.markets/market/{m.get('slug', '')}"),
                    "raw_probability": prob,
                    "liquidity_usd": _safe_float(m.get("totalLiquidity")),
                    "volume_usd": _safe_float(m.get("volume")),
                    "close_date": close_iso,
                }
            )
        return candidates, f"ok ({len(candidates)} open binary markets)"

    def search_kalshi(self, query: str, limit: int = 10) -> tuple[list[dict[str, Any]], str]:
        """Search open regulated binary CFTC contracts on Kalshi public v2 API."""
        if self._kalshi_cache is None:
            try:
                data = self._get_json(
                    f"{KALSHI_API_BASE}/events",
                    {"status": "open", "with_nested_markets": "true", "limit": 150},
                )
                events = data.get("events") if isinstance(data, dict) else []
                parsed_all: list[dict[str, Any]] = []
                for ev in events or []:
                    if not isinstance(ev, dict):
                        continue
                    series_slug = str(ev.get("series_ticker") or ev.get("event_ticker") or "").lower()
                    ev_title = str(ev.get("title") or "")
                    for mk in ev.get("markets") or []:
                        if not isinstance(mk, dict):
                            continue
                        if mk.get("market_type") != "binary" or mk.get("status") not in ("active", "open"):
                            continue
                        bid = _safe_float(mk.get("yes_bid_dollars"), -1.0)
                        ask = _safe_float(mk.get("yes_ask_dollars"), -1.0)
                        last_p = _safe_float(mk.get("last_price_dollars"), -1.0)
                        if 0.0 <= bid <= 1.0 and 0.0 <= ask <= 1.0 and ask >= bid and (bid + ask) > 0:
                            prob = 0.5 * (bid + ask)
                        else:
                            prob = last_p
                        if not (0.002 <= prob <= 0.998):
                            continue
                        liq = _safe_float(mk.get("liquidity_dollars") or mk.get("open_interest_fp"))
                        vol = _safe_float(mk.get("volume_fp") or mk.get("volume"))
                        q_text = str(mk.get("title") or ev_title)
                        parsed_all.append(
                            {
                                "platform": "kalshi",
                                "market_id": str(mk.get("ticker") or ""),
                                "question": q_text,
                                "url": f"https://kalshi.com/markets/{series_slug}",
                                "raw_probability": prob,
                                "liquidity_usd": liq,
                                "volume_usd": vol,
                                "close_date": mk.get("expected_expiration_time") or mk.get("close_time"),
                            }
                        )
                self._kalshi_cache = (parsed_all, f"ok ({len(parsed_all)} active binary markets)")
            except Exception as exc:
                self._kalshi_cache = ([], f"error:{type(exc).__name__}")

        all_markets, stat = self._kalshi_cache
        if not all_markets:
            return [], stat
        q_toks = extract_content_tokens(query)
        if not q_toks:
            return all_markets[:limit], stat
        matched = [
            c for c in all_markets if q_toks & extract_content_tokens(c["question"])
        ]
        return matched[:limit], stat

    def find_anchors(
        self,
        question: MetaculusQuestion,
        max_matches: int = 4,
        min_similarity: float = 0.40,
        min_date_alignment: float = 0.50,
    ) -> tuple[list[MarketAnchorMatch], dict[str, str]]:
        """Query Polymarket + Kalshi + Manifold and return quality-ranked verified matches."""
        queries = build_search_queries(question.title)
        seen_ids: set[tuple[str, str]] = set()
        matches: list[MarketAnchorMatch] = []
        status: dict[str, str] = {}

        for idx, query in enumerate(queries):
            pm_cands, pm_stat = self.search_polymarket(query)
            kl_cands, kl_stat = self.search_kalshi(query)
            mf_cands, mf_stat = self.search_manifold(query)
            status[f"polymarket_q{idx + 1}"] = pm_stat
            status[f"kalshi_q{idx + 1}"] = kl_stat
            status[f"manifold_q{idx + 1}"] = mf_stat

            for cand in pm_cands + kl_cands + mf_cands:
                key = (cand["platform"], cand["market_id"])
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                scored = score_candidate_match(
                    question=question,
                    platform=cand["platform"],
                    market_id=cand["market_id"],
                    market_question=cand["question"],
                    market_url=cand["url"],
                    raw_probability=cand["raw_probability"],
                    liquidity_usd=cand["liquidity_usd"],
                    volume_usd=cand["volume_usd"],
                    close_date=cand["close_date"],
                    min_similarity=min_similarity,
                    min_date_alignment=min_date_alignment,
                )
                if scored is not None:
                    matches.append(scored)

        matches.sort(key=lambda m: m.quality_weight, reverse=True)
        # When an exact/near-exact match exists, filter out much weaker sibling markets
        if matches and matches[0].similarity_score >= 0.90:
            floor_sim = matches[0].similarity_score - 0.18
            matches = [m for m in matches if m.similarity_score >= floor_sim]
        return matches[:max_matches], status
