"""Zero-API-key live public evidence collector with polite rate limiting & transparent provenance.

Supported public sources (zero paid API keys, zero CAPTCHA bypass):
- Wikipedia Search + REST Summary API (always queried for entity/background grounding)
- GDELT 2.0 Doc API (global recent timestamped news articles)
- FRED Public CSV (macro series: FEDFUNDS, CPIAUCSL, UNRATE, DGS10, GDP, SP500)
- arXiv Atom API (triggered for AI/ML/science/biotech/climate/hardware questions)
- Hacker News Algolia API (triggered for tech/AI/startup/crypto/software questions)
- SEC EDGAR EFTS (triggered for corporate/earnings/IPO questions; surfaces HTTP 403 transparently)
"""

from __future__ import annotations

import csv
import io
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote

import httpx

from .market_anchor import build_search_queries
from .models import EvidenceBundle, EvidenceItem, MetaculusQuestion

USER_AGENT = (
    "MetaculusFutureEvalQuantBot/1.0 "
    "(https://github.com/pecopeko/metaculus-futureeval-quant-bot)"
)

ARXIV_KEYWORDS = {
    "ai", "llm", "model", "gpt", "claude", "gemini", "benchmark", "agi",
    "compute", "gpu", "chip", "semiconductor", "quantum", "fusion",
    "vaccine", "clinical", "fda", "drug", "virus", "pandemic", "cancer",
    "climate", "warming", "temperature", "carbon", "space", "launch", "mars",
}

TECH_KEYWORDS = {
    "ai", "openai", "anthropic", "google", "meta", "microsoft", "nvidia",
    "apple", "amazon", "tesla", "xai", "deepseek", "github", "linux",
    "bitcoin", "ethereum", "crypto", "software", "chip", "semiconductor",
    "startup", "ipo", "regulator", "antitrust", "model", "gpu",
}

CORP_KEYWORDS = {
    "earnings", "revenue", "ipo", "sec", "merger", "acquisition", "stock",
    "market cap", "dividend", "bankruptcy", "filing", "10-k", "10-q",
    "nvidia", "apple", "microsoft", "tesla", "amazon", "meta", "alphabet",
}

FRED_SERIES_TRIGGERS: list[tuple[set[str], str, str]] = [
    ({"fed", "federal reserve", "interest rate", "fomc", "fed funds", "rate cut", "rate hike"}, "FEDFUNDS", "Federal Funds Effective Rate (%)"),
    ({"inflation", "cpi", "consumer price"}, "CPIAUCSL", "US CPI All Urban Consumers (Index)"),
    ({"unemployment", "jobs report", "payroll", "labor market"}, "UNRATE", "US Unemployment Rate (%)"),
    ({"treasury", "10-year", "10y", "ust 10y", "bond yield", "yield curve"}, "DGS10", "10-Year Treasury Yield (%)"),
    ({"mortgage", "30-year fixed", "fixed-rate mortgage"}, "MORTGAGE30US", "30-Year Fixed Rate Mortgage Average in the US (%)"),
    ({"vix", "volatility index", "intraday value of the vix"}, "VIXCLS", "CBOE Volatility Index (VIX)"),
    ({"high yield", "option-adjusted spread", "bofa"}, "BAMLH0A0HYM2", "ICE BofA US High Yield OAS (%)"),
    ({"crude oil", "wti", "oil futures"}, "DCOILWTICO", "Crude Oil Prices: WTI ($/bbl)"),
    ({"gdp", "recession", "economic growth"}, "GDP", "US Nominal GDP (Billions USD)"),
    ({"s&p 500", "sp500", "s&p500", "stock market"}, "SP500", "S&P 500 Index"),
]


def _clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def detect_fred_series(text: str) -> list[tuple[str, str]]:
    """Return matching (series_id, label) pairs for macro-relevant questions."""
    low = (text or "").lower()
    matched: list[tuple[str, str]] = []
    for keywords, series_id, label in FRED_SERIES_TRIGGERS:
        if any(kw in low for kw in keywords):
            matched.append((series_id, label))
    return matched[:2]


class EvidenceRetriever:
    """Polite, zero-key public evidence collector."""

    def __init__(
        self,
        timeout: float = 8.0,
        min_request_interval_s: float = 0.2,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.timeout = timeout
        self.min_request_interval_s = min_request_interval_s
        self._external_client = http_client
        self._last_call_ts = 0.0

    def _polite_get(
        self,
        url: str,
        params: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> httpx.Response:
        elapsed = time.monotonic() - self._last_call_ts
        if self.min_request_interval_s > 0 and elapsed < self.min_request_interval_s:
            time.sleep(self.min_request_interval_s - elapsed)
        self._last_call_ts = time.monotonic()

        headers = {"User-Agent": USER_AGENT}
        if extra_headers:
            headers.update(extra_headers)
        if self._external_client is not None:
            return self._external_client.get(
                url, params=params, headers=headers, timeout=self.timeout
            )
        with httpx.Client(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
            return client.get(url, params=params)

    def fetch_wikipedia(self, query: str, limit: int = 3) -> tuple[list[EvidenceItem], str]:
        """Fetch top Wikipedia search results + clean REST summaries."""
        try:
            resp = self._polite_get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": limit,
                },
            )
            if resp.status_code != 200:
                return [], f"http_{resp.status_code}"
            hits = (resp.json().get("query") or {}).get("search") or []
            items: list[EvidenceItem] = []
            for idx, hit in enumerate(hits[:limit]):
                title = str(hit.get("title") or "")
                snippet = _clean_html(str(hit.get("snippet") or ""))
                ts = str(hit.get("timestamp") or "") or None
                page_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                # Enrich top hit with concise REST summary
                if idx == 0 and title:
                    try:
                        s_resp = self._polite_get(
                            f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'))}"
                        )
                        if s_resp.status_code == 200:
                            extract = str(s_resp.json().get("extract") or "").strip()
                            if extract:
                                snippet = extract[:380]
                    except Exception:
                        pass
                items.append(
                    EvidenceItem(
                        source="wikipedia",
                        title=title,
                        snippet=snippet[:380],
                        url=page_url,
                        published_at=ts,
                        relevance_score=round(max(0.4, 0.9 - 0.12 * idx), 2),
                    )
                )
            return items, f"ok ({len(items)} items)"
        except Exception as exc:
            return [], f"error:{type(exc).__name__}"

    def fetch_google_news_rss(self, query: str, limit: int = 6) -> tuple[list[EvidenceItem], str]:
        """Fetch live global news headlines via public Google News RSS (zero API key)."""
        clean_q = " ".join(re.findall(r"[A-Za-z0-9'-]+", query)[:7]).strip()
        if not clean_q:
            return [], "skipped:empty_query"

        seen_titles: set[str] = set()
        items: list[EvidenceItem] = []
        statuses: list[str] = []
        for q_variant, base_rel in ((clean_q, 0.93), (f"{clean_q} when:90d", 0.91)):
            try:
                resp = self._polite_get(
                    "https://news.google.com/rss/search",
                    params={
                        "q": q_variant,
                        "hl": "en-US",
                        "gl": "US",
                        "ceid": "US:en",
                    },
                )
                if resp.status_code != 200:
                    statuses.append(f"http_{resp.status_code}")
                    continue
                xml_txt = resp.text or ""
                if "<!DOCTYPE" in xml_txt.upper() or "<!ENTITY" in xml_txt.upper():
                    statuses.append("rejected_unsafe_xml")
                    continue
                root = ET.fromstring(xml_txt)
                rss_items = root.findall("./channel/item")
                statuses.append("ok")
                for idx, entry in enumerate(rss_items[:limit]):
                    title = " ".join((entry.findtext("title") or "").split()).strip()
                    if not title or title.lower() in seen_titles:
                        continue
                    seen_titles.add(title.lower())
                    link = (entry.findtext("link") or "").strip()
                    pub_date = (entry.findtext("pubDate") or "").strip() or None
                    source_el = entry.find("source")
                    publisher = (source_el.text or "").strip() if source_el is not None else "Google News"
                    desc = _clean_html(entry.findtext("description") or "")
                    snippet = f"Publisher: {publisher}"
                    if pub_date:
                        snippet += f" ({pub_date})"
                    if desc and desc.lower() != title.lower():
                        snippet += f" — {desc[:200]}"
                    items.append(
                        EvidenceItem(
                            source="google_news",
                            title=title[:220],
                            snippet=snippet[:340],
                            url=link,
                            published_at=pub_date,
                            relevance_score=round(max(0.55, base_rel - 0.04 * idx), 2),
                        )
                    )
                    if len(items) >= limit:
                        break
            except Exception as exc:
                statuses.append(f"error:{type(exc).__name__}")
            if len(items) >= limit:
                break

        if items:
            return items[:limit], f"ok ({len(items)} items)"
        return [], ",".join(statuses) or "empty"

    def fetch_gdelt(self, query: str, limit: int = 4) -> tuple[list[EvidenceItem], str]:
        """Fetch recent global news articles via GDELT 2.0 Doc API."""
        # GDELT prefers concise alphanumeric queries
        gdelt_q = " ".join(re.findall(r"[A-Za-z0-9]+", query)[:5]).strip()
        if not gdelt_q:
            return [], "skipped:empty_query"
        try:
            resp = self._polite_get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query": gdelt_q,
                    "mode": "artlist",
                    "maxrecords": limit,
                    "sort": "DateDesc",
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                return [], f"http_{resp.status_code}"
            try:
                data = resp.json()
            except ValueError:
                return [], "non_json_response"
            articles = data.get("articles") or []
            items: list[EvidenceItem] = []
            for idx, art in enumerate(articles[:limit]):
                title = str(art.get("title") or "").strip()
                url = str(art.get("url") or "").strip()
                domain = str(art.get("domain") or "")
                seen_date = str(art.get("seendate") or "") or None
                if not title:
                    continue
                items.append(
                    EvidenceItem(
                        source="gdelt",
                        title=title[:200],
                        snippet=f"Source domain: {domain}; language: {art.get('language', 'unknown')}",
                        url=url,
                        published_at=seen_date,
                        relevance_score=round(max(0.4, 0.85 - 0.1 * idx), 2),
                    )
                )
            return items, f"ok ({len(items)} items)"
        except Exception as exc:
            return [], f"error:{type(exc).__name__}"

    def fetch_arxiv(self, query: str, limit: int = 3) -> tuple[list[EvidenceItem], str]:
        """Fetch recent academic preprints from arXiv Atom API."""
        try:
            resp = self._polite_get(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": limit,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
            )
            if resp.status_code != 200:
                return [], f"http_{resp.status_code}"
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            items: list[EvidenceItem] = []
            for idx, entry in enumerate(root.findall("atom:entry", ns)[:limit]):
                title = " ".join((entry.findtext("atom:title", "", ns) or "").split())
                summary = " ".join((entry.findtext("atom:summary", "", ns) or "").split())
                url = (entry.findtext("atom:id", "", ns) or "").strip()
                published = (entry.findtext("atom:published", "", ns) or "").strip() or None
                if title:
                    items.append(
                        EvidenceItem(
                            source="arxiv",
                            title=title[:220],
                            snippet=summary[:340],
                            url=url,
                            published_at=published,
                            relevance_score=round(max(0.4, 0.85 - 0.1 * idx), 2),
                        )
                    )
            return items, f"ok ({len(items)} items)"
        except Exception as exc:
            return [], f"error:{type(exc).__name__}"

    def fetch_fred_series(self, series_id: str, label: str) -> tuple[Optional[EvidenceItem], Optional[dict[str, Any]], str]:
        """Fetch latest macro observations directly from public FRED CSV endpoint."""
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            resp = self._polite_get(url)
            if resp.status_code != 200:
                return None, None, f"http_{resp.status_code}"
            rows: list[tuple[str, float]] = []
            reader = csv.reader(io.StringIO(resp.text))
            next(reader, None)  # header
            for row in reader:
                if len(row) < 2 or row[1] in ("", "."):
                    continue
                try:
                    rows.append((row[0].strip(), float(row[1])))
                except ValueError:
                    continue
            if not rows:
                return None, None, "empty_series"
            latest_date, latest_val = rows[-1]
            prev_date, prev_val = rows[-2] if len(rows) >= 2 else rows[-1]
            lookback_idx = max(0, len(rows) - 7)
            lb_date, lb_val = rows[lookback_idx]
            delta_recent = latest_val - lb_val
            summary = {
                "series_id": series_id,
                "label": label,
                "latest_date": latest_date,
                "latest_value": latest_val,
                "previous_date": prev_date,
                "previous_value": prev_val,
                "six_obs_delta": round(delta_recent, 4),
                "observations_count": len(rows),
            }
            item = EvidenceItem(
                source="fred",
                title=f"FRED {series_id}: {label}",
                snippet=(
                    f"Latest observation ({latest_date}): {latest_val:.3f}; "
                    f"previous ({prev_date}): {prev_val:.3f}; "
                    f"6-step change since {lb_date}: {delta_recent:+.3f}"
                ),
                url=f"https://fred.stlouisfed.org/series/{series_id}",
                published_at=latest_date,
                relevance_score=0.95,
            )
            return item, summary, f"ok ({latest_date}={latest_val:.3f})"
        except Exception as exc:
            return None, None, f"error:{type(exc).__name__}"

    def fetch_hn_algolia(self, query: str, limit: int = 3) -> tuple[list[EvidenceItem], str]:
        """Fetch recent Hacker News technical discussions via public Algolia API."""
        try:
            resp = self._polite_get(
                "https://hn.algolia.com/api/v1/search",
                params={"query": query, "tags": "story", "hitsPerPage": limit},
            )
            if resp.status_code != 200:
                return [], f"http_{resp.status_code}"
            hits = resp.json().get("hits") or []
            items: list[EvidenceItem] = []
            for idx, hit in enumerate(hits[:limit]):
                title = str(hit.get("title") or "").strip()
                if not title:
                    continue
                oid = hit.get("objectID") or ""
                url = str(hit.get("url") or f"https://news.ycombinator.com/item?id={oid}")
                points = hit.get("points") or 0
                comments = hit.get("num_comments") or 0
                items.append(
                    EvidenceItem(
                        source="hn_algolia",
                        title=title[:200],
                        snippet=f"Hacker News discussion: {points} points, {comments} comments",
                        url=url,
                        published_at=hit.get("created_at"),
                        relevance_score=round(max(0.4, 0.80 - 0.1 * idx), 2),
                    )
                )
            return items, f"ok ({len(items)} items)"
        except Exception as exc:
            return [], f"error:{type(exc).__name__}"

    def fetch_sec_edgar(self, query: str, limit: int = 2) -> tuple[list[EvidenceItem], str]:
        """Query SEC EDGAR EFTS full-text filing search (surfaces 403 transparently if blocked)."""
        try:
            resp = self._polite_get(
                "https://efts.sec.gov/LATEST/search-index",
                params={"q": query, "forms": "10-K,10-Q,8-K"},
                extra_headers={
                    "User-Agent": "HorizonMoneyResearch pecopeko@users.noreply.github.com",
                    "Accept": "application/json",
                },
            )
            if resp.status_code != 200:
                return [], f"http_{resp.status_code}"
            hits = ((resp.json().get("hits") or {}).get("hits")) or []
            items: list[EvidenceItem] = []
            for idx, hit in enumerate(hits[:limit]):
                src = hit.get("_source") or {}
                names = src.get("display_names") or ["SEC Filing"]
                form = src.get("form") or "Filing"
                file_date = src.get("file_date")
                adsh = str(src.get("adsh") or "")
                items.append(
                    EvidenceItem(
                        source="sec_edgar",
                        title=f"SEC {form}: {names[0]}",
                        snippet=f"Filed {file_date}; accession {adsh}",
                        url=f"https://www.sec.gov/edgar/search/#/q={quote(query)}",
                        published_at=file_date,
                        relevance_score=0.85,
                    )
                )
            return items, f"ok ({len(items)} items)"
        except Exception as exc:
            return [], f"error:{type(exc).__name__}"

    def collect(
        self,
        question: MetaculusQuestion,
        enable_network: bool = True,
    ) -> EvidenceBundle:
        """Collect structured, timestamped public evidence tailored to question domain."""
        collected_at = datetime.now(timezone.utc).isoformat()
        queries = build_search_queries(question.title)
        primary_query = queries[0] if queries else question.title[:80]

        if not enable_network:
            return EvidenceBundle(
                query=primary_query,
                collected_at=collected_at,
                items=[],
                source_status={"network": "disabled_offline_mode"},
            )

        combined_text = f"{question.title} {question.description[:400]}".lower()
        items: list[EvidenceItem] = []
        source_status: dict[str, str] = {}
        macro_series: dict[str, dict[str, Any]] = {}

        # Always collect Wikipedia, Google News RSS, & GDELT
        wiki_items, wiki_status = self.fetch_wikipedia(primary_query)
        items.extend(wiki_items)
        source_status["wikipedia"] = wiki_status

        gn_query = " ".join(queries[-1:]) if len(queries) > 1 else question.title[:90]
        gn_items, gn_status = self.fetch_google_news_rss(gn_query)
        items.extend(gn_items)
        source_status["google_news"] = gn_status

        gdelt_items, gdelt_status = self.fetch_gdelt(primary_query)
        items.extend(gdelt_items)
        source_status["gdelt"] = gdelt_status

        # Domain-routed FRED macro series
        fred_targets = detect_fred_series(combined_text)
        if fred_targets:
            for series_id, label in fred_targets:
                f_item, f_meta, f_status = self.fetch_fred_series(series_id, label)
                source_status[f"fred:{series_id}"] = f_status
                if f_item is not None:
                    items.append(f_item)
                if f_meta is not None:
                    macro_series[series_id] = f_meta
        else:
            source_status["fred"] = "skipped:non_macro"

        # Domain-routed arXiv
        if any(kw in combined_text for kw in ARXIV_KEYWORDS):
            ax_items, ax_status = self.fetch_arxiv(primary_query)
            items.extend(ax_items)
            source_status["arxiv"] = ax_status
        else:
            source_status["arxiv"] = "skipped:non_research"

        # Domain-routed Hacker News Algolia
        if any(kw in combined_text for kw in TECH_KEYWORDS):
            hn_items, hn_status = self.fetch_hn_algolia(primary_query)
            items.extend(hn_items)
            source_status["hn_algolia"] = hn_status
        else:
            source_status["hn_algolia"] = "skipped:non_tech"

        # Domain-routed SEC EDGAR
        if any(kw in combined_text for kw in CORP_KEYWORDS):
            sec_items, sec_status = self.fetch_sec_edgar(primary_query)
            items.extend(sec_items)
            source_status["sec_edgar"] = sec_status
        else:
            source_status["sec_edgar"] = "skipped:non_corporate"

        items.sort(key=lambda x: x.relevance_score, reverse=True)
        return EvidenceBundle(
            query=primary_query,
            collected_at=collected_at,
            items=items[:14],
            source_status=source_status,
            macro_series=macro_series,
        )


def format_evidence_digest(bundle: EvidenceBundle, max_items: int = 8) -> str:
    """Format an EvidenceBundle into compact Markdown bullets for prompts & comments."""
    if not bundle.items:
        return "No external evidence items retrieved."
    lines: list[str] = []
    for item in bundle.items[:max_items]:
        date_tag = f" [{item.published_at[:10]}]" if item.published_at else ""
        lines.append(
            f"- **[{item.source.upper()}]{date_tag}** {item.title}: {item.snippet[:220]} ({item.url})"
        )
    return "\n".join(lines)
