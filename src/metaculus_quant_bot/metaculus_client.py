"""Live Metaculus API v2 client verified against Metaculus/metac-bot-template (Nov 2026).

Key verified Metaculus 2026 API facts:
- Base URL: https://www.metaculus.com/api
- Auth: `Authorization: Token <METACULUS_TOKEN>` (required even for read-only GETs as of 2026)
- Fall 2026 FutureEval Tournament ID: 33121 (`fall-futureeval-2026`, $50,000 USD pool)
- MiniBench slug: `minibench` (bi-weekly $1,000 pool)
- Sandbox slug: `bot-testing-area`
- Submit forecast: POST /api/questions/forecast/ with JSON LIST of question forecast dicts
- Post reasoning comment: POST /api/comments/create/ with `included_forecast=True`, `is_private=True`
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Optional

import httpx

from .models import ForecastResult, MetaculusQuestion, NumericScaling, SubmissionReceipt

API_BASE_URL = "https://www.metaculus.com/api"
FALL_2026_FUTUREEVAL_ID = 33121
SUMMER_2026_FUTUREEVAL_ID = 33022
MINIBENCH_SLUG = "minibench"
BOT_TESTING_AREA_SLUG = "bot-testing-area"

DEFAULT_USER_AGENT = (
    "MetaculusFutureEvalQuantBot/1.0 "
    "(https://github.com/pecopeko/metaculus-futureeval-quant-bot)"
)


class MetaculusAPIError(RuntimeError):
    """Raised when the Metaculus API returns a non-success HTTP status."""

    def __init__(self, status_code: int, message: str, url: str = "") -> None:
        super().__init__(f"Metaculus API HTTP {status_code} ({url}): {message}")
        self.status_code = status_code
        self.url = url
        self.raw_message = message


class MetaculusAuthError(MetaculusAPIError):
    """Raised when METACULUS_TOKEN is missing or rejected (HTTP 401/403)."""


def create_forecast_payload(
    question_type: str,
    binary_probability: Optional[float] = None,
    multiple_choice_probabilities: Optional[dict[str, float]] = None,
    continuous_cdf: Optional[list[float]] = None,
) -> dict[str, Any]:
    """Build the exact Metaculus `/api/questions/forecast/` inner payload dict."""
    if question_type == "binary":
        if binary_probability is None:
            raise ValueError("binary_probability is required for binary questions")
        return {
            "probability_yes": round(float(binary_probability), 6),
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        if not multiple_choice_probabilities:
            raise ValueError(
                "multiple_choice_probabilities is required for multiple_choice questions"
            )
        return {
            "probability_yes": None,
            "probability_yes_per_category": {
                str(k): round(float(v), 6)
                for k, v in multiple_choice_probabilities.items()
            },
            "continuous_cdf": None,
        }
    if question_type in ("numeric", "discrete", "date"):
        if not continuous_cdf:
            raise ValueError(
                "continuous_cdf is required for numeric/discrete/date questions"
            )
        return {
            "probability_yes": None,
            "probability_yes_per_category": None,
            "continuous_cdf": [round(float(x), 10) for x in continuous_cdf],
        }
    raise ValueError(f"Unsupported question_type: {question_type}")


def validate_forecast_payload(
    question: MetaculusQuestion,
    payload: dict[str, Any],
) -> list[str]:
    """Validate a forecast payload against strict Metaculus API submission rules."""
    errors: list[str] = []
    qtype = question.question_type

    if qtype == "binary":
        prob = payload.get("probability_yes")
        if not isinstance(prob, (int, float)) or math.isnan(prob):
            errors.append("binary probability_yes must be a finite float")
        elif not (0.001 <= float(prob) <= 0.999):
            errors.append(
                f"binary probability_yes={prob} out of Metaculus [0.001, 0.999] bounds"
            )
        if payload.get("continuous_cdf") is not None:
            errors.append("binary payload must have continuous_cdf=None")
        if payload.get("probability_yes_per_category") is not None:
            errors.append("binary payload must have probability_yes_per_category=None")

    elif qtype == "multiple_choice":
        probs = payload.get("probability_yes_per_category")
        if not isinstance(probs, dict) or not probs:
            errors.append("multiple_choice probability_yes_per_category must be non-empty dict")
        else:
            if question.options:
                expected = set(question.options)
                actual = set(probs.keys())
                if expected != actual:
                    errors.append(
                        f"multiple_choice keys mismatch: expected={sorted(expected)} actual={sorted(actual)}"
                    )
            total = 0.0
            for label, val in probs.items():
                if not isinstance(val, (int, float)) or math.isnan(val):
                    errors.append(f"non-finite probability for option '{label}'")
                elif not (0.001 <= float(val) <= 0.999):
                    errors.append(
                        f"option '{label}' probability={val} outside [0.001, 0.999]"
                    )
                else:
                    total += float(val)
            if abs(total - 1.0) > 1e-4:
                errors.append(f"multiple_choice probabilities sum to {total:.6f}, expected 1.0")

    elif qtype in ("numeric", "discrete", "date"):
        cdf = payload.get("continuous_cdf")
        expected_len = 201
        if (
            qtype == "discrete"
            and question.scaling is not None
            and question.scaling.inbound_outcome_count is not None
        ):
            expected_len = int(question.scaling.inbound_outcome_count) + 1

        if not isinstance(cdf, list):
            errors.append("continuous_cdf must be a list of floats")
        else:
            if len(cdf) != expected_len:
                errors.append(
                    f"continuous_cdf length={len(cdf)}, expected {expected_len}"
                )
            if cdf:
                if any(math.isnan(v) or v < 0.0 or v > 1.0 for v in cdf):
                    errors.append("continuous_cdf contains values outside [0.0, 1.0]")
                if not question.open_lower_bound and abs(cdf[0] - 0.0) > 1e-6:
                    errors.append(
                        f"closed lower bound requires cdf[0]==0.0, got {cdf[0]}"
                    )
                if question.open_lower_bound and cdf[0] < 0.001 - 1e-9:
                    errors.append(
                        f"open lower bound requires cdf[0]>=0.001, got {cdf[0]}"
                    )
                if not question.open_upper_bound and abs(cdf[-1] - 1.0) > 1e-6:
                    errors.append(
                        f"closed upper bound requires cdf[-1]==1.0, got {cdf[-1]}"
                    )
                if question.open_upper_bound and cdf[-1] > 0.999 + 1e-9:
                    errors.append(
                        f"open upper bound requires cdf[-1]<=0.999, got {cdf[-1]}"
                    )

                inbound_count = max(1, len(cdf) - 1)
                max_pmf = 0.2 * (200.0 / inbound_count) + 1e-6
                for i in range(len(cdf) - 1):
                    step = cdf[i + 1] - cdf[i]
                    if step < 5e-5 - 1e-12:
                        errors.append(
                            f"CDF step at index {i}->{i + 1} is {step:.8f} (< 5e-5 minimum)"
                        )
                        break
                    if step > max_pmf:
                        errors.append(
                            f"CDF step at index {i}->{i + 1} is {step:.6f} (> {max_pmf:.6f} cap)"
                        )
                        break

    return errors


def parse_post_to_questions(post: dict[str, Any]) -> list[MetaculusQuestion]:
    """Parse a Metaculus `/api/posts/` record into one or more MetaculusQuestion objects."""
    post_id = int(post["id"])
    raw_questions: list[dict[str, Any]] = []
    if isinstance(post.get("question"), dict):
        raw_questions.append(post["question"])
    elif isinstance(post.get("group_of_questions"), dict):
        sub_qs = post["group_of_questions"].get("questions") or []
        for sq in sub_qs:
            if isinstance(sq, dict):
                raw_questions.append(sq)

    parsed: list[MetaculusQuestion] = []
    for q in raw_questions:
        qtype = q.get("type")
        if qtype not in ("binary", "multiple_choice", "numeric", "discrete", "date"):
            continue
        scaling_raw = q.get("scaling")
        scaling: Optional[NumericScaling] = None
        if isinstance(scaling_raw, dict) and scaling_raw.get("range_min") is not None:
            scaling = NumericScaling(
                range_min=float(scaling_raw["range_min"]),
                range_max=float(scaling_raw["range_max"]),
                zero_point=(
                    float(scaling_raw["zero_point"])
                    if scaling_raw.get("zero_point") is not None
                    else None
                ),
                inbound_outcome_count=(
                    int(scaling_raw["inbound_outcome_count"])
                    if scaling_raw.get("inbound_outcome_count") is not None
                    else None
                ),
            )

        my_forecasts = q.get("my_forecasts") or {}
        already_forecasted = bool(my_forecasts.get("latest"))

        community_pred: Optional[float] = None
        agg = q.get("aggregations") or {}
        recency = agg.get("recency_weighted") or {}
        latest_agg = recency.get("latest") or {}
        centers = latest_agg.get("centers") or []
        if isinstance(centers, list) and centers and isinstance(centers[0], (int, float)):
            community_pred = float(centers[0])

        q_title = q.get("title") or post.get("title") or f"Question {q.get('id')}"
        q_desc = q.get("description") or post.get("description") or ""
        q_url = f"https://www.metaculus.com/questions/{post_id}/"

        options_raw = q.get("options") or []
        options = [str(opt) for opt in options_raw if opt is not None]

        parsed.append(
            MetaculusQuestion(
                question_id=int(q["id"]),
                post_id=post_id,
                title=str(q_title),
                description=str(q_desc),
                resolution_criteria=str(q.get("resolution_criteria") or ""),
                fine_print=str(q.get("fine_print") or ""),
                question_type=qtype,
                status=str(q.get("status") or post.get("status") or "open"),
                url=q_url,
                scheduled_close_time=q.get("scheduled_close_time"),
                scheduled_resolve_time=q.get("scheduled_resolve_time"),
                open_lower_bound=bool(q.get("open_lower_bound", False)),
                open_upper_bound=bool(q.get("open_upper_bound", False)),
                unit=str(q.get("unit") or ""),
                scaling=scaling,
                options=options,
                already_forecasted=already_forecasted,
                community_prediction=community_pred,
                resolution=q.get("resolution"),
            )
        )
    return parsed


class MetaculusClient:
    """Polite, rate-limit-aware HTTP client for the Metaculus v2 API."""

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = API_BASE_URL,
        timeout: float = 20.0,
        max_retries: int = 3,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.token = (token if token is not None else os.getenv("METACULUS_TOKEN", "")).strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._external_client = http_client

    @property
    def has_token(self) -> bool:
        return bool(self.token and self.token != "REPLACE_ME")

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
        }
        if self.has_token:
            headers["Authorization"] = f"Token {self.token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[Any] = None,
    ) -> httpx.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                if self._external_client is not None:
                    resp = self._external_client.request(
                        method,
                        url,
                        params=params,
                        json=json_body,
                        headers=self._headers(),
                        timeout=self.timeout,
                    )
                else:
                    with httpx.Client(
                        timeout=self.timeout,
                        follow_redirects=True,
                        headers=self._headers(),
                    ) as client:
                        resp = client.request(
                            method,
                            url,
                            params=params,
                            json=json_body,
                        )
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(1.5 * (2**attempt))
                    continue
                raise MetaculusAPIError(0, str(exc), url) from exc

            if resp.status_code in (429, 500, 502, 503, 504) and attempt + 1 < self.max_retries:
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 * (2**attempt)
                time.sleep(min(delay, 30.0))
                continue

            if resp.status_code in (401, 403):
                hint = (
                    "Metaculus requires an authenticated bot token for API endpoints. "
                    "Set METACULUS_TOKEN from https://www.metaculus.com/futureeval/participate/."
                )
                raise MetaculusAuthError(
                    resp.status_code,
                    f"{resp.text.strip()} | {hint}",
                    url,
                )

            if resp.status_code >= 400:
                raise MetaculusAPIError(resp.status_code, resp.text[:400], url)

            return resp

        raise MetaculusAPIError(0, str(last_exc or "Unknown request failure"), url)

    def check_connectivity(self) -> dict[str, Any]:
        """Probe the Metaculus API honestly without fabricating access."""
        url = f"{self.base_url}/posts/"
        try:
            resp = self._request(
                "GET",
                "/posts/",
                params={"limit": 1, "tournaments": [BOT_TESTING_AREA_SLUG]},
            )
            data = resp.json()
            return {
                "status": "authenticated_ok",
                "http_status": resp.status_code,
                "url": url,
                "sample_count": data.get("count"),
            }
        except MetaculusAuthError as exc:
            return {
                "status": "auth_required",
                "http_status": exc.status_code,
                "url": url,
                "detail": exc.raw_message,
            }
        except MetaculusAPIError as exc:
            return {
                "status": "api_error",
                "http_status": exc.status_code,
                "url": url,
                "detail": exc.raw_message,
            }

    def list_open_tournament_questions(
        self,
        tournament_id: int | str = FALL_2026_FUTUREEVAL_ID,
        limit: int = 50,
        offset: int = 0,
        skip_already_forecasted: bool = True,
    ) -> list[MetaculusQuestion]:
        """Fetch open binary/multiple_choice/numeric/discrete questions from a tournament."""
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "order_by": "-hotness",
            "forecast_type": "binary,multiple_choice,numeric,discrete",
            "tournaments": [tournament_id],
            "statuses": "open",
            "include_description": "true",
        }
        resp = self._request("GET", "/posts/", params=params)
        payload = resp.json()
        results = payload.get("results") or []
        questions: list[MetaculusQuestion] = []
        for post in results:
            for q in parse_post_to_questions(post):
                if q.status != "open":
                    continue
                if skip_already_forecasted and q.already_forecasted:
                    continue
                questions.append(q)
        return questions

    def get_post_questions(self, post_id: int) -> list[MetaculusQuestion]:
        """Fetch full details for a single Metaculus post ID."""
        resp = self._request("GET", f"/posts/{post_id}/")
        return parse_post_to_questions(resp.json())

    def list_resolved_binary_questions(
        self,
        limit: int = 100,
        offset: int = 0,
        tournament_id: Optional[int | str] = None,
    ) -> list[MetaculusQuestion]:
        """Fetch resolved binary questions directly from Metaculus API (requires token)."""
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "statuses": "resolved",
            "forecast_type": "binary",
            "order_by": "-actual_resolve_time",
            "include_description": "true",
        }
        if tournament_id is not None:
            params["tournaments"] = [tournament_id]
        resp = self._request("GET", "/posts/", params=params)
        results = resp.json().get("results") or []
        out: list[MetaculusQuestion] = []
        for post in results:
            for q in parse_post_to_questions(post):
                if q.question_type == "binary" and q.resolution in ("yes", "no", 1, 0, True, False):
                    out.append(q)
        return out

    def submit_forecast(
        self,
        question: MetaculusQuestion,
        forecast: ForecastResult,
        dry_run: bool = True,
        post_comment: bool = True,
        private_comment: bool = True,
    ) -> SubmissionReceipt:
        """Validate and optionally POST a forecast + reasoning comment to Metaculus."""
        errors = validate_forecast_payload(question, forecast.api_payload)
        forecast_wire = [
            {
                "question": question.question_id,
                "source": "api",
                **forecast.api_payload,
            }
        ]
        comment_wire = {
            "text": forecast.reasoning_markdown,
            "parent": None,
            "included_forecast": True,
            "is_private": private_comment,
            "on_post": question.post_id,
        }

        if errors:
            return SubmissionReceipt(
                question_id=question.question_id,
                post_id=question.post_id,
                dry_run=dry_run,
                submitted=False,
                validation_errors=errors,
                payload_preview={"forecast": forecast_wire[0]},
                message="Validation failed; forecast blocked before submission.",
            )

        if dry_run:
            return SubmissionReceipt(
                question_id=question.question_id,
                post_id=question.post_id,
                dry_run=True,
                submitted=False,
                payload_preview={
                    "forecast_endpoint": f"{self.base_url}/questions/forecast/",
                    "forecast_body": forecast_wire,
                    "comment_endpoint": f"{self.base_url}/comments/create/",
                    "comment_length": len(forecast.reasoning_markdown),
                    "comment_is_private": private_comment,
                },
                message="Dry-run validated OK (no HTTP POST executed).",
            )

        if not self.has_token:
            raise MetaculusAuthError(
                401,
                "Cannot submit live forecast without METACULUS_TOKEN.",
                f"{self.base_url}/questions/forecast/",
            )

        f_resp = self._request("POST", "/questions/forecast/", json_body=forecast_wire)
        c_status: Optional[int] = None
        if post_comment and forecast.reasoning_markdown.strip():
            c_resp = self._request("POST", "/comments/create/", json_body=comment_wire)
            c_status = c_resp.status_code

        return SubmissionReceipt(
            question_id=question.question_id,
            post_id=question.post_id,
            dry_run=False,
            submitted=True,
            forecast_status_code=f_resp.status_code,
            comment_status_code=c_status,
            payload_preview={"forecast_body": forecast_wire},
            message="Live forecast and reasoning comment submitted successfully.",
        )
