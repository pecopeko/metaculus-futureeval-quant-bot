"""Typed data models for Metaculus FutureEval Quant Bot."""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

QuestionType = Literal["binary", "multiple_choice", "numeric", "discrete", "date"]


class NumericScaling(BaseModel):
    """Metaculus continuous/discrete question scale specification."""

    range_min: float
    range_max: float
    zero_point: Optional[float] = None
    inbound_outcome_count: Optional[int] = None


class MetaculusQuestion(BaseModel):
    """Normalized representation of a Metaculus tournament question."""

    question_id: int
    post_id: int
    title: str
    description: str = ""
    resolution_criteria: str = ""
    fine_print: str = ""
    question_type: QuestionType
    status: str = "open"
    url: str = ""
    scheduled_close_time: Optional[str] = None
    scheduled_resolve_time: Optional[str] = None
    open_lower_bound: bool = False
    open_upper_bound: bool = False
    unit: str = ""
    scaling: Optional[NumericScaling] = None
    options: list[str] = Field(default_factory=list)
    already_forecasted: bool = False
    community_prediction: Optional[float] = None
    resolution: Optional[Any] = None


class MarketAnchorMatch(BaseModel):
    """Verified prediction-market contract matched against a Metaculus question."""

    platform: Literal["polymarket", "manifold"]
    market_id: str
    question: str
    url: str
    raw_probability: float
    aligned_probability: float
    polarity_inverted: bool = False
    similarity_score: float
    liquidity_usd: float = 0.0
    volume_usd: float = 0.0
    close_date: Optional[str] = None
    date_alignment_score: float = 1.0
    quality_weight: float = 0.0
    match_reasons: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """Timestamped fact retrieved from a zero-API-key public evidence source."""

    source: str
    title: str
    snippet: str
    url: str
    published_at: Optional[str] = None
    relevance_score: float = 0.0


class EvidenceBundle(BaseModel):
    """Aggregated evidence packet with explicit per-source provenance status."""

    query: str
    collected_at: str
    items: list[EvidenceItem] = Field(default_factory=list)
    source_status: dict[str, str] = Field(default_factory=dict)
    macro_series: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PersonaEstimate(BaseModel):
    """Single forecasting persona output (LLM or statistical engine)."""

    persona: str
    provider: str
    model: str
    probability: Optional[float] = None
    option_probabilities: Optional[dict[str, float]] = None
    percentiles: Optional[dict[int, float]] = None
    rationale: str = ""
    error: Optional[str] = None


class ForecastResult(BaseModel):
    """Complete forecast package ready for dry-run inspection or Metaculus API submission."""

    question_id: int
    post_id: int
    question_type: QuestionType
    title: str
    binary_probability: Optional[float] = None
    multiple_choice_probabilities: Optional[dict[str, float]] = None
    continuous_cdf: Optional[list[float]] = None
    api_payload: dict[str, Any]
    reasoning_markdown: str
    base_rate_prior: Optional[float] = None
    market_anchors: list[MarketAnchorMatch] = Field(default_factory=list)
    persona_estimates: list[PersonaEstimate] = Field(default_factory=list)
    evidence_status: dict[str, str] = Field(default_factory=dict)
    mode: str
    created_at: str


class SubmissionReceipt(BaseModel):
    """Structured outcome from submitting (or dry-running) a forecast + reasoning comment."""

    question_id: int
    post_id: int
    dry_run: bool
    submitted: bool
    forecast_status_code: Optional[int] = None
    comment_status_code: Optional[int] = None
    validation_errors: list[str] = Field(default_factory=list)
    payload_preview: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
