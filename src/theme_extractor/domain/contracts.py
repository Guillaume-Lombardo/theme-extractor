"""Domain contracts for unified topic extraction outputs."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from theme_extractor.domain.enums import BackendName, CommandName, ExtractMethod, OfflinePolicy, OutputFocus


class TopicKeyword(BaseModel):
    """Represent a keyword associated with a topic."""

    term: str
    score: float | None = None


class TopicResult(BaseModel):
    """Represent one extracted topic in the unified output schema."""

    topic_id: int
    label: str | None = None
    score: float | None = None
    keywords: list[TopicKeyword] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)


class DocumentTopicLink(BaseModel):
    """Represent a link between one document and one topic."""

    document_id: str
    topic_id: int
    probability: float | None = None
    rank: int | None = None


class ExtractionRunMetadata(BaseModel):
    """Store normalized metadata about one extraction run."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    command: CommandName
    method: ExtractMethod
    offline_policy: OfflinePolicy
    backend: BackendName | None = None
    index: str | None = None


class UnifiedExtractionOutput(BaseModel):
    """Represent normalized extraction output, topic-first by default."""

    schema_version: str = "1.0"
    focus: OutputFocus = OutputFocus.TOPICS
    topics: list[TopicResult] = Field(default_factory=list)
    document_topics: list[DocumentTopicLink] | None = None
    notes: list[str] = Field(default_factory=list)
    metadata: ExtractionRunMetadata


class BenchmarkOutput(BaseModel):
    """Represent normalized benchmark output across multiple methods."""

    schema_version: str = "1.0"
    command: CommandName = CommandName.BENCHMARK
    methods: list[ExtractMethod]
    outputs: dict[str, UnifiedExtractionOutput]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
