"""Data models for layout-aware document intelligence and grounded citations."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CAPTION = "caption"
    FOOTER = "footer"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ViolationType(str, Enum):
    HIDDEN_FONT = "hidden_font"
    ZERO_OPACITY = "zero_opacity"
    LOW_CONTRAST = "low_contrast"
    VIEWPORT_OVERFLOW = "viewport_overflow"
    ADVERSARIAL_INSTRUCTION = "adversarial_instruction"


class BoundingBox(BaseModel):
    """Normalized spatial coordinates on a 0-1000 integer grid [ymin, xmin, ymax, xmax]."""

    ymin: int = Field(ge=0, le=1000)
    xmin: int = Field(ge=0, le=1000)
    ymax: int = Field(ge=0, le=1000)
    xmax: int = Field(ge=0, le=1000)

    @model_validator(mode="after")
    def validate_coordinates(self) -> BoundingBox:
        if self.ymin >= self.ymax:
            raise ValueError(f"ymin ({self.ymin}) must be strictly less than ymax ({self.ymax})")
        if self.xmin >= self.xmax:
            raise ValueError(f"xmin ({self.xmin}) must be strictly less than xmax ({self.xmax})")
        return self

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def area(self) -> int:
        return self.width * self.height

    def intersection(self, other: BoundingBox) -> BoundingBox | None:
        ymin = max(self.ymin, other.ymin)
        xmin = max(self.xmin, other.xmin)
        ymax = min(self.ymax, other.ymax)
        xmax = min(self.xmax, other.xmax)
        if ymin < ymax and xmin < xmax:
            return BoundingBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)
        return None

    def iou(self, other: BoundingBox) -> float:
        inter = self.intersection(other)
        if not inter:
            return 0.0
        inter_area = inter.area
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0


class TextSpan(BaseModel):
    text: str
    font_size: float = Field(ge=0.0)
    font_color: str = "#000000"
    background_color: str = "#ffffff"
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    bounding_box: BoundingBox
    page_num: int = Field(ge=1)


class TableCell(BaseModel):
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str
    bounding_box: BoundingBox


class TableBlock(BaseModel):
    rows: int
    cols: int
    cells: list[TableCell]
    markdown: str
    bounding_box: BoundingBox
    page_num: int


class DocumentBlock(BaseModel):
    block_id: str
    block_type: BlockType
    text: str
    bounding_box: BoundingBox
    page_num: int
    reading_order: int


class ParsedDocument(BaseModel):
    doc_id: str
    num_pages: int
    blocks: list[DocumentBlock]
    tables: list[TableBlock]
    raw_spans_count: int
    sanitized_spans_count: int


class SecurityViolation(BaseModel):
    violation_type: ViolationType
    details: str
    span_text: str
    bounding_box: BoundingBox | None = None


class SecurityAuditResult(BaseModel):
    doc_id: str
    is_clean: bool
    violations: list[SecurityViolation]
    stripped_spans_count: int
    risk_level: RiskLevel


class GroundedCitation(BaseModel):
    citation_id: str
    doc_id: str
    page_num: int
    bounding_box: BoundingBox
    supporting_text: str
    relevance_score: float


class RetrievalResult(BaseModel):
    query: str
    answer: str
    citations: list[GroundedCitation]
    latency_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)
