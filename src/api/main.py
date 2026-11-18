"""FastAPI application for multimodal document intelligence and grounded retrieval."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.parser.layout_parser import LayoutParser
from src.parser.models import (
    BoundingBox,
    ParsedDocument,
    RetrievalResult,
    SecurityAuditResult,
    TableBlock,
    TextSpan,
)
from src.retrieval.grounded_engine import GroundedEngine
from src.security.prompt_injection_guard import PromptInjectionGuard

app = FastAPI(
    title="Multimodal Document Intelligence API",
    description="Layout-aware document parsing, adversarial prompt injection defense, and grounded spatial RAG.",
    version="0.1.0",
)

# In-memory stores
DOCUMENT_STORE: dict[str, ParsedDocument] = {}
AUDIT_STORE: dict[str, SecurityAuditResult] = {}
ENGINE = GroundedEngine()
GUARD = PromptInjectionGuard()
PARSER = LayoutParser(security_guard=GUARD)


class ParseRequest(BaseModel):
    doc_id: str
    spans: list[TextSpan]
    tables: list[TableBlock] | None = None


class ParseResponse(BaseModel):
    document: ParsedDocument
    audit: SecurityAuditResult


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    target_page: int | None = None
    spatial_filter: BoundingBox | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "multimodal-doc-intelligence"}


@app.post("/v1/documents/parse", response_model=ParseResponse)
def parse_document(payload: ParseRequest) -> ParseResponse:
    if not payload.spans:
        raise HTTPException(status_code=400, detail="Cannot parse empty spans")

    # Step 1: Scan and sanitize for adversarial injection
    _, audit = GUARD.scan_and_sanitize(payload.doc_id, payload.spans)
    AUDIT_STORE[payload.doc_id] = audit

    # Step 2: Parse layout and construct reading blocks
    parsed = PARSER.parse_spans(payload.doc_id, payload.spans, payload.tables)
    DOCUMENT_STORE[payload.doc_id] = parsed

    # Step 3: Index in spatial engine
    ENGINE.index_parsed_document(parsed)

    return ParseResponse(document=parsed, audit=audit)


@app.post("/v1/documents/query", response_model=RetrievalResult)
def query_documents(payload: QueryRequest) -> RetrievalResult:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    return ENGINE.query(
        query_text=payload.query,
        top_k=payload.top_k,
        target_page=payload.target_page,
        spatial_filter=payload.spatial_filter,
    )


@app.get("/v1/documents/{doc_id}/audit", response_model=SecurityAuditResult)
def get_audit(doc_id: str) -> SecurityAuditResult:
    if doc_id not in AUDIT_STORE:
        raise HTTPException(status_code=404, detail=f"Audit for document '{doc_id}' not found")
    return AUDIT_STORE[doc_id]
