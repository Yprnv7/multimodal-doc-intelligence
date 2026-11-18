"""Grounded retrieval engine with pixel coordinate citations and relevance verification."""

from __future__ import annotations

import time

from src.parser.models import (
    BoundingBox,
    GroundedCitation,
    ParsedDocument,
    RetrievalResult,
)
from src.retrieval.spatial_index import SpatialIndex


class GroundedEngine:
    """Answers queries with exact spatial bounding-box citations for multimodal verification."""

    def __init__(self, index: SpatialIndex | None = None) -> None:
        self.index = index or SpatialIndex()

    def index_parsed_document(self, doc: ParsedDocument) -> None:
        self.index.index_document(doc.blocks, doc.tables)

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        target_page: int | None = None,
        spatial_filter: BoundingBox | None = None,
    ) -> RetrievalResult:
        start_time = time.perf_counter()

        matches = self.index.search(
            query=query_text,
            top_k=top_k,
            target_page=target_page,
            spatial_filter=spatial_filter,
        )

        citations: list[GroundedCitation] = []
        synthesized_answers: list[str] = []

        for i, (item_id, score, bbox, page, text) in enumerate(matches):
            citation_id = f"cite-{i+1}"
            citations.append(
                GroundedCitation(
                    citation_id=citation_id,
                    doc_id=item_id,
                    page_num=page,
                    bounding_box=bbox,
                    supporting_text=text[:200],
                    relevance_score=round(score, 4),
                )
            )
            # Create citation snippet with coordinate tag
            coord_tag = f"[p{page}:({bbox.ymin},{bbox.xmin})-({bbox.ymax},{bbox.xmax})]"
            synthesized_answers.append(f"{text[:120]}... {coord_tag}")

        if synthesized_answers:
            final_answer = " ".join(synthesized_answers)
        else:
            final_answer = "No matching grounded evidence located within the document index."

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResult(
            query=query_text,
            answer=final_answer,
            citations=citations,
            latency_ms=round(latency_ms, 3),
            metadata={"num_matches": len(matches)},
        )
