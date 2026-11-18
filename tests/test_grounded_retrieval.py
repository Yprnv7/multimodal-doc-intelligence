"""Tests for grounded spatial retrieval, BM25 ranking, and coordinate citations."""

from src.parser.models import BlockType, BoundingBox, DocumentBlock
from src.retrieval.grounded_engine import GroundedEngine
from src.retrieval.spatial_index import SpatialIndex


def test_spatial_index_search_and_ranking() -> None:
    index = SpatialIndex()
    blocks = [
        DocumentBlock(
            block_id="b1",
            block_type=BlockType.HEADING,
            text="Cloud Infrastructure Architecture",
            bounding_box=BoundingBox(ymin=100, xmin=100, ymax=140, xmax=500),
            page_num=1,
            reading_order=1,
        ),
        DocumentBlock(
            block_id="b2",
            block_type=BlockType.PARAGRAPH,
            text="Kubernetes clusters host our microservices in Dublin region.",
            bounding_box=BoundingBox(ymin=160, xmin=100, ymax=200, xmax=600),
            page_num=1,
            reading_order=2,
        ),
        DocumentBlock(
            block_id="b3",
            block_type=BlockType.PARAGRAPH,
            text="Marketing team launched customer acquisition campaigns.",
            bounding_box=BoundingBox(ymin=100, xmin=100, ymax=140, xmax=500),
            page_num=2,
            reading_order=3,
        ),
    ]

    index.index_document(blocks)
    results = index.search(query="Kubernetes microservices", top_k=2)

    assert len(results) > 0
    top_id, score, _bbox, page, text = results[0]
    assert top_id == "b2"
    assert page == 1
    assert "Kubernetes" in text
    assert score > 0.0


def test_spatial_viewport_filtering() -> None:
    index = SpatialIndex()
    blocks = [
        DocumentBlock(
            block_id="top-block",
            block_type=BlockType.PARAGRAPH,
            text="Database connection pooling configurations.",
            bounding_box=BoundingBox(ymin=100, xmin=100, ymax=200, xmax=400),
            page_num=1,
            reading_order=1,
        ),
        DocumentBlock(
            block_id="bottom-block",
            block_type=BlockType.PARAGRAPH,
            text="Database replica failover strategy.",
            bounding_box=BoundingBox(ymin=700, xmin=100, ymax=800, xmax=400),
            page_num=1,
            reading_order=2,
        ),
    ]
    index.index_document(blocks)

    # Search with spatial filter restricting to top half (ymin 0-300)
    top_filter = BoundingBox(ymin=0, xmin=0, ymax=300, xmax=1000)
    results = index.search(query="Database", spatial_filter=top_filter)

    assert len(results) == 1
    assert results[0][0] == "top-block"


def test_grounded_engine_coordinate_citations() -> None:
    index = SpatialIndex()
    blocks = [
        DocumentBlock(
            block_id="b-audit",
            block_type=BlockType.PARAGRAPH,
            text="Net revenue increased by 28% to 45.2 million euros in 2025.",
            bounding_box=BoundingBox(ymin=250, xmin=150, ymax=290, xmax=650),
            page_num=3,
            reading_order=5,
        )
    ]
    index.index_document(blocks)
    engine = GroundedEngine(index=index)

    result = engine.query(query_text="Net revenue 2025", top_k=1)
    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.page_num == 3
    assert citation.bounding_box.ymin == 250
    assert citation.bounding_box.xmin == 150
    assert citation.bounding_box.ymax == 290
    assert citation.bounding_box.xmax == 650
    assert "[p3:(250,150)-(290,650)]" in result.answer
    assert result.latency_ms > 0.0
