"""Tests for multi-column layout reading order and table block extraction."""

from src.parser.layout_parser import LayoutParser
from src.parser.models import BlockType, BoundingBox, TableCell, TextSpan
from src.parser.table_extractor import TableExtractor


def test_table_extractor_markdown_and_bounds() -> None:
    cells = [
        TableCell(row=0, col=0, text="Metric", bounding_box=BoundingBox(ymin=100, xmin=100, ymax=130, xmax=300)),
        TableCell(row=0, col=1, text="Value", bounding_box=BoundingBox(ymin=100, xmin=310, ymax=130, xmax=500)),
        TableCell(row=1, col=0, text="Revenue", bounding_box=BoundingBox(ymin=140, xmin=100, ymax=170, xmax=300)),
        TableCell(row=1, col=1, text="$12.4M", bounding_box=BoundingBox(ymin=140, xmin=310, ymax=170, xmax=500)),
    ]

    table = TableExtractor.extract_table(cells, page_num=1)
    assert table.rows == 2
    assert table.cols == 2
    assert table.bounding_box.ymin == 100
    assert table.bounding_box.xmin == 100
    assert table.bounding_box.ymax == 170
    assert table.bounding_box.xmax == 500
    assert "| Metric | Value |" in table.markdown
    assert "| Revenue | $12.4M |" in table.markdown


def test_two_column_reading_order() -> None:
    parser = LayoutParser()
    # Left column spans (xmin < 500) and Right column spans (xmin >= 500)
    spans = [
        # Right column (top)
        TextSpan(
            text="Right col paragraph 1",
            font_size=11.0,
            bounding_box=BoundingBox(ymin=100, xmin=550, ymax=130, xmax=900),
            page_num=1,
        ),
        # Left column (top)
        TextSpan(
            text="Left col paragraph 1",
            font_size=11.0,
            bounding_box=BoundingBox(ymin=100, xmin=100, ymax=130, xmax=450),
            page_num=1,
        ),
        # Right column (bottom)
        TextSpan(
            text="Right col paragraph 2",
            font_size=11.0,
            bounding_box=BoundingBox(ymin=200, xmin=550, ymax=230, xmax=900),
            page_num=1,
        ),
        # Left column (bottom)
        TextSpan(
            text="Left col paragraph 2",
            font_size=11.0,
            bounding_box=BoundingBox(ymin=200, xmin=100, ymax=230, xmax=450),
            page_num=1,
        ),
        # Extra spans to trigger 2-column detection threshold (>= 3 per side)
        TextSpan(
            text="Left col paragraph 3",
            font_size=11.0,
            bounding_box=BoundingBox(ymin=300, xmin=100, ymax=330, xmax=450),
            page_num=1,
        ),
        TextSpan(
            text="Right col paragraph 3",
            font_size=11.0,
            bounding_box=BoundingBox(ymin=300, xmin=550, ymax=330, xmax=900),
            page_num=1,
        ),
    ]

    doc = parser.parse_spans("doc-2col", spans)
    texts = [b.text for b in doc.blocks]

    # Reading order must read left column first (1, 2, 3), then right column (1, 2, 3)
    assert texts[0] == "Left col paragraph 1"
    assert texts[1] == "Left col paragraph 2"
    assert texts[2] == "Left col paragraph 3"
    assert texts[3] == "Right col paragraph 1"
    assert texts[4] == "Right col paragraph 2"
    assert texts[5] == "Right col paragraph 3"


def test_heading_and_footer_classification() -> None:
    parser = LayoutParser()
    spans = [
        # Heading font size 18pt
        TextSpan(
            text="Executive Summary",
            font_size=18.0,
            bounding_box=BoundingBox(ymin=50, xmin=100, ymax=80, xmax=400),
            page_num=1,
        ),
        # Body text 11pt
        TextSpan(
            text="Operations grew 14% year over year.",
            font_size=11.0,
            bounding_box=BoundingBox(ymin=120, xmin=100, ymax=150, xmax=600),
            page_num=1,
        ),
        # Footer span ymin 950
        TextSpan(
            text="Page 1 of 12 - Confidential",
            font_size=9.0,
            bounding_box=BoundingBox(ymin=950, xmin=400, ymax=970, xmax=600),
            page_num=1,
        ),
    ]

    doc = parser.parse_spans("doc-classes", spans)
    types = [b.block_type for b in doc.blocks]
    assert types == [BlockType.HEADING, BlockType.PARAGRAPH, BlockType.FOOTER]
