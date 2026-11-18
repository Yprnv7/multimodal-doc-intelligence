"""Layout-aware document parser with multi-column reading order and block grouping."""

from __future__ import annotations

from src.parser.models import (
    BlockType,
    BoundingBox,
    DocumentBlock,
    ParsedDocument,
    TableBlock,
    TextSpan,
)
from src.security.prompt_injection_guard import PromptInjectionGuard


class LayoutParser:
    """Parses raw text spans into structured, layout-aware document blocks."""

    def __init__(self, security_guard: PromptInjectionGuard | None = None) -> None:
        self.guard = security_guard or PromptInjectionGuard()

    def _determine_columns(self, spans: list[TextSpan]) -> bool:
        """Determines if a collection of spans adheres to a 2-column layout."""
        if len(spans) < 6:
            return False
        left_count = sum(1 for s in spans if s.bounding_box.xmax <= 520)
        right_count = sum(1 for s in spans if s.bounding_box.xmin >= 480)
        return left_count >= 3 and right_count >= 3

    def _sort_reading_order(self, spans: list[TextSpan]) -> list[TextSpan]:
        """Orders spans naturally following column and vertical progression."""
        if self._determine_columns(spans):
            left_col = [s for s in spans if s.bounding_box.xmin < 500]
            right_col = [s for s in spans if s.bounding_box.xmin >= 500]
            left_col.sort(key=lambda s: (s.bounding_box.ymin, s.bounding_box.xmin))
            right_col.sort(key=lambda s: (s.bounding_box.ymin, s.bounding_box.xmin))
            return left_col + right_col
        # Single column reading order
        return sorted(spans, key=lambda s: (s.bounding_box.ymin, s.bounding_box.xmin))

    def _classify_block_type(self, spans: list[TextSpan]) -> BlockType:
        avg_font = sum(s.font_size for s in spans) / len(spans)
        max_ymin = max(s.bounding_box.ymin for s in spans)

        if max_ymin >= 920:
            return BlockType.FOOTER
        if avg_font >= 14.0:
            return BlockType.HEADING
        return BlockType.PARAGRAPH

    def parse_spans(
        self,
        doc_id: str,
        raw_spans: list[TextSpan],
        tables: list[TableBlock] | None = None,
    ) -> ParsedDocument:
        tables = tables or []
        # Step 1: Security and injection sanitization
        clean_spans, _ = self.guard.scan_and_sanitize(doc_id, raw_spans)

        # Step 2: Separate spans by page
        pages: dict[int, list[TextSpan]] = {}
        for span in clean_spans:
            pages.setdefault(span.page_num, []).append(span)

        num_pages = max(pages.keys()) if pages else 1
        all_blocks: list[DocumentBlock] = []
        global_order = 1

        for page_num in sorted(pages.keys()):
            page_spans = pages[page_num]
            ordered_spans = self._sort_reading_order(page_spans)

            # Group adjacent spans with close vertical proximity into blocks
            current_group: list[TextSpan] = []
            for span in ordered_spans:
                if not current_group:
                    current_group.append(span)
                    continue

                prev_span = current_group[-1]
                v_dist = span.bounding_box.ymin - prev_span.bounding_box.ymax
                is_same_col = abs(span.bounding_box.xmin - prev_span.bounding_box.xmin) < 80

                # If vertically close and aligned, group into same block
                if 0 <= v_dist <= 25 and is_same_col and abs(span.font_size - prev_span.font_size) < 2.0:
                    current_group.append(span)
                else:
                    # Flush current group as a DocumentBlock
                    b_type = self._classify_block_type(current_group)
                    block_text = " ".join(s.text.strip() for s in current_group)
                    ymin = min(s.bounding_box.ymin for s in current_group)
                    xmin = min(s.bounding_box.xmin for s in current_group)
                    ymax = max(s.bounding_box.ymax for s in current_group)
                    xmax = max(s.bounding_box.xmax for s in current_group)

                    all_blocks.append(
                        DocumentBlock(
                            block_id=f"{doc_id}-p{page_num}-b{global_order}",
                            block_type=b_type,
                            text=block_text,
                            bounding_box=BoundingBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax),
                            page_num=page_num,
                            reading_order=global_order,
                        )
                    )
                    global_order += 1
                    current_group = [span]

            if current_group:
                b_type = self._classify_block_type(current_group)
                block_text = " ".join(s.text.strip() for s in current_group)
                ymin = min(s.bounding_box.ymin for s in current_group)
                xmin = min(s.bounding_box.xmin for s in current_group)
                ymax = max(s.bounding_box.ymax for s in current_group)
                xmax = max(s.bounding_box.xmax for s in current_group)

                all_blocks.append(
                    DocumentBlock(
                        block_id=f"{doc_id}-p{page_num}-b{global_order}",
                        block_type=b_type,
                        text=block_text,
                        bounding_box=BoundingBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax),
                        page_num=page_num,
                        reading_order=global_order,
                    )
                )
                global_order += 1

        return ParsedDocument(
            doc_id=doc_id,
            num_pages=num_pages,
            blocks=all_blocks,
            tables=tables,
            raw_spans_count=len(raw_spans),
            sanitized_spans_count=len(clean_spans),
        )
