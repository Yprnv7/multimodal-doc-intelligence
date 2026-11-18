"""Structured table extraction and Markdown serialization from spatial grid cells."""

from __future__ import annotations

from src.parser.models import BoundingBox, TableBlock, TableCell


class TableExtractor:
    """Reconstructs tabular structures and serializes them into aligned Markdown tables."""

    @staticmethod
    def extract_table(cells: list[TableCell], page_num: int) -> TableBlock:
        if not cells:
            raise ValueError("Cannot extract table from an empty cell collection")

        max_row = max(c.row for c in cells)
        max_col = max(c.col for c in cells)
        num_rows = max_row + 1
        num_cols = max_col + 1

        # Build grid
        grid: list[list[str]] = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        for cell in cells:
            grid[cell.row][cell.col] = cell.text.strip().replace("\n", " ")

        # Compute table bounding box enclosing all cells
        ymin = min(c.bounding_box.ymin for c in cells)
        xmin = min(c.bounding_box.xmin for c in cells)
        ymax = max(c.bounding_box.ymax for c in cells)
        xmax = max(c.bounding_box.xmax for c in cells)
        table_bbox = BoundingBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)

        # Build Markdown representation
        md_lines: list[str] = []
        header = grid[0]
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("| " + " | ".join(["---"] * num_cols) + " |")

        for row in grid[1:]:
            md_lines.append("| " + " | ".join(row) + " |")

        markdown = "\n".join(md_lines)

        return TableBlock(
            rows=num_rows,
            cols=num_cols,
            cells=cells,
            markdown=markdown,
            bounding_box=table_bbox,
            page_num=page_num,
        )
