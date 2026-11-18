"""Spatial and lexical inverted index with layout-weighted ranking."""

from __future__ import annotations

import math
import re

from src.parser.models import BlockType, BoundingBox, DocumentBlock, TableBlock


class SpatialIndex:
    """Combines BM25-style term frequency scoring with 2D spatial layout weighting."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: dict[str, DocumentBlock] = {}
        self.table_blocks: dict[str, TableBlock] = {}
        self.inverted_index: dict[str, list[str]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.avg_doc_len: float = 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def index_document(self, blocks: list[DocumentBlock], tables: list[TableBlock] | None = None) -> None:
        tables = tables or []
        for block in blocks:
            self.docs[block.block_id] = block
            tokens = self._tokenize(block.text)
            self.doc_lengths[block.block_id] = len(tokens)
            for token in set(tokens):
                self.inverted_index.setdefault(token, []).append(block.block_id)

        for i, table in enumerate(tables):
            table_id = f"table-{table.page_num}-{i+1}"
            self.table_blocks[table_id] = table
            tokens = self._tokenize(table.markdown)
            self.doc_lengths[table_id] = len(tokens)
            for token in set(tokens):
                self.inverted_index.setdefault(token, []).append(table_id)

        if self.doc_lengths:
            self.avg_doc_len = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def search(
        self,
        query: str,
        top_k: int = 5,
        target_page: int | None = None,
        spatial_filter: BoundingBox | None = None,
    ) -> list[tuple[str, float, BoundingBox, int, str]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.doc_lengths:
            return []

        scores: dict[str, float] = {}
        total_docs = len(self.doc_lengths)

        for token in query_tokens:
            posting = self.inverted_index.get(token, [])
            df = len(posting)
            if df == 0:
                continue

            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id in posting:
                # Retrieve item details
                if doc_id in self.docs:
                    block = self.docs[doc_id]
                    page = block.page_num
                    bbox = block.bounding_box
                    text = block.text
                    type_boost = 1.4 if block.block_type == BlockType.HEADING else 1.0
                elif doc_id in self.table_blocks:
                    table = self.table_blocks[doc_id]
                    page = table.page_num
                    bbox = table.bounding_box
                    text = table.markdown
                    type_boost = 1.2
                else:
                    continue

                # Apply spatial filters if specified
                if target_page is not None and page != target_page:
                    continue
                if spatial_filter is not None and not bbox.intersection(spatial_filter):
                    continue

                # Compute BM25 TF
                tokens = self._tokenize(text)
                tf = tokens.count(token)
                d_len = self.doc_lengths[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (d_len / max(self.avg_doc_len, 1.0)))
                term_score = idf * ((tf * (self.k1 + 1.0)) / max(denom, 0.001)) * type_boost

                scores[doc_id] = scores.get(doc_id, 0.0) + term_score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_results:
            if doc_id in self.docs:
                b = self.docs[doc_id]
                results.append((doc_id, score, b.bounding_box, b.page_num, b.text))
            else:
                t = self.table_blocks[doc_id]
                results.append((doc_id, score, t.bounding_box, t.page_num, t.markdown))

        return results
