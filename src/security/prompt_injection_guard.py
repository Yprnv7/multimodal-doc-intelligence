"""Adversarial document security guard for prompt injection and steganography defense."""

from __future__ import annotations

import re
from typing import ClassVar

from src.parser.models import (
    RiskLevel,
    SecurityAuditResult,
    SecurityViolation,
    TextSpan,
    ViolationType,
)


class PromptInjectionGuard:
    """Detects and neutralizes adversarial text spans and hidden prompt injections."""

    ADVERSARIAL_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"disregard\s+(the\s+)?system\s+prompt", re.IGNORECASE),
        re.compile(r"\[/?inst\]", re.IGNORECASE),
        re.compile(r"<\|im_start\|>|<\|im_end\|>", re.IGNORECASE),
        re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
        re.compile(r"you\s+must\s+now\s+act\s+as", re.IGNORECASE),
        re.compile(r"new\s+system\s+directive", re.IGNORECASE),
        re.compile(r"exfiltrate\s+(data|credentials|tokens)", re.IGNORECASE),
        re.compile(r"bypass\s+security\s+filters?", re.IGNORECASE),
    ]

    ZERO_WIDTH_CHARS = re.compile(r"[\u200B-\u200D\uFEFF]")

    @staticmethod
    def _hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
        clean = hex_code.lstrip("#")
        if len(clean) == 3:
            clean = "".join([c * 2 for c in clean])
        if len(clean) != 6:
            return (0, 0, 0)
        return (int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16))

    @classmethod
    def _relative_luminance(cls, r: int, g: int, b: int) -> float:
        components = []
        for c in (r, g, b):
            val = c / 255.0
            components.append(val / 12.92 if val <= 0.03928 else ((val + 0.055) / 1.055) ** 2.4)
        return 0.2126 * components[0] + 0.7152 * components[1] + 0.0722 * components[2]

    @classmethod
    def calculate_contrast_ratio(cls, fg_hex: str, bg_hex: str) -> float:
        fg_rgb = cls._hex_to_rgb(fg_hex)
        bg_rgb = cls._hex_to_rgb(bg_hex)
        l1 = cls._relative_luminance(*fg_rgb)
        l2 = cls._relative_luminance(*bg_rgb)
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)

    def scan_and_sanitize(
        self, doc_id: str, raw_spans: list[TextSpan]
    ) -> tuple[list[TextSpan], SecurityAuditResult]:
        clean_spans: list[TextSpan] = []
        violations: list[SecurityViolation] = []

        for span in raw_spans:
            is_malicious = False

            # Check 1: Invisible font size (< 2.0 pt)
            if span.font_size < 2.0:
                violations.append(
                    SecurityViolation(
                        violation_type=ViolationType.HIDDEN_FONT,
                        details=f"Micro-text font size: {span.font_size:.2f}pt (below 2.0pt limit)",
                        span_text=span.text[:80],
                        bounding_box=span.bounding_box,
                    )
                )
                is_malicious = True

            # Check 2: Invisible opacity (< 0.05)
            if span.opacity < 0.05:
                violations.append(
                    SecurityViolation(
                        violation_type=ViolationType.ZERO_OPACITY,
                        details=f"Near-zero opacity: {span.opacity:.3f} (below 0.05 threshold)",
                        span_text=span.text[:80],
                        bounding_box=span.bounding_box,
                    )
                )
                is_malicious = True

            # Check 3: Low contrast / white-on-white text
            contrast = self.calculate_contrast_ratio(span.font_color, span.background_color)
            if contrast < 1.15:
                violations.append(
                    SecurityViolation(
                        violation_type=ViolationType.LOW_CONTRAST,
                        details=(
                            f"Imperceptible contrast ratio: {contrast:.2f}:1 "
                            f"(FG: {span.font_color}, BG: {span.background_color})"
                        ),
                        span_text=span.text[:80],
                        bounding_box=span.bounding_box,
                    )
                )
                is_malicious = True

            # Check 4: Zero-width character insertion
            if self.ZERO_WIDTH_CHARS.search(span.text):
                cleaned_text = self.ZERO_WIDTH_CHARS.sub("", span.text)
                if not cleaned_text.strip():
                    is_malicious = True
                span = span.model_copy(update={"text": cleaned_text})

            # Check 5: Adversarial prompt injection signatures
            for pattern in self.ADVERSARIAL_PATTERNS:
                if pattern.search(span.text):
                    violations.append(
                        SecurityViolation(
                            violation_type=ViolationType.ADVERSARIAL_INSTRUCTION,
                            details=f"Detected adversarial directive matching rule: {pattern.pattern}",
                            span_text=span.text[:80],
                            bounding_box=span.bounding_box,
                        )
                    )
                    is_malicious = True
                    break

            if not is_malicious:
                clean_spans.append(span)

        # Classify risk level
        risk_level = RiskLevel.LOW
        if any(v.violation_type == ViolationType.ADVERSARIAL_INSTRUCTION for v in violations):
            risk_level = RiskLevel.CRITICAL
        elif len(violations) >= 3:
            risk_level = RiskLevel.HIGH
        elif len(violations) > 0:
            risk_level = RiskLevel.MEDIUM

        audit = SecurityAuditResult(
            doc_id=doc_id,
            is_clean=len(violations) == 0,
            violations=violations,
            stripped_spans_count=len(raw_spans) - len(clean_spans),
            risk_level=risk_level,
        )

        return clean_spans, audit
