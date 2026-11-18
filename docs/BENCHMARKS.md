# Multimodal Document Intelligence Benchmark Results

- **Evaluation Date:** 2026-11-18
- **Hardware Platform:** Apple A18 Pro (8.0 GB RAM)
- **Operating System:** Darwin 25.4.0
- **Python Version:** 3.14.6

---

## 1. Layout Parsing & Block Grouping Throughput

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Total Spans Evaluated** | `1,000` spans | Synthetic multi-column document stream |
| **Total Pages Processed** | `20` pages | Grouped into `40` structured blocks |
| **Parsing Throughput** | **`92,887.2` spans/sec** | Reading order sort + column clustering |
| **Page Throughput** | **`1,857.7` pages/sec** | End-to-end block grouping rate |
| **Total Elapsed Time** | `0.0108` s | Local execution duration |

---

## 2. Adversarial Prompt Injection & Steganography Interception

| Attack Vector | Samples | Detected & Stripped | Interception Rate |
| :--- | :--- | :--- | :--- |
| **Micro-Font (< 2.0pt)** | 25 | 25 | 100.0% |
| **Zero-Opacity (< 0.05)** | 25 | 25 | 100.0% |
| **Low Contrast (White-on-White)** | 25 | 25 | 100.0% |
| **Adversarial Directives** | 25 | 25 | 100.0% |
| **Overall Interception Rate** | **`100`** | **`100`** | **`100.0%`** |
| **Scan Overhead per Span** | - | - | **`7.56 µs`** |

---

## 3. Grounded Spatial RAG Latency

| Percentile | Latency (ms) | Notes |
| :--- | :--- | :--- |
| **p50 (Median)** | **`1.942 ms`** | Query BM25 search + coordinate resolution |
| **p95** | **`3.007 ms`** | Multi-topic spatial coordinate lookup |
| **p99** | **`3.865 ms`** | Tail latency across `100` queries |
| **Mean** | **`2.054 ms`** | Average retrieval and citation synthesis |

---

## Reproduce Benchmarks

Run directly from repository root:
```bash
python scripts/run_doc_benchmarks.py
```
