.PHONY: up test bench lint clean

up:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest --cov=src --cov-report=term-missing tests/

bench:
	python scripts/run_doc_benchmarks.py

lint:
	ruff check .
	mypy src

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
