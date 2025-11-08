.PHONY: install lint format type-check clean build publish-test publish

install:
	uv sync

lint:
	uv run pre-commit run --all-files

format:
	uv run black cyclops tests examples
	uv run ruff check --fix cyclops tests examples

type-check:
	uv run mypy cyclops

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info

build:
	uv build

publish-test:
	uv build
	uv publish --publish-url https://test.pypi.org/legacy/

publish:
	uv build
	uv publish
