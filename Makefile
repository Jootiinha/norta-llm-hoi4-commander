.PHONY: setup run web convert-wiki clean

setup:
	bash ./scripts/setup.sh

run:
	poetry run python -B src/norta_llm/run_model.py \
		--model models/qwen3-0.6b \
		--prompt "Monte uma build inicial para o Brasil em Hearts of Iron IV focada em industria e exercito."

web:
	PYTHONPATH=src poetry run python -B -m web_service.server

convert-wiki:
	poetry run python -B src/hoi4_wiki/convert_to_markdown.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
