.PHONY: setup run web extract-data clean

setup:
	bash ./scripts/setup.sh

run:
	poetry run python -B src/norta_llm/run_model.py \
		--model models/qwen3-0.6b \
		--prompt "Monte uma build inicial para o Brasil em Hearts of Iron IV focada em industria e exercito."

web:
	PYTHONPATH=src poetry run python -B -m web_service.server

# Exemplos:
#   make extract-data
#   make extract-data COLLECT_ARGS="--workers 4 --delay 0.2"
#   make extract-data COLLECT_ARGS="--limit 20"
extract-data:
	poetry run python -B src/hoi4_wiki/collect_data.py $(COLLECT_ARGS)
	poetry run python -B src/hoi4_wiki/convert_to_markdown.py $(CONVERT_ARGS)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
