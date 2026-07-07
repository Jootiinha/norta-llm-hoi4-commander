.PHONY: setup run web extract-data chunk-data clean

setup:
	bash ./scripts/setup.sh

run:
	poetry run python -B src/norta_llm/run_model.py \
		--model models/qwen3-0.6b \
		--prompt "Monte uma build inicial para o Brasil em Hearts of Iron IV focada em industria e exercito."

web:
	PYTHONPATH=src poetry run python -B -m web_service.server --reload

# Exemplos:
#   # Coleta completa (script src/hoi4_wiki/collect_data.py):
#   make extract-data
#   make extract-data COLLECT_ARGS="--api-url https://hoi4.paradoxwikis.com/api.php --output data/raw/hoi4_wiki/hoi4_pages.jsonl --user-agent 'joao-hoi4-study-bot/0.1 (contato: joaocrm@outlook.com)' --workers 4 --delay 0.7 --list-delay 0.2 --limit 0 --resume"
#   make extract-data COLLECT_ARGS="--api-url https://example.com/api.php --output data/raw/hoi4_wiki/hoi4_pages.jsonl --user-agent 'meu-bot' --workers 2 --delay 1.0 --list-delay 0.4 --limit 50 --no-resume"
#   # Conversao para markdown (script src/hoi4_wiki/convert_to_markdown.py):
#   make process-data-markdown CONVERT_ARGS="--input data/raw/hoi4_wiki/hoi4_pages.jsonl --output-dir data/interim/hoi4_wiki/pages --manifest data/interim/hoi4_wiki/manifest.jsonl --min-chars 200"
#   make process-data-markdown CONVERT_ARGS="--input data/raw/hoi4_wiki/hoi4_pages.jsonl --output-dir data/interim/hoi4_wiki/pages --manifest data/interim/hoi4_wiki/manifest.jsonl --min-chars 500"
extract-data:
	poetry run python -B src/hoi4_wiki/collect_data.py $(COLLECT_ARGS)

process-data-markdown:
	poetry run python -B src/hoi4_wiki/convert_to_markdown.py $(CONVERT_ARGS)

# Exemplos:
#   make chunk-data
#   # Chunking semântico (padrão: semantic):
#   make chunk-data CHUNK_ARGS="--input-dir data/interim/hoi4_wiki/pages --output data/processed/chunks/chunks.jsonl --manifest data/processed/chunks/manifest.jsonl --max-chars 1800 --overlap-units 1 --chunking-strategy semantic --semantic-threshold 0.31 --min-chunk-sentences 3 --semantic-model sentence-transformers/all-MiniLM-L6-v2"
#   # Chunking por parágrafo (fallback semântico):
#   make chunk-data CHUNK_ARGS="--input-dir data/interim/hoi4_wiki/pages --output data/processed/chunks/chunks.jsonl --manifest data/processed/chunks/manifest.jsonl --max-chars 1800 --overlap-units 1 --chunking-strategy paragraph"
chunk-data:
	bash ./scripts/backup_chunks.sh "$(CHUNK_ARGS)"
	poetry run python -B src/hoi4_wiki/create_chunks.py $(CHUNK_ARGS)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
