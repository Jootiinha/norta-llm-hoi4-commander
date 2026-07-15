.PHONY: setup run extract-data chunk-data profile-chunk-data profile-command web-rag clean

METRICS_DIR ?= metrics
MONITOR_INTERVAL ?= 1.0
WEB_HOST ?= 127.0.0.1
WEB_PORT ?= 7860

setup:
	bash ./scripts/setup.sh

run:
	poetry run python -B src/norta_llm/run_model.py \
		--model models/qwen3-0.6b \
		--prompt "Monte uma build inicial para o Brasil em Hearts of Iron IV focada em industria e exercito."

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
#   make chunk-data CHUNK_ARGS="--input-dir data/interim/hoi4_wiki/pages --output data/processed/chunks/chunks.jsonl --manifest data/processed/chunks/manifest.jsonl --max-chars 1800 --overlap-units 1 --semantic-model BAAI/bge-m3 --semantic-threshold 0.31 --min-chunk-units 3 --structured-group-lines 8 --min-unit-chars 12 --device cuda --batch-size 128"
chunk-data:
	bash ./scripts/backup_chunks.sh "$(CHUNK_ARGS)"
	poetry run python -B src/hoi4_wiki/create_chunks.py $(CHUNK_ARGS)

# Exemplos:
#   make profile-chunk-data
#   make profile-chunk-data MONITOR_INTERVAL=0.5 CHUNK_ARGS="--semantic-model sentence-transformers/all-MiniLM-L6-v2 --max-chars 2400 --overlap-units 1 --semantic-threshold 0.31 --min-chunk-units 3 --structured-group-lines 8 --min-unit-chars 12 --device cuda --batch-size 32"
#	tail -f metrics/20260713-230032-768197-chunk-data/stderr.log
profile-chunk-data:
	bash ./scripts/backup_chunks.sh "$(CHUNK_ARGS)"
	python3 -B scripts/monitor_command.py --name chunk-data --output-dir $(METRICS_DIR) --interval $(MONITOR_INTERVAL) -- poetry run python -B src/hoi4_wiki/create_chunks.py $(CHUNK_ARGS)

# Exemplo:
#   make profile-command COMMAND="poetry run python -B src/rag/index_hoi4_qdrant.py"
profile-command:
	python3 -B scripts/monitor_command.py --name command --output-dir $(METRICS_DIR) --interval $(MONITOR_INTERVAL) -- $(COMMAND)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

# make index-data INDEX_ARGS="--device cuda --encode-batch-size 256 --upload-batch-size 256 --max-seq-length 512"
# make index-data INDEX_ARGS="--model intfloat/multilingual-e5-small --device cuda --encode-batch-size 256 --upload-batch-size 512 --max-seq-length 512"
# make index-data INDEX_ARGS="--model intfloat/multilingual-e5-large --device cuda --encode-batch-size 32 --upload-batch-size 128 --max-seq-length 512"
index-data:
	poetry run python -B src/rag/index_hoi4_qdrant.py $(INDEX_ARGS)

web-rag:
	poetry run python -B src/rag/web_hoi4_rag.py --host $(WEB_HOST) --port $(WEB_PORT)
