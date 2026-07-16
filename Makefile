.PHONY: setup extract-data process-data-markdown chunk-data ask-rag index-data web-rag clean

setup:
	bash ./scripts/setup.sh

extract-data:
	poetry run python -B src/hoi4_wiki/collect_data.py

process-data-markdown:
	poetry run python -B src/hoi4_wiki/convert_to_markdown.py

chunk-data:
	bash ./scripts/backup_chunks.sh
	poetry run python -B -m src.rag.chunking.cli

ask-rag:
	poetry run python -B -m src.rag.questions.cli

index-data:
	poetry run python -B -m src.rag.indexing.cli

web-rag:
	poetry run python -B src/rag/web_hoi4_rag.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
