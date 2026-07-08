docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/vectorstores/qdrant:/qdrant/storage qdrant/qdrant

# sudo chown -R joaocrm /home/joaocrm/projects/norta-llm-hoi4-commander/vectorstores/qdrant_storage