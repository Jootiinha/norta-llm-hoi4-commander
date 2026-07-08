# norta-llm-hoi4-commander

Assistente local para experimentar modelos de linguagem como comandante de Hearts of Iron IV, com foco em recomendacoes de builds, estrategias, prioridades de pesquisa, composicao industrial e acompanhamento de inferencia.

O projeto usa `transformers`, mantem os pesos dos modelos fora do Git e expoe os fluxos de inferencia e RAG para serem operados pela camada separada `norta-llm-lab-visualization`.

## Arquitetura alvo

```text
HOI4 Wiki
   ↓
Coleta via MediaWiki API
   ↓
Limpeza + normalizacao
   ↓
Chunks com metadados
   ↓
1. RAG
   embeddings + Qdrant/Chroma
   ↓
2. Dataset SFT
   perguntas/respostas
   ↓
3. LoRA/QLoRA
   ajuste de comportamento
```

A visualizacao e observabilidade das execucoes locais agora vivem em um projeto separado. Este repositorio continua responsavel pelo pipeline de dados, modelos e comandos de inferencia/RAG.

## Requisitos

- Python 3.11 ou 3.12
- Poetry
- Git LFS, recomendado para baixar repositorios do Hugging Face
- GPU NVIDIA com `nvidia-smi`, opcional, mas recomendada para modelos maiores
- Conta/token do Hugging Face para modelos gated, quando aplicavel

## Instalacao

```bash
make setup
```

O setup configura o Poetry para criar a virtualenv em `.venv/`, seleciona Python 3.12 ou 3.11 e instala as dependencias do `pyproject.toml`.

Se for baixar modelos via `git clone`, instale e habilite o Git LFS:

```bash
git lfs install
```

Se algum modelo exigir autenticacao:

```bash
poetry run huggingface-cli login
```

## Estrutura do projeto

```text
.
├── adapters/               # adapters LoRA/QLoRA gerados localmente, ignorados pelo Git
│   └── lora/
├── configs/                # configuracoes versionaveis de coleta, RAG, SFT e treino
├── data/                   # dados derivados da HOI4 Wiki, ignorados pelo Git
│   ├── raw/hoi4_wiki/      # respostas brutas da MediaWiki API
│   ├── interim/hoi4_wiki/  # paginas limpas e normalizadas
│   └── processed/chunks/   # chunks com metadados prontos para RAG/SFT
├── datasets/               # datasets gerados localmente
│   └── sft/                # perguntas/respostas para fine-tuning supervisionado
├── models/                 # modelos baixados localmente, ignorados pelo Git
├── scripts/                # scripts de conveniencia
│   └── setup.sh            # configura venv e instala dependencias
├── src/                    # codigo Python do projeto
│   ├── hoi4_wiki/          # coleta MediaWiki, limpeza, normalizacao e chunking
│   ├── norta_llm/          # fluxo de inferencia, independente da web
│   │   └── run_model.py
│   ├── rag/                # embeddings, indexacao e recuperacao
│   ├── sft/                # geracao/validacao de dataset SFT
│   ├── training/           # treino LoRA/QLoRA e avaliacao
├── vectorstores/           # indices locais Qdrant/Chroma, ignorados pelo Git
│   ├── chroma/
│   └── qdrant/
├── Makefile                # atalhos de execucao
├── pyproject.toml          # dependencias e configuracao do Poetry
└── README.md               # documentacao do projeto
```

## Convenções de artefatos

- `data/raw/hoi4_wiki/`: snapshots brutos da MediaWiki API. Use para reprocessar sem baixar tudo novamente.
- `data/interim/hoi4_wiki/`: texto limpo, normalizado e ainda proximo da estrutura original das paginas.
- `data/processed/chunks/`: chunks com metadados, como pagina, secao, URL, revisao, idioma e versao do jogo quando disponivel.
- `vectorstores/qdrant/` e `vectorstores/chroma/`: indices vetoriais locais para RAG.
- `datasets/sft/`: datasets de perguntas/respostas derivados dos chunks e de curadoria manual.
- `adapters/lora/`: adapters LoRA/QLoRA treinados localmente.
- `models/`: modelos base baixados do Hugging Face.

Esses diretorios guardam dados e artefatos potencialmente grandes. O Git versiona apenas a estrutura com `.gitkeep`; os conteudos gerados ficam locais.

## Baixar modelos

Os modelos devem ficar em `models/`. Esse diretorio e ignorado pelo Git para evitar versionar arquivos grandes como `.safetensors`, checkpoints e estados de treino.

### Opcao recomendada: huggingface-cli

```bash
mkdir -p models

poetry run huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir models/qwen3-0.6b \
  --local-dir-use-symlinks False

poetry run huggingface-cli download CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it \
  --local-dir models/gemma-3-gaia-ptbr-4b-it \
  --local-dir-use-symlinks False

poetry run huggingface-cli download amadeusai/AV-FI-Qwen2.5-7B-PT-BR-Instruct \
  --local-dir models/amadeus-verbo-qwen2.5-7b-ptbr-instruct \
  --local-dir-use-symlinks False

poetry run huggingface-cli download baidu/Unlimited-OCR \
  --local-dir models/unlimited-ocr \
  --local-dir-use-symlinks False
```

### Opcao alternativa: git clone

```bash
mkdir -p models

git clone https://huggingface.co/Qwen/Qwen3-0.6B \
  models/qwen3-0.6b

git clone https://huggingface.co/CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it \
  models/gemma-3-gaia-ptbr-4b-it

git clone https://huggingface.co/amadeusai/AV-FI-Qwen2.5-7B-PT-BR-Instruct \
  models/amadeus-verbo-qwen2.5-7b-ptbr-instruct

git clone https://huggingface.co/baidu/Unlimited-OCR \
  models/unlimited-ocr
```

Links de referencia:

- https://huggingface.co/Qwen/Qwen3-0.6B
- https://huggingface.co/CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it
- https://huggingface.co/amadeusai/AV-FI-Qwen2.5-7B-PT-BR-Instruct
- https://huggingface.co/baidu/Unlimited-OCR

## Rodar inferencia

O script padrao usa `models/qwen3-0.6b`:

```bash
make run
```

## Preparar dados da HOI4 Wiki

Para coletar dados brutos em `data/raw/hoi4_wiki/hoi4_pages.jsonl` e converter para Markdown limpo:

```bash
make extract-data
```

O coletor suporta retomada automatica. Se o arquivo de saida ja existir, paginas ja presentes sao puladas por padrao. Para uma coleta mais rapida, use poucos workers e reduza o intervalo com cuidado:

```bash
make extract-data COLLECT_ARGS="--workers 4 --delay 0.2"
```

Para testar em poucas paginas:

```bash
make extract-data COLLECT_ARGS="--limit 20"
```

O conversor remove itens recorrentes de menu e navegacao da wiki, como tabela de conteudo, navboxes, secoes de ferramentas, referencias e imagens de navegacao. A saida fica em `data/interim/hoi4_wiki/pages/`, com um `manifest.jsonl` em `data/interim/hoi4_wiki/`.

Para transformar as paginas limpas em chunks para RAG e SFT:

```bash
make chunk-data
```

O chunker le `data/interim/hoi4_wiki/pages/`, produz `data/processed/chunks/chunks.jsonl` e grava um `manifest.jsonl` com contagem de chunks por pagina. Para mudar o tamanho dos chunks:

```bash
make chunk-data CHUNK_ARGS="--max-chars 1600 --overlap-units 1"
```

Antes de cada nova execucao de `make chunk-data`, o projeto salva automaticamente um snapshot dos artefatos atuais em `data/processed/chunks/history/<timestamp>/`. Quando existirem, ele copia `chunks.jsonl`, `manifest.jsonl` e `summary.json`, e grava um `backup.env` com a data e os `CHUNK_ARGS` usados na rodada anterior. Isso facilita comparar iteracoes de chunking sem perder o estado anterior.

## Indexar no Qdrant

Para indexar os chunks no Qdrant com o embedder padrao:

```bash
make index-data
```

O payload salvo no Qdrant guarda apenas metadados do chunk e uma referencia por `chunk_id`. O texto completo continua em `data/processed/chunks/chunks.jsonl`, reduzindo o tamanho do indice e o volume de escrita no banco.

O indexador usa `BAAI/bge-m3` por padrao, que prioriza qualidade multilíngue, mas pode ficar lento em colecoes grandes. Para acelerar, reduza o comprimento maximo processado por chunk, aumente o `batch-size` conforme sua VRAM e, se necessario, troque para um embedder menor:

```bash
make index-data INDEX_ARGS="--device cuda --batch-size 128 --max-seq-length 1024"
make index-data INDEX_ARGS="--model intfloat/multilingual-e5-small --device cuda --batch-size 256 --max-seq-length 512"
```

Se voce trocar o modelo de embedding na indexacao, use o mesmo modelo na consulta:

```bash
poetry run python -B src/rag/ask_hoi4_rag.py \
  --model models/qwen3-0.6b \
  --question "Quais focos iniciais ajudam o Brasil a industrializar?" \
  --embedding-model intfloat/multilingual-e5-small
```

Se os chunks estiverem em outro caminho, a consulta precisa apontar para o mesmo arquivo usado na indexacao:

```bash
poetry run python -B src/rag/ask_hoi4_rag.py \
  --model models/qwen3-0.6b \
  --question "Quais focos iniciais ajudam o Brasil a industrializar?" \
  --chunks-path data/processed/chunks/chunks.jsonl
```

## Visualizacao e operacao

Para acompanhar execucoes, historico, metricas em tempo real e visualizacoes de dados, use o projeto separado:

```bash
make lab-ui
```

O comando sobe `../norta-llm-lab-visualization`, que funciona como camada de frontend/BFF e persiste metricas em SQLite local.

Depois abra:

```text
http://127.0.0.1:8100
```

Para escolher outro modelo ou prompt:

```bash
poetry run python -B src/norta_llm/run_model.py \
  --model models/gemma-3-gaia-ptbr-4b-it \
  --prompt "Monte uma build inicial para o Brasil em Hearts of Iron IV focada em industria e exercito." \
  --max-new-tokens 300
```

## Limpeza

Remove diretorios `__pycache__`:

```bash
make clean
```

## Observacoes

- Nao commit arquivos dentro de `models/`; baixe os modelos novamente quando preparar outro ambiente.
- Modelos grandes podem exigir bastante VRAM/RAM. Comece pelo `Qwen/Qwen3-0.6B` se quiser validar o ambiente.
- Para modelos gated, aceite os termos no Hugging Face e rode `poetry run huggingface-cli login` antes do download.
