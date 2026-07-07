# norta-llm-hoi4-commander

Assistente local para experimentar modelos de linguagem como comandante de Hearts of Iron IV, com foco em recomendacoes de builds, estrategias, prioridades de pesquisa, composicao industrial e acompanhamento de inferencia.

O projeto usa `transformers`, mantem os pesos dos modelos fora do Git e oferece uma interface web para rodar prompts, acompanhar historico e monitorar uso de CPU/GPU durante as execucoes.

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

O monitor web continua sendo o fluxo de acompanhamento das execucoes locais. Ele deve permanecer separado do pipeline de dados, mas reutilizar os modelos, adapters e indices gerados quando esses componentes forem implementados.

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
│   ├── web_service/        # servidor e interface web
│   │   ├── server.py
│   │   └── metrics.py
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

## Interface web

Para acompanhar execucoes, historico, metricas em tempo real e min/max/media por execucao:

```bash
make web
```

Depois abra:

```text
http://127.0.0.1:8000
```

O servidor web agora roda com Flask em modo de autoreload por padrao nesse comando. Ao salvar mudancas em `src/`, o processo reinicia automaticamente para refletir as alteracoes na pagina. Se quiser desabilitar isso, rode:

```bash
PYTHONPATH=src poetry run python -B -m web_service.server --no-reload
```

A interface web lista automaticamente os modelos encontrados em `models/`, permite iniciar execucoes, selecionar execucoes anteriores e acompanhar CPU, RAM, GPU, VRAM e uso do processo. Cada inferencia roda como processo isolado em uma sessao propria, separada do servidor web.

No formulario da web, voce pode ligar ou desligar metricas em tempo real e escolher o intervalo de coleta. Use intervalos maiores, como 5 segundos, quando quiser reduzir a interferencia do monitoramento.

Os dados de cada execucao ficam em `logs/runs/<run_id>/`:

- `console.log`: saida completa do programa
- `metrics.jsonl`: amostras de metricas
- `summary.json`: resumo com minimo, maximo e media
- `run.env`: parametros usados na execucao
- `status.txt`: estado final ou atual

O console nao e exibido na interface web; ele fica salvo em `console.log`.

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
