# AGENTS.md

Orientacoes para agentes que forem trabalhar neste repositorio.

## Visao geral

Este projeto e um assistente local para estudar modelos de linguagem usando Hearts of Iron IV como dominio de exemplo. O fluxo alvo coleta conteudo da HOI4 Wiki via MediaWiki API, limpa e normaliza paginas, gera chunks com metadados, cria RAG com embeddings + Qdrant/Chroma, deriva dataset SFT de perguntas/respostas e treina adapters LoRA/QLoRA para ajustar comportamento.

A visualizacao e observabilidade das execucoes locais foram movidas para o projeto irmao `../norta-llm-lab-visualization`. Este repositorio continua focado no pipeline de dados, inferencia e RAG.

O codigo Python fica diretamente em `src/` e deve ser executado pelos caminhos dos arquivos.

## Estrutura

```text
.
├── adapters/               # adapters LoRA/QLoRA locais; nao versionar conteudo gerado
│   └── lora/
├── configs/                # configuracoes versionaveis de coleta, RAG, SFT e treino
├── data/                   # dados da HOI4 Wiki; nao versionar conteudo gerado
│   ├── raw/hoi4_wiki/      # respostas brutas da MediaWiki API
│   ├── interim/hoi4_wiki/  # paginas limpas e normalizadas
│   └── processed/chunks/   # chunks com metadados
├── datasets/               # datasets gerados localmente; nao versionar conteudo gerado
│   └── sft/
├── models/                 # modelos locais; nao versionar pesos
├── scripts/                # scripts de conveniencia
│   └── setup.sh            # configura venv e instala dependencias
├── src/                    # codigo Python do projeto
│   ├── hoi4_wiki/          # coleta, limpeza, normalizacao e chunking
│   ├── norta_llm/          # fluxo de inferencia, independente da web
│   │   └── run_model.py
│   ├── rag/                # embeddings, indexacao e recuperacao
│   ├── sft/                # geracao/validacao de dataset SFT
│   ├── training/           # LoRA/QLoRA e avaliacao
├── vectorstores/           # indices Qdrant/Chroma locais; nao versionar conteudo gerado
│   ├── chroma/
│   └── qdrant/
├── Makefile                # atalhos de execucao
├── pyproject.toml          # dependencias do Poetry
├── poetry.lock             # lockfile do Poetry
└── README.md               # documentacao para usuarios
```

## Comandos uteis

Instalar dependencias:

```bash
make setup
```

O alvo `setup` executa `scripts/setup.sh`, configura `.venv` local via Poetry e instala as dependencias.

Rodar inferencia padrao:

```bash
make run
```

Rodar a camada de visualizacao/orquestracao:

```bash
make lab-ui
```

Esse comando sobe o projeto `../norta-llm-lab-visualization`, que lista modelos a partir de `models/`, roda inferencias em processo isolado, persiste metricas em SQLite local e oferece historico e visualizacao de datasets.

Coletar dados brutos da HOI4 Wiki e converter para Markdown limpo:

```bash
make extract-data
```

Para acelerar a coleta, prefira paralelismo moderado:

```bash
make extract-data COLLECT_ARGS="--workers 4 --delay 0.2"
```

O alvo coleta `data/raw/hoi4_wiki/hoi4_pages.jsonl`, remove menu/navegacao e grava paginas em `data/interim/hoi4_wiki/pages/` com manifesto em `data/interim/hoi4_wiki/manifest.jsonl`.

Gerar chunks para RAG e SFT:

```bash
make chunk-data
```

O chunker le `data/interim/hoi4_wiki/pages/` e grava `data/processed/chunks/chunks.jsonl` com um `manifest.jsonl` por pagina.

Executar inferencia manualmente:

```bash
poetry run python -B src/norta_llm/run_model.py \
  --model models/qwen3-0.6b \
  --prompt "Monte uma build inicial para o Brasil em Hearts of Iron IV focada em industria e exercito."
```

Limpar caches Python:

```bash
make clean
```

## Regras de manutencao

- Nao versione modelos, pesos, checkpoints ou caches do Hugging Face.
- Nao versione snapshots da wiki, datasets gerados, indices vetoriais, adapters ou artefatos de treino.
- Mantenha arquivos grandes dentro de `models/`, que e ignorado pelo Git.
- Preserve `models/.gitkeep` para manter a pasta no repositorio.
- Preserve os `.gitkeep` em `data/`, `datasets/`, `vectorstores/` e `adapters/` para manter a estrutura.
- Prefira comandos via `Makefile` quando existirem.
- Ao alterar coleta/limpeza/chunking da HOI4 Wiki, trabalhe em `src/hoi4_wiki/`.
- Ao alterar RAG, embeddings, indexacao ou recuperacao, trabalhe em `src/rag/`.
- Ao alterar geracao de perguntas/respostas para SFT, trabalhe em `src/sft/`.
- Ao alterar LoRA/QLoRA, scripts de treino ou avaliacao, trabalhe em `src/training/`.
- Ao alterar o fluxo de IA, trabalhe em `src/norta_llm/`.
- Ao alterar a visualizacao, observabilidade ou a camada de orquestracao, trabalhe no projeto `../norta-llm-lab-visualization`.
- Ao criar scripts auxiliares de shell, coloque-os em `scripts/`.
- Atualize o `README.md` quando mudar comandos, estrutura ou fluxo de uso.
- Evite mudar os nomes dos diretorios de modelos sem atualizar o `Makefile`, os scripts em `scripts/` e o README.

## Validacao antes de finalizar

Para mudancas que nao carregam modelos, rode pelo menos validacoes de sintaxe/compilacao dos arquivos alterados.

Para mudancas em inferencia, valide com um modelo local pequeno quando disponivel:

```bash
make run
```

Se a validacao de inferencia nao for executada por custo de memoria, tempo ou ausencia do modelo local, registre isso na resposta final.

## Observacoes sobre modelos

Os comandos de download estao documentados no `README.md`. Modelos gated podem exigir aceite dos termos no Hugging Face e login com:

```bash
poetry run huggingface-cli login
```
