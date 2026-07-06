# AGENTS.md

Orientacoes para agentes que forem trabalhar neste repositorio.

## Visao geral

Este projeto e um assistente local para testar modelos de linguagem como comandante de Hearts of Iron IV, com foco em recomendacoes de builds, estrategias, prioridades de pesquisa, composicao industrial e acompanhamento de recursos de CPU/GPU durante inferencias.

O codigo Python fica diretamente em `src/` e deve ser executado pelos caminhos dos arquivos.

## Estrutura

```text
.
├── models/                 # modelos locais; nao versionar pesos
├── scripts/                # scripts de conveniencia
│   └── setup.sh            # configura venv e instala dependencias
├── src/                    # codigo Python do projeto
│   ├── norta_llm/          # fluxo de inferencia, independente da web
│   │   └── run_model.py
│   ├── web_service/        # servidor e interface web
│   │   ├── server.py
│   │   └── metrics.py
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

Rodar interface web de acompanhamento:

```bash
make web
```

A interface web e o fluxo principal para acompanhar historico, min/max/media e execucoes. Ela lista modelos a partir de `models/`, roda inferencias em processo isolado, permite ligar/desligar metricas em tempo real e salva a saida completa em `logs/runs/<run_id>/console.log`.

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
- Mantenha arquivos grandes dentro de `models/`, que e ignorado pelo Git.
- Preserve `models/.gitkeep` para manter a pasta no repositorio.
- Prefira comandos via `Makefile` quando existirem.
- Ao alterar o fluxo de IA, trabalhe em `src/norta_llm/`.
- Ao alterar a interface/servidor web ou a coleta de metricas da web, trabalhe em `src/web_service/`.
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
