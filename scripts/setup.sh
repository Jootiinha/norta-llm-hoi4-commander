#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry nao encontrado."
  echo "Instale com: curl -sSL https://install.python-poetry.org | python3 -"
  exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
else
  PYTHON_BIN="python3"
fi

PYTHON_VERSION="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

case "$PYTHON_VERSION" in
  3.11|3.12)
    ;;
  *)
    echo "Python $PYTHON_VERSION nao e suportado. Use Python 3.11 ou 3.12."
    exit 1
    ;;
esac

echo "Configurando Poetry para usar .venv no projeto..."
poetry config virtualenvs.in-project true --local

echo "Selecionando interpretador: $PYTHON_BIN"
poetry env use "$PYTHON_BIN"

echo "Instalando dependencias..."
poetry install

echo
echo "Ambiente pronto."
echo "Ative com: source .venv/bin/activate"
echo "Ou rode comandos via: poetry run <comando>"
