#!/bin/bash

set -xeuo pipefail

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${VENV_DIR:-gemma.venv}"
PYTHON="${PYTHON:-"$(which python)"}"

echo "Setting up venv dir: ${VENV_DIR}"

${PYTHON} -m venv "${VENV_DIR}" || echo "Could not create venv."
source "${VENV_DIR}/bin/activate" || echo "Could not activate venv"

# Upgrade pip and install requirements. 'python' is used here in order to
# reference to the python executable from the venv.
python -m pip install --upgrade pip || echo "Could not upgrade pip"

# Get iree-ir-tool for IR postprocessing.
#python -m pip install \
#  --find-links https://iree.dev/pip-release-links.html \
#  iree-compiler

python -m pip install --upgrade -r "${TD}/requirements.txt"

echo "Activate venv with:"
echo "  source ${VENV_DIR}/bin/activate"
