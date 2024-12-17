#!/usr/bin/env bash
shopt -s expand_aliases
HERE=$(dirname "$0")

. "automlbenchmark/frameworks/shared/setup.sh" "$HERE" true
export AR=/usr/bin/ar
PIP install -r "$HERE/requirements.txt"

PY -c "from alpha_automl import __version__; print(__version__)" >> "${HERE}/.installed"
