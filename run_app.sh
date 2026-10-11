#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ -f .venv/lib/openmp/libomp.dylib ]; then
    export DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/openmp${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi
exec .venv/bin/python -m streamlit run app.py "$@"
