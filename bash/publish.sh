#!/usr/bin/env bash
# Test, write the coverage badge and build the package with uv, then optionally upload it.

usage_help_exit() {
    echo "Usage: $0 <action>"
    echo "Actions: build, upload, both"
    exit 2
}

ACTION="$1"

if [[ "$ACTION" == "upload" ]]; then
    UPLOAD="true"
    BUILD="false"
elif [[ "$ACTION" == "build" ]]; then
    UPLOAD="false"
    BUILD="true"
elif [[ "$ACTION" == "both" ]]; then
    UPLOAD="true"
    BUILD="true"
else
    echo "Unknown action: ${ACTION}"
    echo
    usage_help_exit
fi

echo "Publish cullet action ${ACTION} -> build ${BUILD} upload ${UPLOAD}"

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "$BUILD" == "true" ]]; then
    # the oldest supported python, and the full extra so the torch code is covered
    uv venv --allow-existing --python 3.10 .venv
    uv pip install --python .venv -U -e ".[full,dev]" "genbadge[coverage]"
    QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest --cov --cov-report=xml
    mkdir -p docs
    .venv/bin/genbadge coverage -i .covreport/coverage.xml -o docs/coverage.svg
    rm -rf dist
    uv build
fi

if [[ "$UPLOAD" == "true" ]]; then
    uv publish
fi
