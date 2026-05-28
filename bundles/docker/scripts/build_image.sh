#!/usr/bin/env bash
CURRENT_DIR=$(dirname "$0")

cd "$CURRENT_DIR"/../../.. || exit

docker build -t color-transfer-function-designer -f ./bundles/docker/Dockerfile .
