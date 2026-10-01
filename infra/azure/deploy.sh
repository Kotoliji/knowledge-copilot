#!/usr/bin/env bash
# End-to-end Azure deployment. Requires: az CLI (logged in), docker.
set -euo pipefail

RG="${RG:-rg-knowledge-copilot}"
LOCATION="${LOCATION:-swedencentral}"
NAME="${NAME:-kcopilot}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%s)}"

az group create -n "$RG" -l "$LOCATION" -o none

# 1) Registry first (needed to push the image before the app exists)
ACR=$(az deployment group create -g "$RG" -f infra/azure/main.bicep \
      -p name="$NAME" image="mcr.microsoft.com/k8se/quickstart:latest" \
      --query properties.outputs.acrLoginServer.value -o tsv)

# 2) Build + push
az acr login -n "${ACR%%.*}"
docker build -t "$ACR/knowledge-copilot:$TAG" .
docker push "$ACR/knowledge-copilot:$TAG"

# 3) Deploy the real image
az deployment group create -g "$RG" -f infra/azure/main.bicep \
   -p name="$NAME" image="$ACR/knowledge-copilot:$TAG" \
   --query properties.outputs -o json
