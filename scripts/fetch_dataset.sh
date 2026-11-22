#!/usr/bin/env bash
# Fetches the APISentry dataset from Zenodo if not already present.
# Source: https://zenodo.org/records/22658802
set -euo pipefail

DATASET_DIR="data/external/apisentry"
CSV_PATH="$DATASET_DIR/dataset_processed.csv"
ZENODO_URL="https://zenodo.org/records/22658802/files/apisentry-ai-anon.zip"

if [ -f "$CSV_PATH" ]; then
    echo "Dataset already present: $CSV_PATH"
    echo "Rows: $(wc -l < "$CSV_PATH")"
    exit 0
fi

echo "Dataset not found. Fetching from Zenodo..."
mkdir -p "$DATASET_DIR"

if ! command -v curl &> /dev/null; then
    echo "ERROR: curl not found. Please install curl."
    exit 1
fi

echo "Downloading from $ZENODO_URL..."
curl -L -o /tmp/apisentry.zip "$ZENODO_URL"

echo "Extracting..."
unzip -o /tmp/apisentry.zip -d /tmp/apisentry_extract/

echo "Copying files..."
find /tmp/apisentry_extract/ -name "dataset_processed.csv" -exec cp {} "$CSV_PATH" \;
find /tmp/apisentry_extract/ -name "encoders.json" -exec cp {} "$DATASET_DIR/" \; 2>/dev/null || true
find /tmp/apisentry_extract/ -name "feature_columns.json" -exec cp {} "$DATASET_DIR/" \; 2>/dev/null || true

rm -rf /tmp/apisentry.zip /tmp/apisentry_extract/

echo "Dataset ready: $CSV_PATH"
echo "Rows: $(wc -l < "$CSV_PATH")"

