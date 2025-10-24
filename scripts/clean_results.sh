#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
RESULTS_DIR="$ROOT_DIR/results"
WIPE_SUMMARY=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --wipe-summary   Remove results/summary.csv in addition to run artifacts.
  --help           Show this help message.

This script deletes generated run artifacts under results/ (logs, metrics,
sampling outputs, Taguchi reports, etc.). Use --wipe-summary if you also want
to clear the summary ledger before a fresh experiment batch.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wipe-summary)
      WIPE_SUMMARY=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "results/ directory not found (looked in $RESULTS_DIR). Nothing to clean."
  exit 0
fi

echo "Cleaning run artifacts under $RESULTS_DIR ..."

# Remove subdirectories (logs, metrics, smoke runs, etc.)
find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

# Remove common report files
rm -f "$RESULTS_DIR/taguchi_report.csv"

if [[ $WIPE_SUMMARY -eq 1 ]]; then
  rm -f "$RESULTS_DIR/summary.csv"
  echo "Removed summary.csv."
else
  echo "Preserved summary.csv (use --wipe-summary to remove it)."
fi

echo "Done."
