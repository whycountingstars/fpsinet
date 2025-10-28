#!/usr/bin/env bash
# create_fpsinet_zip.sh
# Usage: ./create_fpsinet_zip.sh [output_filename]
# Default output filename: fpsinet.zip

set -e

OUT=${1:-fpsinet.zip}
ROOT=$(pwd)

echo "Creating zip: ${OUT} from ${ROOT}"

# Common excludes
EXCLUDES=(
  "*.git*"
  "venv/*"
  "env/*"
  "__pycache__/*"
  "checkpoints/*"
  "outputs/*"
  "vis/*"
  "*.pth"
  "*.pt"
  ".DS_Store"
)

if [ -d ".git" ]; then
  echo "Found .git. Using git archive to create zip (clean archive, no untracked files)."
  git archive --format=zip -o "${OUT}" HEAD
  echo "Created ${OUT} (git archive)."
else
  # Build zip exclude args
  EXCL_ARGS=()
  for e in "${EXCLUDES[@]}"; do
    EXCL_ARGS+=("-x" "${e}")
  done
  echo "No .git found. Using zip to pack current directory excluding common patterns."
  # Ensure zip is installed
  if ! command -v zip >/dev/null 2>&1; then
    echo "Error: zip command not found. Install zip (e.g. apt-get install zip / brew install zip) and retry."
    exit 1
  fi
  # Create zip
  zip -r "${OUT}" . "${EXCL_ARGS[@]}"
  echo "Created ${OUT} (zip)."
fi

echo "Done. File: ${OUT}"