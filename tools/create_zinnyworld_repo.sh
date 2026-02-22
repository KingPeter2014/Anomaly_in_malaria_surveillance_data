#!/usr/bin/env bash
set -euo pipefail

# Creates a standalone git repository for ZinnyWorld outside the current repository.
# Usage:
#   bash tools/create_zinnyworld_repo.sh [destination_path]
# Example:
#   bash tools/create_zinnyworld_repo.sh ../zinnyworld

DEST="${1:-../zinnyworld}"
SRC_DIR="zinnyworld"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source folder '$SRC_DIR' not found."
  exit 1
fi

mkdir -p "$DEST"

# Copy planning assets into the new repository directory
cp -R "$SRC_DIR"/* "$DEST"/

pushd "$DEST" >/dev/null
if [[ ! -d .git ]]; then
  git init -b main >/dev/null
fi

git add .
if ! git diff --cached --quiet; then
  git commit -m "Initialize ZinnyWorld planning baseline" >/dev/null
fi

REPO_ABS_PATH="$(pwd)"
popd >/dev/null

echo "Standalone ZinnyWorld repository is ready at: $REPO_ABS_PATH"
echo "Next: cd '$REPO_ABS_PATH' && git remote add origin <your_repo_url> && git push -u origin main"
