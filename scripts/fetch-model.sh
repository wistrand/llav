#!/usr/bin/env bash
# Download the pinned Qwen3.5-4B Q8_0 GGUF and verify its SHA-256.
#   scripts/fetch-model.sh [DIR]    (default: current directory)
set -euo pipefail

dir="${1:-.}"
revision=4168f45a16a1290d65a4ec0fa312ae917a4c15d6
file=Qwen_Qwen3.5-4B-Q8_0.gguf
sha256=5c74c0ede371924357dff0cb6ba145bd67208b9b2389ded681adfff3f7608db7

mkdir -p "$dir"
cd "$dir"
curl -fL --retry 5 -C - -o "$file" \
  "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/resolve/$revision/$file"
# BSD sha256sum needs "-" to read the check list from stdin; older macOS has only shasum.
if command -v sha256sum >/dev/null; then
  echo "$sha256  $file" | sha256sum -c -
else
  echo "$sha256  $file" | shasum -a 256 -c -
fi
echo "Model ready: $(pwd)/$file"
