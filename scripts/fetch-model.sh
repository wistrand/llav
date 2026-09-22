#!/usr/bin/env bash
# Download a pinned Q8_0 GGUF and verify its SHA-256.
#   scripts/fetch-model.sh [DIR] [MODEL]    (default: current directory, qwen3.5-4b)
# MODEL is one of:
#   qwen3.5-4b          Qwen3.5-4B, the default; the validation in agent_docs/research.md used it
#   granite-4.0-micro   IBM Granite 4.0 Micro (3B), Apache 2.0
#   smollm3-3b          Hugging Face SmolLM3-3B, Apache 2.0
set -euo pipefail

dir="${1:-.}"
model="${2:-qwen3.5-4b}"

case "$model" in
  qwen3.5-4b)
    repo=bartowski/Qwen_Qwen3.5-4B-GGUF
    revision=4168f45a16a1290d65a4ec0fa312ae917a4c15d6
    file=Qwen_Qwen3.5-4B-Q8_0.gguf
    sha256=5c74c0ede371924357dff0cb6ba145bd67208b9b2389ded681adfff3f7608db7
    ;;
  granite-4.0-micro)
    repo=ibm-granite/granite-4.0-micro-GGUF
    revision=ec48475f0c811d812fbfb61975717a9c36eeb652
    file=granite-4.0-micro-Q8_0.gguf
    sha256=a023da9d89a7f3c5369ac1acfe8fa57277eaec591e1b23dde0cd8b01e0bd3fd6
    ;;
  smollm3-3b)
    repo=ggml-org/SmolLM3-3B-GGUF
    revision=4965cb60b150737b68a0408c36aeefb65078f894
    file=SmolLM3-Q8_0.gguf
    sha256=8aa8cc74656137174a1988d993b00828e65a86fd68773412b632a75aa1373248
    ;;
  *)
    echo "Unknown model '$model'; choose qwen3.5-4b, granite-4.0-micro or smollm3-3b" >&2
    exit 2
    ;;
esac

mkdir -p "$dir"
cd "$dir"
curl -fL --retry 5 -C - -o "$file" "https://huggingface.co/$repo/resolve/$revision/$file"
# BSD sha256sum needs "-" to read the check list from stdin; older macOS has only shasum.
if command -v sha256sum >/dev/null; then
  echo "$sha256  $file" | sha256sum -c -
else
  echo "$sha256  $file" | shasum -a 256 -c -
fi
echo "Model ready: $(pwd)/$file"
