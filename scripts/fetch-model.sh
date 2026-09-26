#!/usr/bin/env bash
# Download a pinned GGUF (Q8_0, or Q4_K_M for the 30B) and verify its SHA-256.
#   scripts/fetch-model.sh [DIR] [MODEL]    (default: current directory, qwen3.5-4b)
# MODEL is one of:
#   qwen3.5-4b          Qwen3.5-4B, the default; the validation in agent_docs/research.md used it
#   qwen3.5-2b          Qwen3.5-2B, Apache 2.0; about twice as fast as the default on a long state
#   granite-4.0-h-tiny  IBM Granite 4.0 H Tiny (7B MoE, about 1B active), Apache 2.0
#   granite-4.2-3b      IBM Granite 4.2 3B, Apache 2.0
#   smollm3-3b          Hugging Face SmolLM3-3B, Apache 2.0
#   gemma-4-e4b         Google Gemma 4 E4B (8B total, about 4B active), Apache 2.0; matches the default on the
#                       benchmark sets, the strongest non-Chinese model screened (agent_docs/comparisons.md)
#   muse-glimmer-30b    Meta Muse Glimmer 30B, Apache 2.0, Q4_K_M 16.8 GB; needs a GPU with about 18 GB free
# Only the default is validated; agent_docs/comparisons.md has the screening results for the others.
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
  qwen3.5-2b)
    repo=bartowski/Qwen_Qwen3.5-2B-GGUF
    revision=7d26695454df6de5fbcce2e58681e62dae06ce43
    file=Qwen_Qwen3.5-2B-Q8_0.gguf
    sha256=be647507ce6cde229b838924d47bfff9763171105563f7f908670dae57c4dbe2
    ;;
  granite-4.2-3b)
    repo=ibm-granite/granite-4.2-3b-GGUF
    revision=c40945d71cd90f249a56985e8155551a9188dc30
    file=granite-4.2-3b-Q8_0.gguf
    sha256=fbe986738041418e26de9e123ba740cb654931f85bf572a71bd01f9e6b85e53d
    ;;
  granite-4.0-h-tiny)
    repo=ibm-granite/granite-4.0-h-tiny-GGUF
    revision=08d5a8a9741dd5c1a95d2d39e25253226aa1464e
    file=granite-4.0-h-tiny-Q8_0.gguf
    sha256=6d89e0698c7e88ebe26efcf59aa22786bf36b729705edbd76c6b118c8df9b297
    ;;
  smollm3-3b)
    repo=ggml-org/SmolLM3-3B-GGUF
    revision=4965cb60b150737b68a0408c36aeefb65078f894
    file=SmolLM3-Q8_0.gguf
    sha256=8aa8cc74656137174a1988d993b00828e65a86fd68773412b632a75aa1373248
    ;;
  gemma-4-e4b)
    repo=bartowski/google_gemma-4-E4B-it-GGUF
    revision=029e94146666900b08caf49a3b47b413dfa8ec66
    file=google_gemma-4-E4B-it-Q8_0.gguf
    sha256=6a6eba0d36a051b5d924211a889c1436717006e7c5d413830c47caa1d46cb598
    ;;
  muse-glimmer-30b)
    repo=meta-models/Muse-Glimmer-30B-GGUF
    revision=70bf1b61ac09f91b24d39038091b41c582bc5d7a
    file=Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
    sha256=4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e
    ;;
  *)
    echo "Unknown model '$model'; choose qwen3.5-4b, qwen3.5-2b, granite-4.2-3b, granite-4.0-h-tiny, smollm3-3b," \
      "gemma-4-e4b or muse-glimmer-30b" >&2
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
