#!/bin/bash
# Runs on the rented box (paths are the box's: /root/llav, /root/models). Screen popular GGUFs under llav: download, serve without the helper on 8766, run benchmark.py accuracy and
# timing, keep the logs, stop, delete the file. The main llav on 8765 is stopped first (GPU memory) and
# restarted at the end. Rerun to resume: finished models are skipped.
set -u
cd /root/llav
OUT=/root/screen; mkdir -p $OUT
MODELS=(
  "llama-3.2-3b|https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf"
  "llama-3.1-8b|https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
  "mistral-7b-v0.3|https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q8_0.gguf"
  "gemma-4-e4b|https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF/resolve/main/google_gemma-4-E4B-it-Q8_0.gguf"
  "gemma-4-12b|https://huggingface.co/bartowski/gemma-4-12B-it-GGUF/resolve/main/gemma-4-12B-it-Q8_0.gguf"
  "gpt-oss-20b|https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf"
)
tmux kill-session -t llav 2>/dev/null
for p in $(pgrep -f "[p]ython3 -m llav --gguf"); do kill -TERM $p; done; sleep 8
for entry in "${MODELS[@]}"; do
  name=${entry%%|*}; url=${entry##*|}; file=/root/models/screen-$name.gguf
  [ -f $OUT/$name.done ] && continue
  echo "== $name start $(date +%T)"
  [ -f $file ] || curl -sL -o $file "$url" || { echo "== $name download failed"; continue; }
  PYTHONPATH=src nohup python3 -m llav --gguf $file --llama-server /root/llama.cpp/build/bin/llama-server \
    --port 8766 --llama-port 8090 > $OUT/$name-server.log 2>&1 < /dev/null &
  spid=$!
  for i in $(seq 240); do curl -sf http://127.0.0.1:8766/health >/dev/null && break; kill -0 $spid 2>/dev/null || break; sleep 2; done
  if curl -sf http://127.0.0.1:8766/health >/dev/null; then
    curl -s http://127.0.0.1:8766/v1/models > $OUT/$name-models.json
    python3 scripts/benchmark.py accuracy http://127.0.0.1:8766 > $OUT/$name-accuracy.txt 2>&1
    python3 scripts/benchmark.py timing http://127.0.0.1:8766 > $OUT/$name-timing.txt 2>&1
    python3 scripts/perturb.py run http://127.0.0.1:8766 /root/perturb-data/banking77.jsonl /root/perturb-data/ag_news.jsonl \
      --limit 150 --repeats 1 --rotations 0 --perms 6 --out $OUT/$name-perturb.jsonl > $OUT/$name-perturb.log 2>&1
    python3 scripts/perturb.py analyze $OUT/$name-perturb.jsonl > $OUT/$name-perturb-analysis.txt 2>&1
    kill -TERM $spid; wait $spid 2>/dev/null
  else
    echo "== $name: llav did not start (startup probe or load failure)"; tail -5 $OUT/$name-server.log
  fi
  rm -f $file
  touch $OUT/$name.done
  echo "== $name done $(date +%T)"
done
tmux new-session -d -s llav -c /root/llav "PYTHONPATH=src python3 -m llav --gguf /root/models/Qwen_Qwen3.5-4B-Q8_0.gguf --llama-server /root/llama.cpp/build/bin/llama-server --web-ui --port 8765 --native-readout /root/llav-readout 2>&1 | tee /root/llav.log"
touch $OUT/screen.done
echo "== all done $(date +%T)"
