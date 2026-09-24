#!/usr/bin/env bash
# Set up and start llav on a rented CUDA box (Vast.ai, RunPod, ...) reached as root over SSH.
#   scripts/remote-gpu.sh HOST SSH_PORT [MODEL] [-- LLAV_ARGS...]
# Copies this checkout, builds llama-server with CUDA (once), fetches MODEL with fetch-model.sh (default
# qwen3.5-4b) and runs llav in the tmux session "llav". Safe to rerun: it re-syncs the code and restarts llav.
# The instance needs a CUDA *devel* image (nvcc) and apt, e.g. vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-2026-09-07
# (Ubuntu 24.04: the native CUDA architecture flag needs CMake 3.24+, newer than 22.04 ships).
# llav listens on the box's localhost:8765, since Vast images proxy Jupyter on 8080; the printed SSH tunnel maps
# it to local port 8080. Passing --port in LLAV_ARGS breaks the startup wait and the tunnel line.
# TS_AUTHKEY=tskey-... joins the box to your tailnet (userspace networking: Vast containers have no /dev/net/tun)
# and serves llav at https://<TS_HOSTNAME>.<tailnet>.ts.net, so no SSH tunnel and no public port. Use an
# ephemeral, pre-approved key; the box keeps it in /root/.llav-ts-authkey to survive a rerun. TS_HOSTNAME
# defaults to llav-gpu. The node stays in the tailnet until you delete it there, even after the box is gone.
# NATIVE=1 also builds native/llav-readout and starts llav with it: one batched pass per request instead of
# one llama-server pass per question, which is worth most on a fast GPU (see native/README.md).
# LLAMA_BIN=FILE.tgz uploads a prebuilt build/bin instead of compiling, when the box has none yet. Save one from a
# finished box with: ssh -p PORT root@HOST 'tar -C /root/llama.cpp/build -czf - bin' > FILE.tgz
# It runs only on the GPU architectures it was built for (native: the build box's), and must unpack at the same path.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,3p' "$0" >&2
  exit 2
fi
host="$1"
port="$2"
shift 2
model=qwen3.5-4b
if [[ $# -gt 0 && "$1" != "--" ]]; then
  model="$1"
  shift
fi
[[ "${1:-}" == "--" ]] && shift
llav_args=""
[[ $# -gt 0 ]] && llav_args="$(printf ' %q' "$@")"

repo="$(cd "$(dirname "$0")/.." && pwd)"
# Keepalives: a dropped session kills the remote build (a rerun resumes it).
ssh_cmd=(ssh -p "$port" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "root@$host")

"${ssh_cmd[@]}" "mkdir -p /root/llav"
rsync -a --delete --exclude .git --exclude .claude --exclude '*.gguf' --exclude models/ --exclude __pycache__/ \
  -e "ssh -p $port" "$repo/" "root@$host:/root/llav/"

if [[ -n "${LLAMA_BIN:-}" ]]; then
  if ! "${ssh_cmd[@]}" "test -x /root/llama.cpp/build/bin/llama-server"; then
    "${ssh_cmd[@]}" "mkdir -p /root/llama.cpp/build && tar -C /root/llama.cpp/build -xzf -" < "$LLAMA_BIN"
  fi
fi

"${ssh_cmd[@]}" "MODEL=$(printf %q "$model") LLAV_ARGS=$(printf %q "$llav_args") NATIVE=$(printf %q "${NATIVE:-}") \
  TS_AUTHKEY=$(printf %q "${TS_AUTHKEY:-}") TS_HOSTNAME=$(printf %q "${TS_HOSTNAME:-llav-gpu}") bash -s" <<'EOF'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

need=()
for tool in git cmake g++ python3 curl tmux; do
  command -v "$tool" >/dev/null || need+=("$tool")
done
if [[ ${#need[@]} -gt 0 ]]; then
  apt-get update -qq
  apt-get install -y -qq git cmake build-essential python3 curl tmux >/dev/null
fi

# Non-interactive SSH shells often lack the CUDA toolkit on PATH.
export PATH="/usr/local/cuda/bin:$PATH"

# Built from upstream rather than taken from an image: Vast's llama-cpp image ships the unsloth fork, whose
# patches the readout has not been validated against.
server=/root/llama.cpp/build/bin/llama-server
# --version fails when an uploaded build lacks its libraries; rebuild then rather than failing at llav startup.
if ! "$server" --version >/dev/null 2>&1; then
  # init + fetch rather than clone: an unusable uploaded build leaves a non-empty /root/llama.cpp that clone refuses.
  if [[ ! -d /root/llama.cpp/.git ]]; then
    git init -q /root/llama.cpp
    git -C /root/llama.cpp fetch -q --depth 1 https://github.com/ggml-org/llama.cpp HEAD
    git -C /root/llama.cpp checkout -q FETCH_HEAD
  fi
  # Compile kernels for this box's GPU only; the all-architectures default takes several times longer.
  cmake -S /root/llama.cpp -B /root/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_CURL=OFF
  # nproc reports the host's cores, not the container's CPU quota; nvcc jobs sized to the host can exhaust RAM.
  jobs="$(nproc)"
  if read -r quota period < /sys/fs/cgroup/cpu.max 2>/dev/null && [[ "$quota" != max ]]; then
    jobs=$(( (quota + period - 1) / period ))
  elif read -r quota < /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null && [[ "$quota" -gt 0 ]]; then
    read -r period < /sys/fs/cgroup/cpu/cpu.cfs_period_us
    jobs=$(( (quota + period - 1) / period ))
  fi
  cmake --build /root/llama.cpp/build -j "$jobs" --target llama-server
fi

if [[ -n "$NATIVE" ]]; then
  # Built here against this llama.cpp: the helper uses its C API, so both must come from one version.
  g++ -O2 -std=c++17 -pthread -o /root/llav-readout /root/llav/native/llav-readout.cpp \
    -I/root/llama.cpp/include -I/root/llama.cpp/ggml/include \
    -L/root/llama.cpp/build/bin -lllama -Wl,-rpath,/root/llama.cpp/build/bin
  LLAV_ARGS="$LLAV_ARGS --native-readout /root/llav-readout"
fi

# Tailscale, when asked for: the key is stored so a rerun does not need it again.
if [[ -n "$TS_AUTHKEY" ]]; then  # not a && chain: under set -e an empty key would end the run here
  printf %s "$TS_AUTHKEY" > /root/.llav-ts-authkey
  chmod 600 /root/.llav-ts-authkey
fi
if [[ -s /root/.llav-ts-authkey ]]; then
  command -v tailscaled >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
  # Userspace networking: these containers have no TUN device. tailscale serve then proxies to llav.
  pgrep -x tailscaled >/dev/null || (tailscaled --tun=userspace-networking \
    --socks5-server=localhost:1055 --state=/var/lib/tailscale/tailscaled.state \
    > /root/tailscaled.log 2>&1 &)
  for _ in $(seq 30); do tailscale status >/dev/null 2>&1 && break; sleep 1; done
  # A bad or expired key must not stop llav from starting; the tunnel still works.
  tailscale up --authkey "$(cat /root/.llav-ts-authkey)" --hostname "$TS_HOSTNAME" --ssh >/dev/null \
    || echo "tailscale up failed; continuing without it (see /root/tailscaled.log)" >&2
fi

gguf="$(/root/llav/scripts/fetch-model.sh /root/models "$MODEL" | sed -n 's/^Model ready: //p')"

tmux kill-session -t llav 2>/dev/null || true
tmux new-session -d -s llav -c /root/llav \
  "PYTHONPATH=src python3 -m llav --gguf $gguf --llama-server $server --web-ui --port 8765$LLAV_ARGS 2>&1 | tee /root/llav.log"

for _ in $(seq 120); do
  curl -sf -o /dev/null http://127.0.0.1:8765/ && break
  sleep 1
done
if [[ -s /root/.llav-ts-authkey ]]; then
  # --bg survives this shell; https needs MagicDNS and HTTPS certificates enabled in the tailnet.
  tailscale serve --bg 8765 >/dev/null 2>&1 || echo "tailscale serve failed; see tailscale serve status" >&2
  echo "tailnet URL: $(tailscale status --json | python3 -c 'import json,sys; print("https://" + json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || echo unknown)"
fi
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
tail -n 5 /root/llav.log
EOF

cat <<MSG

llav is running in tmux session "llav" on $host (log: /root/llav.log).
Tunnel:  ssh -N -o ServerAliveInterval=15 -o ServerAliveCountMax=6 -p $port -L 8080:localhost:8765 root@$host
         then open http://localhost:8080 (the box drops idle sessions after ~30 s without the keepalives;
         with TS_AUTHKEY the tailnet URL above needs no tunnel at all)
Attach:  ssh -t -p $port root@$host tmux attach -t llav
Destroy the instance when done; a stopped instance still bills for storage.
MSG
