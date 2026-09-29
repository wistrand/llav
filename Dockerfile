# llav in a container, laid out for a Hugging Face Docker Space (port 7860, non-root user 1000) but usable
# anywhere Docker runs. Nothing is compiled: llama-server comes from llama.cpp's own prebuilt CUDA image.
#
#   docker build -t llav .                                  # default model, qwen3.5-4b
#   docker build -t llav --build-arg MODEL=qwen3.5-2b .     # any name scripts/fetch-model.sh accepts
#   docker run --gpus all -p 7860:7860 -e LLAV_API_KEY=secret llav
#   docker run -p 7860:7860 llav                            # no GPU: llama.cpp finds no CUDA device and runs on
#                                                           # the CPU; slow but correct
#
# On a Space, set LLAV_API_KEY as a secret or the server is open to everyone. Extra llav flags go in LLAV_ARGS,
# for example LLAV_ARGS="--slots 2 --ctx 4096".
#
# The default image tag tracks llama.cpp's newest build; pin one with --build-arg LLAMA_IMAGE=... once you have
# a build that works. For a CPU-only image use ghcr.io/ggml-org/llama.cpp:server, or :server-vulkan for
# non-NVIDIA GPUs; the rest of this file is unchanged.

# CUDA 12.8 on Ubuntu 24.04 with kernels for every supported GPU and a CPU backend per x86 level.
ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda

# ---------------------------------------------------------------------------------------------------------------
FROM ubuntu:24.04 AS model

ARG MODEL=qwen3.5-4b
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY scripts/fetch-model.sh /fetch-model.sh
# Pinned revision and SHA-256 per model live in that script; a checksum mismatch fails the build.
RUN /fetch-model.sh /model "${MODEL}"

# ---------------------------------------------------------------------------------------------------------------
FROM ${LLAMA_IMAGE}

# The image has llama-server and its backend libraries in /app, plus libgomp and curl. It lacks Python.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 images ship an "ubuntu" user that already holds uid 1000, which Spaces run the container as.
RUN userdel -r ubuntu 2>/dev/null; useradd -m -u 1000 llav
USER llav
WORKDIR /home/llav

COPY --from=model --chown=llav /model /model
COPY --chown=llav src /home/llav/src

# The image's libraries sit beside the binaries in /app and its entrypoint runs from there; llav runs from
# its home directory, so the loader needs the path spelled out.
ENV PATH=/app:$PATH \
    LD_LIBRARY_PATH=/app \
    PYTHONPATH=/home/llav/src \
    PYTHONUNBUFFERED=1 \
    LLAV_ARGS="--web-ui"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/health || exit 1

# The base image's entrypoint is llama-server itself; llav starts llama-server, so run llav instead. The glob
# matches the single file fetch-model.sh left in /model; the file name becomes the model id (llav-<gguf name>).
ENTRYPOINT []
# llama-server logs to a file under llav's scratch directory; tail it into the container log so a start that
# never reaches "serving" can be diagnosed without a shell in the container.
CMD ["sh", "-c", "llama-server --version; (sleep 10; exec tail -n +1 -F /tmp/llav-*/llama-server.log) & exec python3 -m llav --gguf /model/*.gguf --host 0.0.0.0 --port 7860 $LLAV_ARGS"]
