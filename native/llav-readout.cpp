// Batched option-logit readout against libllama, for llav's optional fast path.
//
//   llav-readout --model FILE.gguf [--ctx 8192] [--seq 32] [--ngl 999] [--threads N]
//
// Speaks a binary protocol on stdin/stdout so llav needs no parser on either side. One request holds a
// state prefix and every question's suffix; the prefix is decoded once, copied to one sequence per
// question with llama_memory_seq_cp, and all suffixes are decoded in a single llama_decode. That is the
// point of this program: llama-server evaluates each question in its own pass, which costs a fixed
// per-pass overhead that dominates on a fast GPU.
//
// Request : "LLVR" n_prefix n_labels n_suffix, prefix tokens, label tokens, then per suffix: n, tokens.
// Response: "LLVA" status evaluated, then n_suffix * n_labels float log-probabilities (natural log,
//           normalized over the whole vocabulary, matching llama-server's n_probs values).
// All integers are little-endian int32, floats little-endian binary32. status 0 is success.
//
// The prefix of the previous request stays resident, so repeat requests skip its decode entirely.

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool read_exact(void * out, size_t bytes) {
    return bytes == 0 || fread(out, 1, bytes, stdin) == bytes;
}

bool read_i32(int32_t & value) {
    return read_exact(&value, sizeof(value));
}

bool read_tokens(std::vector<llama_token> & out, int32_t count) {
    out.resize(count);
    return read_exact(out.data(), sizeof(llama_token) * static_cast<size_t>(count));
}

void write_response(int32_t status, int32_t evaluated, const std::vector<float> & values) {
    fwrite("LLVA", 1, 4, stdout);
    fwrite(&status, sizeof(status), 1, stdout);
    fwrite(&evaluated, sizeof(evaluated), 1, stdout);
    if (!values.empty()) {
        fwrite(values.data(), sizeof(float), values.size(), stdout);
    }
    fflush(stdout);
}

// Natural-log softmax of the label tokens over the whole vocabulary, as llama-server reports it.
void label_logprobs(const float * logits, int32_t n_vocab, const std::vector<llama_token> & labels,
                    std::vector<float> & out) {
    float top = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        top = logits[i] > top ? logits[i] : top;
    }
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        sum += std::exp(static_cast<double>(logits[i] - top));
    }
    const float norm = top + static_cast<float>(std::log(sum));
    for (llama_token label : labels) {
        out.push_back(label >= 0 && label < n_vocab ? logits[label] - norm : -1e30f);
    }
}

struct Args {
    std::string model;
    int32_t ctx = 8192;
    int32_t seq = 32;
    int32_t ngl = 999;
    int32_t threads = 4;
};

bool parse(int argc, char ** argv, Args & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        const bool has_value = i + 1 < argc;
        if (flag == "--model" && has_value) {
            args.model = argv[++i];
        } else if (flag == "--ctx" && has_value) {
            args.ctx = std::stoi(argv[++i]);
        } else if (flag == "--seq" && has_value) {
            args.seq = std::stoi(argv[++i]);
        } else if (flag == "--ngl" && has_value) {
            args.ngl = std::stoi(argv[++i]);
        } else if (flag == "--threads" && has_value) {
            args.threads = std::stoi(argv[++i]);
        } else {
            fprintf(stderr, "llav-readout: unknown argument %s\n", flag.c_str());
            return false;
        }
    }
    if (args.model.empty()) {
        fprintf(stderr, "llav-readout: --model is required\n");
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse(argc, argv, args)) {
        return 2;
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = args.ngl;
    llama_model * model = llama_model_load_from_file(args.model.c_str(), model_params);
    if (model == nullptr) {
        fprintf(stderr, "llav-readout: cannot load %s\n", args.model.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(args.ctx) * (static_cast<uint32_t>(args.seq) + 1);
    ctx_params.n_batch = 2048;
    ctx_params.n_ubatch = 512;
    ctx_params.n_seq_max = static_cast<uint32_t>(args.seq) + 1;  // sequence 0 holds the prefix
    ctx_params.n_threads = args.threads;
    ctx_params.n_threads_batch = args.threads;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "llav-readout: cannot create a context\n");
        llama_model_free(model);
        return 1;
    }
    llama_memory_t memory = llama_get_memory(ctx);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // Tell llav the helper is up before it sends anything; it waits for this line.
    fprintf(stderr, "llav-readout ready: vocab %d, %d sequences of %d tokens\n", n_vocab, args.seq, args.ctx);
    fflush(stderr);

    std::vector<llama_token> resident;  // the prefix currently in sequence 0
    std::vector<llama_token> prefix;
    std::vector<llama_token> labels;
    std::vector<std::vector<llama_token>> suffixes;

    char magic[4];
    while (read_exact(magic, sizeof(magic))) {
        if (memcmp(magic, "LLVR", 4) != 0) {
            fprintf(stderr, "llav-readout: bad request magic\n");
            return 1;
        }
        int32_t n_prefix = 0;
        int32_t n_labels = 0;
        int32_t n_suffix = 0;
        if (!read_i32(n_prefix) || !read_i32(n_labels) || !read_i32(n_suffix)) {
            return 1;
        }
        if (n_prefix < 0 || n_labels <= 0 || n_suffix <= 0 || n_suffix > args.seq) {
            write_response(1, 0, {});
            continue;
        }
        if (!read_tokens(prefix, n_prefix) || !read_tokens(labels, n_labels)) {
            return 1;
        }
        suffixes.resize(n_suffix);
        int32_t n_suffix_tokens = 0;
        for (int32_t i = 0; i < n_suffix; ++i) {
            int32_t count = 0;
            if (!read_i32(count) || count <= 0 || !read_tokens(suffixes[i], count)) {
                return 1;
            }
            n_suffix_tokens += count;
        }

        int32_t evaluated = 0;
        if (prefix != resident) {
            llama_memory_clear(memory, true);
            resident.clear();
            llama_batch batch = llama_batch_init(n_prefix, 0, 1);
            batch.n_tokens = n_prefix;
            for (int32_t i = 0; i < n_prefix; ++i) {
                batch.token[i] = prefix[i];
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 0;
            }
            const int32_t status = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (status != 0) {
                fprintf(stderr, "llav-readout: prefix decode failed (%d)\n", status);
                write_response(2, 0, {});
                continue;
            }
            resident = prefix;
            evaluated += n_prefix;
        }

        // One sequence per question, each a copy of the prefix, then every suffix in one decode.
        llama_batch batch = llama_batch_init(n_suffix_tokens, 0, 1);
        batch.n_tokens = 0;
        std::vector<int32_t> output_index(n_suffix, -1);
        for (int32_t i = 0; i < n_suffix; ++i) {
            const llama_seq_id seq = i + 1;
            llama_memory_seq_rm(memory, seq, -1, -1);
            llama_memory_seq_cp(memory, 0, seq, -1, -1);
            for (size_t j = 0; j < suffixes[i].size(); ++j) {
                const int32_t at = batch.n_tokens++;
                batch.token[at] = suffixes[i][j];
                batch.pos[at] = n_prefix + static_cast<int32_t>(j);
                batch.n_seq_id[at] = 1;
                batch.seq_id[at][0] = seq;
                batch.logits[at] = j + 1 == suffixes[i].size();
                if (batch.logits[at]) {
                    output_index[i] = at;
                }
            }
        }
        evaluated += n_suffix_tokens;
        const int32_t status = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (status != 0) {
            fprintf(stderr, "llav-readout: suffix decode failed (%d)\n", status);
            write_response(3, 0, {});
            continue;
        }

        std::vector<float> values;
        values.reserve(static_cast<size_t>(n_suffix) * n_labels);
        for (int32_t i = 0; i < n_suffix; ++i) {
            const float * logits = llama_get_logits_ith(ctx, output_index[i]);
            if (logits == nullptr) {
                write_response(4, 0, {});
                values.clear();
                break;
            }
            label_logprobs(logits, n_vocab, labels, values);
        }
        if (!values.empty()) {
            write_response(0, evaluated, values);
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
