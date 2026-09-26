// Batched option-logit readout against libllama, for llav's optional fast path.
//
//   llav-readout --model FILE.gguf [--ctx 8192] [--seq 32] [--ngl 999] [--threads N]
//
// Speaks a binary protocol on stdin/stdout so llav needs no parser on either side. One request holds a
// state prefix and every question's suffix; the prefix is decoded once, copied to one sequence per
// question with llama_memory_seq_cp, and all suffixes are decoded together, in as few llama_decode calls as
// the context's batch size allows. That is the point of this program: llama-server evaluates each question in
// its own pass, which costs a fixed per-pass overhead that dominates on a fast GPU.
//
// Request : "LLVR" n_prefix n_labels n_suffix, prefix tokens, label tokens, then per suffix: n, tokens.
// Response: "LLVA" status evaluated, then per suffix n_labels raw logits followed by the log of the
//           vocabulary normalizer (logsumexp over every logit). llav softmaxes the label logits over the
//           declared options, which cancels the normalizer; it needs the normalizer only for the candidate
//           mass, the share of the whole vocabulary the options received.
// All integers are little-endian int32, floats little-endian binary32. status 0 is success.
//
// The prefix of the previous request stays resident, so repeat requests skip its decode entirely.
// The ready line names the protocol version; llav refuses a helper that does not match.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <unistd.h>
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

// Answers go to a private copy of the original stdout (see main), never to stdout itself.
FILE * answers = nullptr;

void write_response(int32_t status, int32_t evaluated, const std::vector<float> & values) {
    fwrite("LLVA", 1, 4, answers);
    fwrite(&status, sizeof(status), 1, answers);
    fwrite(&evaluated, sizeof(evaluated), 1, answers);
    if (!values.empty()) {
        fwrite(values.data(), sizeof(float), values.size(), answers);
    }
    fflush(answers);
}

// exp(x) for x <= 0, within 5e-6 relative error: 2^t split into an integer power, set in the exponent bits,
// and a polynomial for the fraction. Unlike std::exp it has no library call, so the loop below vectorizes;
// the normalizer only feeds the candidate mass, which is reported to four decimals.
inline float exp_nonpositive(float x) {
    x = x < -87.0f ? -87.0f : x;
    const float t = x * 1.4426950409f;
    const int32_t whole = static_cast<int32_t>(t);  // toward zero, so the fraction is in (-1, 0]
    const float f = t - static_cast<float>(whole);
    const float p = 1.0f + f * (0.6931471806f + f * (0.2402265070f + f * (0.0555041087f + f * (0.0096181291f
                    + f * (0.0013333558f + f * (0.0001540353f + f * 0.0000152527f))))));
    const int32_t bits = (whole + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    return p * scale;
}

// Largest logit and sum of exp(logit - largest) over [begin, end).
void partial_normalizer(const float * logits, int32_t begin, int32_t end, float & top, float & sum) {
    top = -INFINITY;
    for (int32_t i = begin; i < end; ++i) {
        top = logits[i] > top ? logits[i] : top;
    }
    float lanes[16] = {};
    int32_t i = begin;
    for (; i + 16 <= end; i += 16) {
        for (int32_t j = 0; j < 16; ++j) {
            lanes[j] += exp_nonpositive(logits[i + j] - top);
        }
    }
    double total = 0.0;
    for (float lane : lanes) {
        total += lane;
    }
    for (; i < end; ++i) {
        total += exp_nonpositive(logits[i] - top);
    }
    sum = static_cast<float>(total);
}

// log of sum(exp(logit)) over the vocabulary, per row. The pass reads every logit (248k for Qwen3.5), about
// 1 ms per row on one core, which would add a large share to a batched request on a fast GPU; the vocabulary
// is split across `threads`, idle once the decode has finished.
std::vector<float> log_normalizers(const std::vector<const float *> & rows, int32_t n_vocab, int32_t threads) {
    const int32_t chunks = std::max(1, std::min(threads, n_vocab));
    std::vector<float> tops(rows.size() * chunks);
    std::vector<float> sums(rows.size() * chunks);
    auto work = [&](int32_t chunk) {
        const int32_t begin = static_cast<int32_t>(static_cast<int64_t>(n_vocab) * chunk / chunks);
        const int32_t end = static_cast<int32_t>(static_cast<int64_t>(n_vocab) * (chunk + 1) / chunks);
        for (size_t row = 0; row < rows.size(); ++row) {
            partial_normalizer(rows[row], begin, end, tops[row * chunks + chunk], sums[row * chunks + chunk]);
        }
    };
    std::vector<std::thread> pool;
    for (int32_t chunk = 1; chunk < chunks; ++chunk) {
        pool.emplace_back(work, chunk);
    }
    work(0);
    for (std::thread & thread : pool) {
        thread.join();
    }
    std::vector<float> out;
    for (size_t row = 0; row < rows.size(); ++row) {
        const float * top = &tops[row * chunks];
        const float largest = *std::max_element(top, top + chunks);
        double total = 0.0;
        for (int32_t chunk = 0; chunk < chunks; ++chunk) {
            total += sums[row * chunks + chunk] * std::exp(static_cast<double>(top[chunk] - largest));
        }
        out.push_back(largest + static_cast<float>(std::log(total)));
    }
    return out;
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

    // Anything llama.cpp, ggml or a GPU driver prints to stdout would land in the middle of a binary answer
    // and desynchronize llav. Keep the real stdout for answers only and point stdout at stderr.
    const int answer_fd = dup(STDOUT_FILENO);
    if (answer_fd < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0 || (answers = fdopen(answer_fd, "wb")) == nullptr) {
        fprintf(stderr, "llav-readout: cannot set up the answer channel\n");
        return 1;
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
    // No swa_full, unlike llama-server's --swa-full: the helper only extends a resident prefix forward, which
    // needs just the last window, and a full cache over (seq + 1) * ctx positions cost 2.6 GB more on Gemma 3
    // 1B for identical answers (agent_docs/comparisons.md).
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "llav-readout: cannot create a context\n");
        llama_model_free(model);
        return 1;
    }
    llama_memory_t memory = llama_get_memory(ctx);
    const int32_t n_batch = static_cast<int32_t>(llama_n_batch(ctx));
    llama_batch batch = llama_batch_init(n_batch, 0, 1);  // reused for every decode
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // Tell llav the helper is up before it sends anything; it waits for this line.
    fprintf(stderr, "llav-readout ready: protocol 2, vocab %d, %d sequences of %d tokens\n", n_vocab, args.seq,
            args.ctx);
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
            // In chunks of at most n_batch tokens: llama_decode aborts the process on a larger batch rather
            // than returning an error.
            int32_t status = 0;
            for (int32_t start = 0; start < n_prefix && status == 0; start += n_batch) {
                const int32_t count = std::min(n_batch, n_prefix - start);
                batch.n_tokens = count;
                for (int32_t i = 0; i < count; ++i) {
                    batch.token[i] = prefix[start + i];
                    batch.pos[i] = start + i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = 0;
                }
                status = llama_decode(ctx, batch);
            }
            if (status != 0) {
                fprintf(stderr, "llav-readout: prefix decode failed (%d)\n", status);
                write_response(2, 0, {});
                continue;
            }
            resident = prefix;
            evaluated += n_prefix;
        }

        // One sequence per question, each a copy of the prefix, then the suffixes packed into as few decodes
        // as n_batch allows. A suffix may span decodes; its logits come from the decode holding its last
        // token and are copied out before the next decode reuses the buffer.
        for (int32_t i = 0; i < n_suffix; ++i) {
            llama_memory_seq_rm(memory, i + 1, -1, -1);
            llama_memory_seq_cp(memory, 0, i + 1, -1, -1);
        }
        std::vector<float> values(static_cast<size_t>(n_suffix) * (n_labels + 1));
        int32_t status = 0;
        int32_t question = 0;
        size_t offset = 0;  // next token of suffixes[question]
        while (question < n_suffix && status == 0) {
            batch.n_tokens = 0;
            std::vector<std::pair<int32_t, int32_t>> outputs;  // (question, index in this batch)
            while (question < n_suffix && batch.n_tokens < n_batch) {
                const std::vector<llama_token> & suffix = suffixes[question];
                const int32_t at = batch.n_tokens++;
                batch.token[at] = suffix[offset];
                batch.pos[at] = n_prefix + static_cast<int32_t>(offset);
                batch.n_seq_id[at] = 1;
                batch.seq_id[at][0] = question + 1;
                batch.logits[at] = offset + 1 == suffix.size();
                if (batch.logits[at]) {
                    outputs.emplace_back(question, at);
                    ++question;
                    offset = 0;
                } else {
                    ++offset;
                }
            }
            status = llama_decode(ctx, batch);
            if (status != 0) {
                break;
            }
            std::vector<const float *> rows;
            for (const auto & [index, at] : outputs) {
                rows.push_back(llama_get_logits_ith(ctx, at));
            }
            if (std::find(rows.begin(), rows.end(), nullptr) != rows.end()) {
                status = -1;
                break;
            }
            const std::vector<float> normalizers = log_normalizers(rows, n_vocab, args.threads);
            for (size_t k = 0; k < outputs.size(); ++k) {
                float * out = &values[static_cast<size_t>(outputs[k].first) * (n_labels + 1)];
                for (int32_t j = 0; j < n_labels; ++j) {  // raw: llav's softmax over the options cancels any constant
                    const llama_token label = labels[j];
                    out[j] = label >= 0 && label < n_vocab ? rows[k][label] : -1e30f;
                }
                out[n_labels] = normalizers[k];
            }
        }
        if (status != 0) {
            fprintf(stderr, "llav-readout: suffix decode failed (%d)\n", status);
            write_response(status == -1 ? 4 : 3, 0, {});
            continue;
        }
        evaluated += n_suffix_tokens;
        write_response(0, evaluated, values);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
