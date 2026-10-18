#include "llama-moe-profile.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {

// gate, up, down or gate_up, down
static bool llama_moe_profile_has_experts(const llama_layer & layer) {
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            return true;
        }
    }
    return false;
}

}

struct llama_moe_profile::impl {
    int32_t n_layer;
    int32_t n_expert;
    int32_t n_expert_used;
    int64_t max_tokens;

    std::string output_path;

    std::vector<uint64_t> count; // [n_layer*n_expert] tokens that selected the expert
    std::vector<double>   wsum;  // [n_layer*n_expert] sum of the expert weights over those tokens

    // host storage of the ids and the weights, one tensor per layer
    ggml_backend_buffer_ptr buf;

    mutable ggml_context_ptr ctx; // the views are created lazily during the graph build

    std::vector<ggml_tensor *> ids;     // [n_layer] I32 [n_expert_used, max_tokens], nullptr if the layer has no experts
    std::vector<ggml_tensor *> weights; // [n_layer] F32 [n_expert_used, max_tokens]

    // views of the base tensors for the n_tokens of each graph, kept so graph reuse finds the same tensors
    mutable std::map<std::pair<int32_t, int64_t>, ggml_tensor *> view_ids;
    mutable std::map<std::pair<int32_t, int64_t>, ggml_tensor *> view_weights;

    // the views are created lazily in a fixed-size context, so the number of distinct batch sizes is capped
    static constexpr int max_views = 1024;
    mutable int n_views = 0;

    ggml_tensor * view_of(std::map<std::pair<int32_t, int64_t>, ggml_tensor *> & views, ggml_tensor * base, int32_t il, int64_t n_tokens, const char * name) const {
        if (base == nullptr || n_tokens <= 0 || n_tokens > max_tokens) {
            return nullptr;
        }
        const auto key = std::make_pair(il, n_tokens);
        const auto it = views.find(key);
        if (it != views.end()) {
            return it->second;
        }
        if (n_views >= max_views) {
            LLAMA_LOG_WARN("%s: too many distinct batch sizes, skipping profiling for layer %d, %d tokens\n", __func__, il, (int) n_tokens);
            return nullptr;
        }
        ggml_tensor * res = ggml_view_2d(ctx.get(), base, n_expert_used, n_tokens, base->nb[1], 0);
        ggml_format_name(res, "moe_profile.%s-%d", name, il);
        ggml_backend_view_init(res);
        views[key] = res;
        ++n_views;
        return res;
    }

    impl(const llama_model & model, int64_t max_tokens, std::string output_path) :
            n_layer((int32_t) model.layers.size()),
              n_expert((int32_t) model.hparams.n_expert),
              n_expert_used((int32_t) model.hparams.n_expert_used_max()),
              max_tokens(max_tokens),
              output_path(std::move(output_path)) {
        if (n_expert == 0 || n_expert_used == 0 || max_tokens <= 0) {
            throw std::runtime_error("MoE profile requires a MoE model");
        }

        count.assign((size_t) n_layer*n_expert, 0);
        wsum.assign((size_t) n_layer*n_expert, 0.0);

        int n_tensors = 0;
        for (int32_t il = 0; il < n_layer; ++il) {
            if (llama_moe_profile_has_experts(model.layers[il])) {
                n_tensors += 2;
            }
        }
        if (n_tensors == 0) {
            LLAMA_LOG_WARN("%s: no MoE layers to profile\n", __func__);
            return;
        }

        ggml_init_params params = {
            /*.mem_size   =*/ (size_t) (n_tensors + 2*max_views)*ggml_tensor_overhead() + 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx.reset(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to create the MoE profile context");
        }

        ids.assign(n_layer, nullptr);
        weights.assign(n_layer, nullptr);
        for (int32_t il = 0; il < n_layer; ++il) {
            if (!llama_moe_profile_has_experts(model.layers[il])) {
                continue;
            }
            ids[il]     = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, max_tokens);
            weights[il] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_expert_used, max_tokens);
            ggml_format_name(ids[il],     "moe_profile.ids-%d",     il);
            ggml_format_name(weights[il], "moe_profile.weights-%d", il);
        }

        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        if (!buf) {
            throw std::runtime_error("failed to allocate the MoE profile buffers");
        }
        ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        size_t bytes = 0;
        for (int32_t il = 0; il < n_layer; ++il) {
            if (ids[il] != nullptr) {
                bytes += ggml_nbytes(ids[il]) + ggml_nbytes(weights[il]);
            }
        }
        LLAMA_LOG_INFO("%s: MoE profile = %.2f MiB for %d layers, %d experts, %d used, up to %" PRId64 " tokens\n",
            __func__, bytes/1024.0/1024.0, n_layer, n_expert, n_expert_used, max_tokens);
    }

    ~impl() {
        write();
    }

    ggml_tensor * get_ids(int32_t il, int64_t n_tokens) const {
        return view_of(view_ids, il >= 0 && il < n_layer ? ids[il] : nullptr, il, n_tokens, "ids");
    }

    ggml_tensor * get_weights(int32_t il, int64_t n_tokens) const {
        return view_of(view_weights, il >= 0 && il < n_layer ? weights[il] : nullptr, il, n_tokens, "weights");
    }

    void accumulate(int32_t il, int64_t n_tokens) {
        if (il < 0 || il >= n_layer || ids[il] == nullptr) {
            return;
        }
        const auto it_ids = view_ids.find(std::make_pair(il, n_tokens));
        const auto it_w   = view_weights.find(std::make_pair(il, n_tokens));
        if (it_ids == view_ids.end() || it_w == view_weights.end()) {
            return;
        }

        const int32_t * ids_data = (const int32_t *) it_ids->second->data;
        const float *   w_data   = (const float *)   it_w->second->data;

        const size_t n = (size_t) n_expert_used*n_tokens;
        for (size_t i = 0; i < n; ++i) {
            GGML_ASSERT(ids_data[i] >= 0 && ids_data[i] < n_expert);
            const size_t k = (size_t) il*n_expert + ids_data[i];
            count[k] += 1;
            wsum[k]  += w_data[i];
        }
    }

    void write() {
        if (output_path.empty() || ids.empty()) {
            return;
        }

        size_t n_selected = 0;
        for (uint64_t c : count) {
            if (c > 0) {
                n_selected++;
            }
        }

        const std::string tmp = output_path + ".tmp";
        FILE * f = std::fopen(tmp.c_str(), "w");
        if (f == nullptr) {
            LLAMA_LOG_WARN("%s: failed to open %s\n", __func__, tmp.c_str());
            return;
        }

        fprintf(f, "layer,expert,count,weight_sum\n");
        for (int32_t il = 0; il < n_layer; ++il) {
            for (int32_t e = 0; e < n_expert; ++e) {
                const size_t k = (size_t) il*n_expert + e;
                if (count[k] == 0) {
                    continue;
                }
                // fixed-point to keep the output independent of the locale's decimal separator
                const long long w = std::llround(wsum[k] * 1000000.0);
                fprintf(f, "%d,%d,%" PRIu64 ",%s%lld.%06lld\n", il, e, count[k],
                        w < 0 ? "-" : "", std::llabs(w)/1000000, std::llabs(w)%1000000);
            }
        }

        if (std::fclose(f) != 0) {
            LLAMA_LOG_WARN("%s: failed to close %s\n", __func__, tmp.c_str());
            return;
        }
        if (std::rename(tmp.c_str(), output_path.c_str()) != 0) {
            LLAMA_LOG_WARN("%s: failed to rename %s to %s\n", __func__, tmp.c_str(), output_path.c_str());
            return;
        }

        LLAMA_LOG_INFO("%s: wrote %s: %zu (layer,expert) pairs seen, %d tokens profiled\n",
            __func__, output_path.c_str(), n_selected, (int) total_tokens());
    }

    uint64_t total_tokens() const {
        uint64_t sum = 0;
        for (uint64_t c : count) {
            sum += c;
        }
        return sum / n_expert_used;
    }
};

llama_moe_profile::llama_moe_profile(const llama_model & model, int64_t max_tokens, std::string output_path) :
    pimpl(new impl(model, max_tokens, std::move(output_path))) {
}

llama_moe_profile::~llama_moe_profile() = default;

ggml_tensor * llama_moe_profile::get_ids(int32_t il, int64_t n_tokens) const {
    return pimpl->get_ids(il, n_tokens);
}

ggml_tensor * llama_moe_profile::get_weights(int32_t il, int64_t n_tokens) const {
    return pimpl->get_weights(il, n_tokens);
}

void llama_moe_profile::accumulate(int32_t il, int64_t n_tokens) {
    pimpl->accumulate(il, n_tokens);
}

void llama_moe_profile::write() {
    pimpl->write();
}

int32_t llama_moe_profile::n_expert() const {
    return pimpl->n_expert;
}
