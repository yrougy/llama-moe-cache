#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_model;

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // profile_path seeds the cache from the routing counts of a --moe-profile-output CSV, static_mode then keeps those experts resident
    // pin_mib locks the hottest experts that are not resident in RAM, 0 = only fault them in without locking
    // cpu_fallback computes the experts that are not resident on the CPU, the GPU reads a zero slot for them, so nothing is uploaded
    // tail_skip drops the non-resident experts selected only at a rank at or above this value: the CPU fallback skips
    // their ids, the other modes map them to the zero slot, 0 = disabled
    // adapt_tokens re-ranks the resident tier of a static CPU-fallback cache from the live routing counts every N
    // cached tokens, so a workload that drifts from the profile pulls its hot experts into the bank, 0 = disabled
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
                     const std::string & profile_path, bool static_mode, int64_t pin_mib, bool cpu_fallback, int32_t tail_skip, int32_t adapt_tokens);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the map of the experts of layer il that the CPU computes, nullptr if the cache has no CPU fallback
    ggml_tensor * get_cpu_map(int32_t il) const;

    // the non-resident experts selected only at a rank at or above this value are dropped, 0 = disabled
    int32_t tail_skip() const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // write the per (layer, expert) selection counts and residency to a CSV file
    void export_stats(const std::string & path) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
