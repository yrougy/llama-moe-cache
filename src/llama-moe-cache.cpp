#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __unix__
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace {

// LRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
// in static mode every layer owns a disjoint slot range, so no layer can evict a slot another layer is reading
struct moe_cache_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // doubly linked list of the slots of each chain, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    std::vector<int32_t> lo;    // [n_layer] first scratch slot of the layer
    std::vector<int32_t> hi;    // [n_layer] one past the last scratch slot of the layer
    std::vector<int32_t> chain; // [n_layer] chain used by the layer
    std::vector<int32_t> tier_lo; // [n_layer] first pinned slot of a static cache layer, the tier re-ranking swaps experts inside its range
    std::vector<int32_t> head;  // [n_chain]
    std::vector<int32_t> tail;  // [n_chain]

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    // tail_skip: the experts selected only at a rank at or above this value are dropped, 0 = disabled; n_expert_used
    // gives the rank of an id in the flat ids array; minrank[e] is the best rank of e in the current ids
    int32_t tail_skip = 0;
    int32_t n_expert_used = 0;
    std::vector<int16_t> minrank; // [n_expert]

    // static mode: the first slots of each layer range hold the experts seeded from the routing profile and are
    // never evicted, the rest of the range is an LRU serving the experts the profile did not rank resident
    bool     static_mode = false;
    int32_t  n_resident  = 0;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        lo.assign(n_layer, 0);
        hi.assign(n_layer, n_slots);
        chain.assign(n_layer, 0);
        tier_lo.assign(n_layer, 0);
        head.assign(n_layer, 0);
        tail.assign(n_layer, n_slots - 1);
        seen.assign(n_expert, 0);
        minrank.resize(n_expert);
    }

    // give a slot to each expert of layer il in the order of ranked, at most q of them
    // consecutive experts get consecutive slots, so the uploads of a hit batch merge
    void seed(int32_t il, const std::vector<int32_t> & ranked, int32_t q) {
        int32_t * slots = slot_map[il];
        for (int32_t i = 0; i < (int32_t) ranked.size() && i < q; ++i) {
            const int32_t e = ranked[i];
            GGML_ASSERT(e >= 0 && e < n_expert && slots[e] < 0);
            const int32_t s = lo[il]++;
            GGML_ASSERT(s < hi[il]);
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            n_resident++;
        }
    }

    // the LRU chain of each layer covers only its scratch slots [lo[il], hi[il]), the seeded slots are outside
    void init_chains(const std::vector<int32_t> & layers) {
        for (int32_t il : layers) {
            const int32_t c = chain[il];
            for (int32_t s = lo[il]; s < hi[il]; ++s) {
                prev[s] = s - 1;
                next[s] = s + 1 < hi[il] ? s + 1 : -1;
            }
            head[c] = lo[il] < hi[il] ? lo[il] : -1;
            tail[c] = hi[il] - 1;
            if (head[c] >= 0) {
                prev[head[c]] = -1;
            }
        }
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // move slot s to the tail (most recently used) of the chain of layer il
    void touch(int32_t s, int32_t il) {
        const int32_t c = chain[il];
        if (s == tail[c]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[c] = next[s];
        }
        prev[next[s]] = prev[s];

        prev[s] = tail[c];
        next[s] = -1;
        next[tail[c]] = s;
        tail[c] = s;
    }

    // give a slot to each expert selected by ids in layer il, the misses evict the least recently used scratch slot
    // in static mode the seeded experts are never evicted, the rest of the range is an LRU serving the experts the profile did not rank resident
    // an expert that is only selected at a rank at or above tail_skip and is not resident is dropped: its slot becomes
    // the zero slot the GPU reads, so its contribution is 0 and nothing is uploaded for it
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit, size_t & n_scratch, size_t & n_skip) {
        fills.clear();
        n_hit = 0;
        n_scratch = 0;
        n_skip = 0;

        const int32_t c = chain[il];

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            const int32_t e = ids[i];
            if (seen[e] != seen_gen) {
                seen[e] = seen_gen;
                uniq.push_back(e);
                minrank[e] = (int16_t) n_expert_used;
            }
            const int16_t r = (int16_t)(i % n_expert_used);
            if (r < minrank[e]) {
                minrank[e] = r;
            }
        }

        int32_t * slots = slot_map[il];

        size_t n_need = 0;
        for (int32_t e : uniq) {
            if (slots[e] >= 0 && slots[e] < n_slots) {
                n_hit++;
            }
            else if (tail_skip > 0 && minrank[e] >= tail_skip) {
                slots[e] = n_slots; // the zero slot, the GPU reads it and adds nothing
                n_skip++;
            }
            else {
                n_need++;
            }
        }
        if (n_hit + n_need > (size_t) (hi[il] - lo[il])) {
            return false;
        }

        // hits go to the tail first, so the head can be evicted below
        for (int32_t e : uniq) {
            if (slots[e] >= lo[il] && slots[e] < n_slots) {
                touch(slots[e], il);
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0 && slots[e] < n_slots) {
                continue;
            }
            const int32_t s = head[c];
            if (s < 0) {
                return false;
            }
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            touch(s, il);
            if (static_mode) {
                n_scratch++;
            }
            fills.push_back({ e, s });
        }
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

// the per-(layer,expert) routing counts written by --moe-profile-output
// ranked[il] holds the experts of layer il in the order of descending use, ties by ascending index
// counts[il*n_expert + e] holds the raw counts, they seed the moving average of the tier re-ranking
static bool llama_moe_cache_load_profile(const char * path, int32_t n_layer, int32_t n_expert,
        std::vector<std::vector<int32_t>> & ranked, std::vector<uint64_t> & counts) {
    FILE * f = std::fopen(path, "r");
    if (f == nullptr) {
        LLAMA_LOG_WARN("%s: failed to open %s\n", __func__, path);
        return false;
    }

    char line[256];
    if (std::fgets(line, sizeof(line), f) == nullptr || std::strcmp(line, "layer,expert,count,weight_sum\n") != 0) {
        LLAMA_LOG_WARN("%s: %s has no routing counts\n", __func__, path);
        std::fclose(f);
        return false;
    }

    std::vector<std::vector<std::pair<uint64_t, int32_t>>> heat(n_layer);
    int n_rows = 0;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        int il, e;
        unsigned long long c;
        double w;
        if (std::sscanf(line, "%d,%d,%llu,%lf", &il, &e, &c, &w) != 4) {
            continue;
        }
        if (il < 0 || il >= n_layer || e < 0 || e >= n_expert) {
            continue;
        }
        heat[il].push_back({ c, e });
        n_rows++;
    }
    std::fclose(f);

    if (n_rows == 0) {
        LLAMA_LOG_WARN("%s: %s is empty\n", __func__, path);
        return false;
    }

    ranked.assign(n_layer, {});
    counts.assign((size_t) n_layer*n_expert, 0);
    for (int32_t il = 0; il < n_layer; ++il) {
        std::sort(heat[il].begin(), heat[il].end(), [](const auto & a, const auto & b) {
            if (a.first != b.first) {
                return a.first > b.first;
            }
            return a.second < b.second;
        });
        for (const auto & h : heat[il]) {
            ranked[il].push_back(h.second);
            counts[(size_t) il*n_expert + h.second] = h.first;
        }
    }

    LLAMA_LOG_INFO("%s: %s: %d (layer,expert) pairs, %d layers with routing data\n",
        __func__, path, n_rows, (int) std::count_if(heat.begin(), heat.end(), [](const auto & v) { return !v.empty(); }));
    return true;
}

// the RAM the system can fault in, 0 if unknown
static int64_t llama_moe_cache_ram_available(void) {
#ifdef __linux__
    FILE * f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr) {
        return 0;
    }
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
        unsigned long long kb;
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
        std::fclose(f);
        return (int64_t) kb * 1024;
        }
    }
    std::fclose(f);
#endif
    return 0;
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        int32_t n_extra = 1;              // slots past n_slots: the zero slot the GPU reads for the CPU experts, and the padding CUDA reads past the last one
        bool cpu_fallback = false;        // the experts that are not resident are computed on the CPU, nothing is uploaded
        int64_t max_batch_cached = max_batch; // tokens a batch can have before the group stops caching
    int64_t adapt_done = 0; // tokens cached since this group's last tier re-ranking
        moe_cache_lru lru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * cpu_map  = nullptr;   // I32 [1, n_expert] in host memory, -1 for the experts the GPU computes
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits    = 0;
        size_t misses  = 0;
        size_t bytes   = 0;
        size_t scratch = 0; // misses that took a scratch slot, the experts the profile did not seed resident
        size_t skipped = 0; // misses that tail_skip dropped instead of computing them on the CPU
    };

    static constexpr int64_t max_batch = 32;

    // in static mode each layer's scratch pool serves this many tokens, batches above it are not cached
    // 4 = the size of a verified MTP draft batch, so speculative decoding stays on the cached path
    static constexpr int64_t max_batch_scratch = 4;

    // above this batch size the GPU uses the matrix-multiplication path with ids, whose inverse map has no entry for
    // an expert that is not in the cache, so a cached batch must stay in the vector path, which reads the zero slot
    static constexpr int64_t max_batch_mmvq = 4;

    bool static_mode;

    // bytes of the hottest non-resident experts locked in RAM, 0 = populate only, never lock
    int64_t pin_bytes;

    // the experts that are not resident in the cache are computed on the CPU, the GPU reads a zero slot for them
    bool cpu_fallback;

    // with the CPU fallback, the non-resident experts selected at a rank at or above this value are skipped: the CPU op
    // skips their ids and the GPU reads the zero slot for them, so their contribution is 0
    int32_t tail_skip;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;

    std::vector<uint64_t> use; // [n_layer*n_expert] selections seen by the copy callback, the runtime routing counts

    int32_t adapt_tokens = 0; // re-rank the resident tier of a static CPU-fallback cache every N cached tokens from the live routing counts, 0 = disabled
    std::vector<uint64_t> prev_use; // [n_layer*n_expert] selection counts at the previous re-ranking
    std::vector<uint64_t> heat;     // [n_layer*n_expert] exponential moving average of the per-window selection counts, what the tier is ranked by

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    // the experts of the cached layers that the profile did not seed resident are the complement: their pages are
    // faulted into RAM so a decode miss never reads the GGUF from disk, the hottest are locked so they can never
    // be evicted, and the pages of the resident ones are dropped from the page cache after their upload to the banks
    void cache_complement(const llama_model & model, const std::vector<std::vector<int32_t>> & ranked) {
        if (!llama_mmap::POPULATE_SUPPORTED && !llama_mmap::PIN_SUPPORTED) {
            LLAMA_LOG_WARN("%s: keeping the MoE complement in RAM is not supported on this platform\n", __func__);
            return;
        }

        const int32_t n_expert = (int32_t) model.hparams.n_expert;

        struct expert_range {
            llama_mmap * mm;
            size_t  first;
            size_t  last;
            bool    resident;
        };

        std::vector<expert_range> ranges;
        std::vector<std::vector<int32_t>> idx(model.layers.size()*n_expert); // il*n_expert + e -> ranges of all expert tensors

        for (size_t il = 0; il < model.layers.size(); ++il) {
            if (layers[il].ig < 0) {
                continue;
            }
            for (const ggml_tensor * w : layers[il].experts) {
                size_t first, last;
                llama_mmap * mm = model.mmap_range(w, first, last);
                if (mm == nullptr) {
                    LLAMA_LOG_WARN("%s: the experts of layer %zu are not mmap'ed, the MoE complement is not kept in RAM\n", __func__, il);
                    return;
                }
                const size_t expert_size = w->nb[2];
                for (int32_t e = 0; e < n_expert; ++e) {
                    idx[il*n_expert + e].push_back((int32_t) ranges.size());
                    ranges.push_back({ mm, first + e*expert_size, first + (e + 1)*expert_size, false });
                }
            }
        }

        // a seeded expert is in the bank, its pages are no longer read during decode
        for (const group & g : groups) {
            for (int32_t s = 0; s < g.n_slots; ++s) {
                const int32_t key = g.lru.key_of[s];
                if (key < 0) {
                    continue;
                }
                for (const int32_t ir : idx[key]) {
                    ranges[ir].resident = true;
                }
            }
        }

        // merge the contiguous ranges of a mapping so one system call covers as many experts as possible
        auto merge = [&](const std::vector<expert_range> & v) {
            std::vector<std::pair<llama_mmap *, std::pair<size_t, size_t>>> res;
            for (const expert_range & r : v) {
                if (!res.empty() && res.back().first == r.mm && res.back().second.second == r.first) {
                    res.back().second.second = r.last;
                } else {
                    res.emplace_back(r.mm, std::make_pair(r.first, r.last));
                }
            }
            return res;
        };

        auto by_file_pos = [](const expert_range & a, const expert_range & b) {
            return a.mm != b.mm ? a.mm < b.mm : a.first < b.first;
        };

        std::vector<expert_range> cold; // resident experts, their pages can be dropped
        for (const expert_range & r : ranges) {
            if (r.resident) {
                cold.push_back(r);
            }
        }
        std::sort(cold.begin(), cold.end(), by_file_pos);

        const double mib = 1024.0*1024.0;

        size_t cold_bytes = 0;
        for (const auto & m : merge(cold)) {
            m.first->release(m.second.first, m.second.second);
            cold_bytes += m.second.second - m.second.first;
        }

        // the complement must fit in the RAM the system can actually give: fault in the hottest experts first and
        // stop at the budget, the coldest ones stay cold and their misses read the GGUF from disk
        int64_t budget = llama_moe_cache_ram_available();
        if (budget > 0) {
            budget -= 1024*1024*1024; // keep a margin for the process itself
        } else {
            budget = INT64_MAX / 4; // the available RAM is unknown, fault in everything
        }

        // the bytes of the GGUF that are not expert tensors of the cached layers are read on a cold cache during decode,
        // fault them in first so the decode hot path never goes to the disk for them
        size_t rest_bytes = 0;
        {
            std::vector<expert_range> all = ranges;
            std::sort(all.begin(), all.end(), by_file_pos);

            llama_mmap * mm  = nullptr;
            size_t           pos = 0;
            auto flush = [&](size_t first, size_t last) {
                if (first < last) {
                    mm->populate(first, last);
                    rest_bytes += last - first;
                }
            };
            for (const expert_range & r : all) {
                if (r.mm != mm) {
                    if (mm != nullptr) {
                        flush(pos, mm->size());
                    }
                    mm  = r.mm;
                    pos = 0;
                }
                flush(pos, r.first);
                pos = std::max(pos, r.last);
            }
            if (mm != nullptr) {
                flush(pos, mm->size());
            }
        }
        budget -= (int64_t) rest_bytes;

        // select the ranges of the complement to fault in, the hottest first, until the budget is exhausted
        std::vector<char> selected(ranges.size(), 0);
        size_t warm_bytes = 0;
        size_t left_bytes = 0; // the complement left out of RAM, its misses read the disk
        auto select = [&](int32_t ir) {
            const size_t len = ranges[ir].last - ranges[ir].first;
            if ((int64_t) len > budget) {
                left_bytes += len;
                return;
            }
            selected[ir] = 1;
            warm_bytes   += len;
            budget       -= (int64_t) len;
        };
        for (size_t il = 0; il < model.layers.size(); ++il) {
            if (layers[il].ig < 0) {
                continue;
            }
            for (const int32_t e : ranked[il]) {
                for (const int32_t ir : idx[il*n_expert + e]) {
                    if (!ranges[ir].resident) {
                        select(ir);
                    }
                }
            }
        }
        // the experts without routing data in the profile, in file order
        for (size_t ir = 0; ir < ranges.size(); ++ir) {
            if (!ranges[ir].resident && !selected[ir]) {
                select((int32_t) ir);
            }
        }

        // merge the selected ranges so one system call covers as many experts as possible
        std::vector<expert_range> warm;
        for (size_t ir = 0; ir < ranges.size(); ++ir) {
            if (selected[ir]) {
                warm.push_back(ranges[ir]);
            }
        }
        std::sort(warm.begin(), warm.end(), by_file_pos);
        for (const auto & m : merge(warm)) {
            m.first->populate(m.second.first, m.second.second);
        }

        // lock the hottest experts of the complement, stop at the limit of the process
        // the locked bytes must also fit in RLIMIT_MEMLOCK, mlock fails with ENOMEM beyond it; the accounting uses
        // the page-aligned lengths that mlock actually locks, so only a small slack is kept
        int64_t pin_limit = pin_bytes;
#ifdef __unix__
        {
            struct rlimit rl;
            if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY) {
                const int64_t cap = (int64_t) rl.rlim_cur - 4*1024*1024;
                if (cap < pin_limit) {
                    pin_limit = cap;
                }
            }
        }
#endif
#ifdef __unix__
        const size_t page_size = sysconf(_SC_PAGESIZE);
#else
        const size_t page_size = 4096;
#endif
        size_t  pinned = 0;
        size_t  n_pinned = 0;
        bool    stop = false;
        for (size_t il = 0; il < model.layers.size() && !stop; ++il) {
            if (layers[il].ig < 0) {
                continue;
            }
            for (const int32_t e : ranked[il]) {
                for (const int32_t ir : idx[il*n_expert + e]) {
                    const expert_range & r = ranges[ir];
                    if (r.resident || !selected[ir]) {
                        continue;
                    }
                    const size_t first_a = r.first & ~(page_size - 1);
                    const size_t last_a  = std::min((r.last + page_size - 1) & ~(page_size - 1), r.mm->size());
                    const size_t len     = last_a > first_a ? last_a - first_a : 0;
                    if ((int64_t) pinned + (int64_t) len > pin_limit) {
                        stop = true;
                        break;
                    }
                    if (!r.mm->pin(r.first, r.last)) {
                        stop = true;
                        break;
                    }
                    pinned   += len;
                    n_pinned++;
                }
                if (stop) {
                    break;
                }
            }
        }

        LLAMA_LOG_INFO("%s: MoE complement: %.2f MiB faulted into RAM, %.2f MiB of the rest faulted into RAM, %.2f MiB pinned in %zu experts, %.2f MiB of resident pages dropped\n",
                __func__, warm_bytes/mib, rest_bytes/mib, pinned/mib, n_pinned, cold_bytes/mib);
        if (left_bytes > 0) {
            LLAMA_LOG_WARN("%s: MoE complement: %.2f MiB left out of RAM, their misses will read the GGUF from disk\n", __func__, left_bytes/mib);
        }
    }

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
            const std::string & profile_path, bool static_mode_in, int64_t pin_mib, bool cpu_fallback_in, int32_t tail_skip_in, int32_t adapt_tokens_in) :
            static_mode(static_mode_in), pin_bytes((int64_t) pin_mib*1024*1024), cpu_fallback(cpu_fallback_in), tail_skip(tail_skip_in), n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()),
            use(model.layers.size()*model.hparams.n_expert, 0) {
        if (tail_skip != 0 && (tail_skip < 1 || tail_skip >= n_expert_used)) {
            LLAMA_LOG_WARN("%s: MoE tail skip needs a rank between 1 and %d, disabled\n", __func__, n_expert_used - 1);
            tail_skip = 0;
        }
        if (adapt_tokens_in > 0) {
            if (!static_mode || !cpu_fallback) {
                LLAMA_LOG_WARN("%s: MoE cache adaptation needs a static cache with the CPU fallback, disabled\n", __func__);
            } else {
                adapt_tokens = adapt_tokens_in;
            }
        }
        prev_use.assign(use.size(), 0);
        heat.assign(use.size(), 0);
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + g.n_extra), alignment);
            }
            return res;
        };

        for (group & g : groups) {
            // with the CPU fallback or a tail skip, slot n_slots is the all-zero slot the GPU reads for an expert it
            // must not compute, and one more slot is kept so CUDA reading past it stays inside the buffer
            g.n_extra = (cpu_fallback || tail_skip > 0) ? 2 : 1;
            g.cpu_fallback = cpu_fallback;
        }

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            g.lru.tail_skip     = tail_skip;
            g.lru.n_expert_used = n_expert_used;
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(cpu_fallback ? 2 : 1);
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + g.n_extra);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    // the view includes the zero slot the GPU reads for the experts the CPU computes
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + 1, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (cpu_fallback) {
                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        }

        if (static_mode && !model.hparams.no_alloc) {
            std::vector<std::vector<int32_t>> ranked;
            std::vector<uint64_t>             counts;
            if (profile_path.empty() ||
                    !llama_moe_cache_load_profile(profile_path.c_str(), model.layers.size(), model.hparams.n_expert, ranked, counts)) {
                LLAMA_LOG_WARN("%s: a static MoE cache needs a routing profile, using LRU instead\n", __func__);
                static_mode = false;
                adapt_tokens = 0;
            }
            if (adapt_tokens > 0) {
                heat = std::move(counts); // the profile counts are the prior of the moving average: the tier only moves when the live routing persistently contradicts the seed
            }
            for (size_t ig = 0; static_mode && ig < groups.size(); ++ig) {
                group & g = groups[ig];
                if (g.n_slots == 0) {
                    continue;
                }
                const int32_t n_layers = (int32_t) g.layers.size();

                g.lru.static_mode = true;
                g.lru.n_resident  = 0;

                int32_t q = 0;
                if (cpu_fallback) {
                    // there is no LRU and no scratch, every slot is resident for the whole run: the GPU reads the zero
                    // slot for an expert the CPU computes, so its contribution is exactly 0 and nothing is uploaded
                    q = g.n_slots / n_layers;
                    int32_t extra = q > 0 ? g.n_slots % n_layers : 0;
                    int32_t base  = 0;
                    for (int32_t il : g.layers) {
                        const int32_t qi = q + (extra > 0 ? 1 : 0);
                        extra    -= (extra > 0 ? 1 : 0);
                        g.lru.chain[il]   = il;
                        g.lru.tier_lo[il] = base;
                        g.lru.lo[il]      = base;
                        g.lru.hi[il]      = base + qi;
                        base += qi;
                        g.lru.seed(il, ranked[il], qi);
                    }
                    g.max_batch_cached = max_batch_mmvq;

                    size_t n_cpu = 0;
                    for (int32_t il : g.layers) {
                        int32_t * slots = g.lru.slot_map[il];
                        int32_t * cpu   = (int32_t *) layers[il].cpu_map->data;
                        for (int32_t e = 0; e < g.lru.n_expert; ++e) {
                            if (slots[e] >= 0) {
                                cpu[e] = -1;
                            }
                            else {
                                slots[e] = g.n_slots;
                                cpu[e]   = e;
                                n_cpu++;
                            }
                        }
                    }
                    LLAMA_LOG_INFO("%s: CPU fallback: %d resident experts, %d experts left to the CPU, quota %d per layer, up to %d tokens per batch\n",
                        __func__, g.lru.n_resident, (int) n_cpu, q, (int) g.max_batch_cached);
                }
                else {
                    // every layer owns a disjoint range of slots, so no layer can evict a slot another layer is reading:
                    // the first slots of the range hold the experts seeded from the profile, the rest is that layer's LRU
                    const int32_t n_scratch = std::min(g.n_slots / 2, (int32_t)(n_expert_used * n_layers * max_batch_scratch));
                    if (n_scratch < n_expert_used * n_layers) {
                        LLAMA_LOG_WARN("%s: MoE cache is too small to be static, using LRU instead\n", __func__);
                        static_mode = false;
                        adapt_tokens = 0;
                        break;
                    }

                    const int32_t budget   = g.n_slots - n_scratch;
                    q                        = budget / n_layers;
                    int32_t       extra    = q > 0 ? budget % n_layers : 0;
                    int32_t       sc       = n_scratch / n_layers;
                    int32_t       sc_extra = n_scratch % n_layers;
                    int32_t       base     = 0;
                    int64_t       mb       = max_batch;
                    for (int32_t il : g.layers) {
                        const int32_t qi  = q + (extra > 0 ? 1 : 0);
                        const int32_t sci = sc + (sc_extra > 0 ? 1 : 0);
                        extra    -= (extra > 0 ? 1 : 0);
                        sc_extra -= (sc_extra > 0 ? 1 : 0);
                        g.lru.chain[il]   = il;
                        g.lru.tier_lo[il] = base;
                        g.lru.lo[il]      = base;
                        g.lru.hi[il]      = base + qi + sci;
                        base += qi + sci;
                        g.lru.seed(il, ranked[il], qi);
                        mb = std::min(mb, (int64_t) sci / n_expert_used);
                    }
                    g.max_batch_cached = mb;

                    LLAMA_LOG_INFO("%s: static MoE cache: %d resident experts, %d scratch slots, quota %d per layer, up to %d tokens per batch\n",
                        __func__, g.lru.n_resident, n_scratch, q, (int) g.max_batch_cached);
                }

                g.lru.init_chains(g.layers);

                // the seeded slots are resident from the first ubatch, so their bytes must be in the bank now
                size_t seeded_bytes = 0;
                for (int32_t s = 0; s < g.n_slots; ++s) {
                    const int32_t key = g.lru.key_of[s];
                    if (key < 0) {
                        continue;
                    }
                    const int32_t il  = key / g.lru.n_expert;
                    const int32_t e   = key % g.lru.n_expert;
                    for (size_t ip = 0; ip < layers[il].experts.size(); ++ip) {
                        const ggml_tensor * w    = layers[il].experts[ip];
                        ggml_tensor       * bank = g.banks[ip];
                        const size_t expert_size = w->nb[2];
                        ggml_backend_tensor_set(bank, (const uint8_t *) w->data + e*expert_size, s*expert_size, expert_size);
                        seeded_bytes += expert_size;
                    }
                }
                LLAMA_LOG_INFO("%s: uploaded %.2f MiB of seeded experts\n", __func__, seeded_bytes/1024.0/1024.0);
            }

            if (static_mode) {
                cache_complement(model, ranked);
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        log_stats();
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];
        const group & g = groups[l.ig];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > g.max_batch_cached) {
            return nullptr;
        }
        // with the CPU fallback the slots never change, so the number of tokens is not limited by the slots of the layer
        if (!g.cpu_fallback && n_tokens*n_expert_used > g.lru.hi[il] - g.lru.lo[il]) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_cpu_map(int32_t il) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        return layers[il].cpu_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        // a slot of n_slots is the zero slot of a dropped expert, it is not the expert and must not be copied
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0 || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    // live re-ranking of the resident tier of a static CPU-fallback cache: the tier follows an exponential moving
    // average of the per-window selection counts, so a workload that drifts from the profile pulls its hot experts
    // into the bank and pushes the cold ones back to the CPU, while one short noisy window cannot override the seed.
    // a promoted expert is written into the slot of a demoted one, the maps are updated in place before this batch's
    // slot map is copied to the device, so the batch sees one consistent tier
    void adapt_tier(group & g, ggml_backend_t backend) {
        const int32_t n_expert = g.lru.n_expert;
        int32_t n_swap = 0;

        std::vector<uint64_t> delta(n_expert);
        std::vector<int32_t>  order(n_expert);
        std::vector<char>     in_target(n_expert, 0);
        std::vector<int32_t>  demote, promote;

        for (int32_t il : g.layers) {
            const int32_t tier0 = g.lru.tier_lo[il];
            const int32_t q     = (g.cpu_fallback ? g.lru.hi[il] : g.lru.lo[il]) - tier0; // pinned slots of the layer
            if (q <= 0) {
                continue;
            }
            int32_t * slots = g.lru.slot_map[il];
            const uint64_t * use_row  = use.data() + (size_t) il*n_expert;
            uint64_t *       prev_row = prev_use.data() + (size_t) il*n_expert;
            uint64_t *       heat_row = heat.data() + (size_t) il*n_expert;

            for (int32_t e = 0; e < n_expert; ++e) {
                delta[e]    = use_row[e] - prev_row[e];
                prev_row[e] = use_row[e];
                heat_row[e] = heat_row[e] - heat_row[e]/4 + delta[e]; // the window counts in, three quarters of the history out
                order[e]    = e;
            }
            const auto resident = [&](int32_t e) { return slots[e] >= 0 && slots[e] < g.lru.n_slots; };
            std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
                if (heat_row[a] != heat_row[b]) {
                    return heat_row[a] > heat_row[b];
                }
                if (resident(a) != resident(b)) {
                    return resident(a); // on ties, keep the currently resident experts
                }
                return a < b;
            });

            std::fill(in_target.begin(), in_target.end(), 0);
            for (int32_t i = 0; i < q; ++i) {
                in_target[order[i]] = 1;
            }
            promote.clear();
            demote.clear();
            for (int32_t i = 0; i < q; ++i) {
                if (!resident(order[i])) {
                    promote.push_back(order[i]);
                }
            }
            for (int32_t e = 0; e < n_expert && (int32_t) demote.size() < (int32_t) promote.size(); ++e) {
                if (resident(e) && slots[e] >= tier0 && slots[e] < tier0 + q && !in_target[e]) {
                    demote.push_back(e);
                }
            }
            // bound the upload cost of a re-ranking: the tier converges over successive windows
            const int32_t cap = q/4 > 0 ? q/4 : 1;
            if ((int32_t) demote.size() > cap) {
                demote.resize(cap);
                promote.resize(cap);
            }

            int32_t * cpu = g.cpu_fallback ? (int32_t *) layers[il].cpu_map->data : nullptr;
            for (size_t i = 0; i < demote.size(); ++i) {
                const int32_t s  = slots[demote[i]];
                const int32_t e1 = demote[i];
                const int32_t e2 = promote[i];
                slots[e1] = g.cpu_fallback ? g.lru.n_slots : -1;
                slots[e2] = s;
                g.lru.key_of[s] = il*n_expert + e2;
                if (cpu != nullptr) {
                    cpu[e1] = e1;
                    cpu[e2] = -1;
                }
                for (size_t ip = 0; ip < layers[il].experts.size(); ++ip) {
                    const ggml_tensor * w = layers[il].experts[ip];
                    ggml_backend_tensor_set_async(backend, g.banks[ip],
                            (const uint8_t *) w->data + (size_t) e2*w->nb[2], (size_t) s*w->nb[2], w->nb[2]);
                }
                n_swap++;
            }
        }
        if (n_swap > 0) {
            LLAMA_LOG_INFO("%s: %d experts swapped into the resident tier\n", __func__, n_swap);
        }
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        if (g.cpu_fallback) {
            // the slot map is static, the experts the CPU computes read the zero slot, so there is nothing to upload
            const int32_t * slots = g.lru.slot_map[il];
            size_t n_gpu = 0;
            size_t n_skip = 0;
            for (size_t i = 0; i < ids.size(); ++i) {
                use[(size_t) il*g.lru.n_expert + ids[i]]++;
                if (slots[ids[i]] < g.n_slots) {
                    n_gpu++;
                }
                else if (tail_skip > 0 && (int32_t)(i % n_expert_used) >= tail_skip) {
                    n_skip++;
                }
            }
            if (adapt_tokens > 0 && il == g.layers.front()) {
                // the first layer of the group is copied once per cached batch: count the batch's tokens there
                g.adapt_done += ids.size() / n_expert_used;
                if (g.adapt_done >= adapt_tokens) {
                    g.adapt_done = 0;
                    adapt_tier(g, backend);
                }
            }
            stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
            st.hits   += n_gpu;
            st.misses += ids.size() - n_gpu;
            st.skipped += n_skip;
            ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
            return true;
        }

        size_t n_hit = 0;
        size_t n_scratch = 0;
        size_t n_skip = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit, n_scratch, n_skip)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        for (size_t i = 0; i < ids.size(); ++i) {
            use[(size_t) il*g.lru.n_expert + ids[i]]++;
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;
        st.scratch += n_scratch;
        st.skipped += n_skip;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB, scratch = %zu, skipped = %zu\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0, st.scratch, st.skipped);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }

        // how much of the model is resident, and how many expert computations the cache served
        int32_t n_expert = 0;
        for (const layer & l : layers) {
            if (l.slot_map != nullptr) {
                n_expert = l.slot_map->ne[1];
                break;
            }
        }
        if (n_expert == 0) {
            return;
        }

        size_t n_cached_layers = 0;
        size_t n_resident      = 0;
        size_t resident_bytes  = 0;
        size_t gpu_experts     = 0;
        size_t cached_experts  = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            if (layers[il].ig < 0) {
                continue;
            }
            n_cached_layers++;
            const group & g = groups[layers[il].ig];
            const int32_t * slots = g.lru.slot_map[il];
            if (slots == nullptr) {
                continue;
            }
            for (int32_t e = 0; e < n_expert; ++e) {
                cached_experts += use[il*n_expert + e];
                if (slots[e] >= 0 && slots[e] < g.n_slots) {
                    n_resident++;
                    for (const ggml_tensor * ref : g.ref) {
                        resident_bytes += ref->nb[2];
                    }
                    gpu_experts += use[il*n_expert + e];
                }
            }
        }

        const size_t tokens = cached_experts / n_expert_used;
        LLAMA_LOG_INFO("llama_moe_cache: %zu/%zu layers cached, %zu/%zu experts resident (%.2f MiB), %zu tokens served by the GPU cache, %zu expert computations left to the CPU\n",
            n_cached_layers, layers.size(), n_resident, layers.size()*n_expert, resident_bytes/1024.0/1024.0,
            tokens, cached_experts - gpu_experts + (layers.size() - n_cached_layers)*tokens*n_expert_used);
    }

    void export_stats(const char * path) const {
        int32_t n_expert = 0;
        for (const layer & l : layers) {
            if (l.slot_map != nullptr) {
                n_expert = l.slot_map->ne[1];
                break;
            }
        }
        if (n_expert == 0) {
            return;
        }

        const std::string tmp = std::string(path) + ".tmp";
        FILE * f = std::fopen(tmp.c_str(), "w");
        if (f == nullptr) {
            LLAMA_LOG_WARN("%s: failed to open %s\n", __func__, tmp.c_str());
            return;
        }

        fprintf(f, "layer,expert,use,resident\n");
        for (size_t il = 0; il < layers.size(); ++il) {
            const int32_t ig = layers[il].ig;
            const int32_t * slots = ig < 0 ? nullptr : groups[ig].lru.slot_map[il];
            for (int32_t e = 0; e < n_expert; ++e) {
                const bool resident = slots != nullptr && slots[e] >= 0 && slots[e] < groups[ig].n_slots;
                fprintf(f, "%zu,%d,%zu,%d\n", il, e, use[il*n_expert + e], resident ? 1 : 0);
            }
        }

        if (std::fclose(f) != 0) {
            LLAMA_LOG_WARN("%s: failed to close %s\n", __func__, tmp.c_str());
            return;
        }
        if (std::rename(tmp.c_str(), path) != 0) {
            LLAMA_LOG_WARN("%s: failed to rename %s to %s\n", __func__, tmp.c_str(), path);
            return;
        }
        LLAMA_LOG_INFO("%s: wrote %s\n", __func__, path);
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
        const std::string & profile_path, bool static_mode, int64_t pin_mib, bool cpu_fallback, int32_t tail_skip, int32_t adapt_tokens) :
    pimpl(new impl(model, backends, bufts, size, profile_path, static_mode, pin_mib, cpu_fallback, tail_skip, adapt_tokens)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_cpu_map(int32_t il) const {
    return pimpl->get_cpu_map(il);
}

int32_t llama_moe_cache::tail_skip() const {
    return pimpl->tail_skip;
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}

void llama_moe_cache::export_stats(const std::string & path) const {
    pimpl->export_stats(path.c_str());
}
