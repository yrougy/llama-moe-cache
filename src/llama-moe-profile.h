#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct llama_model;

// counts which experts the router selects in each layer, so the MoE cache can be seeded from real usage
// the graph copies the top-k ids and the expert weights of each layer into host tensors owned here, they are summed up after each compute
class llama_moe_profile {
public:
    llama_moe_profile(const llama_model & model, int64_t max_tokens, std::string output_path);
    ~llama_moe_profile();

    // the host tensors that receive the selected expert ids and the expert weights of layer il for n_tokens tokens
    // nullptr if the layer has no experts, or n_tokens does not fit in the buffer
    ggml_tensor * get_ids(int32_t il, int64_t n_tokens) const;
    ggml_tensor * get_weights(int32_t il, int64_t n_tokens) const;

    // add the ids and the weights the graph wrote for layer il and n_tokens tokens
    void accumulate(int32_t il, int64_t n_tokens);

    // write the counts to the output file, CSV: layer,expert,count,weight_sum
    void write();

    int32_t n_expert() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_profile_ptr = std::unique_ptr<llama_moe_profile>;
