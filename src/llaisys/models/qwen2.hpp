#pragma once

#include "../../../include/llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include <vector>

namespace llaisys {

class Qwen2Model {
private:
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    llaisysDeviceType_t device;
    std::vector<int> device_ids;
    
    // KV-Cache
    struct KVCache {
        tensor_t k_cache;
        tensor_t v_cache;
        size_t size;
    };
    std::vector<KVCache> kv_caches;
    
    // Temporary tensors
    tensor_t attn_output;
    tensor_t mlp_output;
    tensor_t layer_output;
    tensor_t hidden_states;
    
    // Initialize KV-Cache
    void init_kv_cache();
    
    // Forward pass functions
    tensor_t forward_embedding(int64_t* token_ids, size_t ntoken);
    tensor_t forward_layer(int layer_idx, tensor_t input, size_t seq_len);
    tensor_t forward_norm(tensor_t input, tensor_t weight, float eps);
    tensor_t forward_linear(tensor_t input, tensor_t weight, tensor_t bias = nullptr);
    
public:
    Qwen2Model(const LlaisysQwen2Meta* meta, llaisysDeviceType_t device, int* device_ids, int ndevice);
    ~Qwen2Model();
    
    LlaisysQwen2Weights* get_weights();
    int64_t infer(int64_t* token_ids, size_t ntoken);
};

} // namespace llaisys
