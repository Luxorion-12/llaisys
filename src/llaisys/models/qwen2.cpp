#include "qwen2.hpp"
#include "../../core/context/context.hpp"
#include "../../utils/check.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include <cmath>
#include <cstring>

namespace llaisys {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta* meta, llaisysDeviceType_t device, int* device_ids, int ndevice) {
    // Copy meta data
    this->meta = *meta;
    this->device = device;
    
    // Copy device IDs
    for (int i = 0; i < ndevice; ++i) {
        this->device_ids.push_back(device_ids[i]);
    }
    
    // Initialize weights
    weights.in_embed = nullptr;
    weights.out_embed = nullptr;
    weights.out_norm_w = nullptr;
    weights.attn_norm_w = new llaisysTensor_t[meta->nlayer];
    weights.attn_q_w = new llaisysTensor_t[meta->nlayer];
    weights.attn_q_b = new llaisysTensor_t[meta->nlayer];
    weights.attn_k_w = new llaisysTensor_t[meta->nlayer];
    weights.attn_k_b = new llaisysTensor_t[meta->nlayer];
    weights.attn_v_w = new llaisysTensor_t[meta->nlayer];
    weights.attn_v_b = new llaisysTensor_t[meta->nlayer];
    weights.attn_o_w = new llaisysTensor_t[meta->nlayer];
    weights.mlp_norm_w = new llaisysTensor_t[meta->nlayer];
    weights.mlp_gate_w = new llaisysTensor_t[meta->nlayer];
    weights.mlp_up_w = new llaisysTensor_t[meta->nlayer];
    weights.mlp_down_w = new llaisysTensor_t[meta->nlayer];
    
    // Initialize KV-Cache
    init_kv_cache();
    
    // Initialize temporary tensors
    // TODO: Allocate temporary tensors based on meta data
}

Qwen2Model::~Qwen2Model() {
    // Free weights
    delete[] weights.attn_norm_w;
    delete[] weights.attn_q_w;
    delete[] weights.attn_q_b;
    delete[] weights.attn_k_w;
    delete[] weights.attn_k_b;
    delete[] weights.attn_v_w;
    delete[] weights.attn_v_b;
    delete[] weights.attn_o_w;
    delete[] weights.mlp_norm_w;
    delete[] weights.mlp_gate_w;
    delete[] weights.mlp_up_w;
    delete[] weights.mlp_down_w;
    
    // KV-Cache is managed by shared_ptr, no need to delete
    
    // Temporary tensors are managed by shared_ptr, no need to delete
}

void Qwen2Model::init_kv_cache() {
    kv_caches.resize(meta.nlayer);
    
    // For each layer, create KV-Cache tensors
    for (size_t i = 0; i < meta.nlayer; ++i) {
        // Calculate KV-Cache shape: [maxseq, nkvhead, d]
        std::vector<size_t> k_shape = {meta.maxseq, meta.nkvh, meta.dh};
        std::vector<size_t> v_shape = {meta.maxseq, meta.nkvh, meta.dh};
        
        // Get device ID
        int device_id = device_ids.empty() ? 0 : device_ids[0];
        
        // Create KV-Cache tensors
        kv_caches[i].k_cache = Tensor::create(k_shape, meta.dtype, device, device_id);
        kv_caches[i].v_cache = Tensor::create(v_shape, meta.dtype, device, device_id);
        kv_caches[i].size = 0;
    }
}

LlaisysQwen2Weights* Qwen2Model::get_weights() {
    return &weights;
}

 tensor_t Qwen2Model::forward_embedding(int64_t* token_ids, size_t ntoken) {
    // Create input tensor for token IDs
    std::vector<size_t> index_shape = {ntoken};
    int device_id = device_ids.empty() ? 0 : device_ids[0];
    tensor_t index = Tensor::create(index_shape, LLAISYS_DTYPE_I64, device, device_id);
    
    // Copy token IDs to input tensor
    std::byte* index_data = index->data();
    memcpy(index_data, token_ids, ntoken * sizeof(int64_t));
    
    // Create output tensor
    std::vector<size_t> out_shape = {ntoken, meta.hs};
    tensor_t out = Tensor::create(out_shape, meta.dtype, device, device_id);
    
    // Get embedding weights
    // TODO: Convert llaisysTensor_t to tensor_t
    tensor_t weight = Tensor::create({meta.voc, meta.hs}, meta.dtype, device, device_id);
    
    // Perform embedding lookup
    llaisys::ops::embedding(out, index, weight);
    
    return out;
}

 tensor_t Qwen2Model::forward_layer(int layer_idx, tensor_t input, size_t seq_len) {
    // Get layer weights
    // TODO: Convert llaisysTensor_t to tensor_t
    int device_id = device_ids.empty() ? 0 : device_ids[0];
    tensor_t attn_norm_w = Tensor::create({meta.hs}, meta.dtype, device, device_id);
    tensor_t attn_q_w = Tensor::create({meta.hs, meta.hs}, meta.dtype, device, device_id);
    tensor_t attn_q_b = Tensor::create({meta.hs}, meta.dtype, device, device_id);
    tensor_t attn_k_w = Tensor::create({meta.hs, meta.nkvh * meta.dh}, meta.dtype, device, device_id);
    tensor_t attn_k_b = Tensor::create({meta.nkvh * meta.dh}, meta.dtype, device, device_id);
    tensor_t attn_v_w = Tensor::create({meta.hs, meta.nkvh * meta.dh}, meta.dtype, device, device_id);
    tensor_t attn_v_b = Tensor::create({meta.nkvh * meta.dh}, meta.dtype, device, device_id);
    tensor_t attn_o_w = Tensor::create({meta.hs, meta.hs}, meta.dtype, device, device_id);
    tensor_t mlp_norm_w = Tensor::create({meta.hs}, meta.dtype, device, device_id);
    tensor_t mlp_gate_w = Tensor::create({meta.hs, meta.di}, meta.dtype, device, device_id);
    tensor_t mlp_up_w = Tensor::create({meta.hs, meta.di}, meta.dtype, device, device_id);
    tensor_t mlp_down_w = Tensor::create({meta.di, meta.hs}, meta.dtype, device, device_id);
    
    // 1. RMS Norm for attention
    tensor_t attn_norm_out = forward_norm(input, attn_norm_w, meta.epsilon);
    
    // 2. Compute Q, K, V
    tensor_t q = forward_linear(attn_norm_out, attn_q_w, attn_q_b);
    tensor_t k = forward_linear(attn_norm_out, attn_k_w, attn_k_b);
    tensor_t v = forward_linear(attn_norm_out, attn_v_w, attn_v_b);
    
    // 3. Reshape Q, K, V for attention
    // Q shape: [seq_len, nh, dh]
    // K shape: [seq_len, nkvh, dh]
    // V shape: [seq_len, nkvh, dh]
    std::vector<size_t> q_shape = {seq_len, meta.nh, meta.dh};
    std::vector<size_t> kv_shape = {seq_len, meta.nkvh, meta.dh};
    
    tensor_t q_reshaped = Tensor::create(q_shape, q->dtype(), q->deviceType(), q->deviceId());
    tensor_t k_reshaped = Tensor::create(kv_shape, k->dtype(), k->deviceType(), k->deviceId());
    tensor_t v_reshaped = Tensor::create(kv_shape, v->dtype(), v->deviceType(), v->deviceId());
    
    // Copy data (simplified, should use proper rearrangement)
    memcpy(q_reshaped->data(), q->data(), q->numel() * q->elementSize());
    memcpy(k_reshaped->data(), k->data(), k->numel() * k->elementSize());
    memcpy(v_reshaped->data(), v->data(), v->numel() * v->elementSize());
    
    // 4. Get KV-Cache
    auto& kv_cache = kv_caches[layer_idx];
    size_t cache_size = kv_cache.size;
    
    // 5. Update KV-Cache
    // Resize KV-Cache if needed
    if (cache_size + seq_len > meta.maxseq) {
        // Reset cache if maxseq exceeded
        cache_size = 0;
    }
    
    // Update K-Cache
    std::vector<size_t> k_cache_shape = {cache_size + seq_len, meta.nkvh, meta.dh};
    tensor_t k_cache = Tensor::create(k_cache_shape, k->dtype(), k->deviceType(), k->deviceId());
    
    // Copy existing cache
    if (cache_size > 0) {
        std::vector<size_t> old_k_shape = {cache_size, meta.nkvh, meta.dh};
        tensor_t old_k_cache = Tensor::create(old_k_shape, k->dtype(), k->deviceType(), k->deviceId());
        memcpy(old_k_cache->data(), kv_cache.k_cache->data(), old_k_cache->numel() * old_k_cache->elementSize());
        memcpy(k_cache->data(), old_k_cache->data(), old_k_cache->numel() * old_k_cache->elementSize());
    }
    
    // Copy new K values
    memcpy(reinterpret_cast<std::byte*>(reinterpret_cast<uintptr_t>(k_cache->data()) + cache_size * meta.nkvh * meta.dh * k_cache->elementSize()), 
           k_reshaped->data(), 
           k_reshaped->numel() * k_reshaped->elementSize());
    
    // Update V-Cache
    std::vector<size_t> v_cache_shape = {cache_size + seq_len, meta.nkvh, meta.dh};
    tensor_t v_cache = Tensor::create(v_cache_shape, v->dtype(), v->deviceType(), v->deviceId());
    
    // Copy existing cache
    if (cache_size > 0) {
        std::vector<size_t> old_v_shape = {cache_size, meta.nkvh, meta.dh};
        tensor_t old_v_cache = Tensor::create(old_v_shape, v->dtype(), v->deviceType(), v->deviceId());
        memcpy(old_v_cache->data(), kv_cache.v_cache->data(), old_v_cache->numel() * old_v_cache->elementSize());
        memcpy(v_cache->data(), old_v_cache->data(), old_v_cache->numel() * old_v_cache->elementSize());
    }
    
    // Copy new V values
    memcpy(reinterpret_cast<std::byte*>(reinterpret_cast<uintptr_t>(v_cache->data()) + cache_size * meta.nkvh * meta.dh * v_cache->elementSize()), 
           v_reshaped->data(), 
           v_reshaped->numel() * v_reshaped->elementSize());
    
    // Update KV-Cache
    kv_cache.k_cache = k_cache;
    kv_cache.v_cache = v_cache;
    kv_cache.size = cache_size + seq_len;
    
    // 6. Compute attention
    float scale = 1.0f / std::sqrt(static_cast<float>(meta.dh));
    std::vector<size_t> attn_out_shape = {seq_len, meta.nh, meta.dh};
    tensor_t attn_val = Tensor::create(attn_out_shape, q->dtype(), q->deviceType(), q->deviceId());
    
    llaisys::ops::self_attention(attn_val, q_reshaped, k_cache, v_cache, scale);
    
    // 7. Reshape attention output for linear layer
    std::vector<size_t> attn_val_flat_shape = {seq_len, meta.hs};
    tensor_t attn_val_flat = Tensor::create(attn_val_flat_shape, attn_val->dtype(), attn_val->deviceType(), attn_val->deviceId());
    memcpy(attn_val_flat->data(), attn_val->data(), attn_val->numel() * attn_val->elementSize());
    
    // 8. Attention output linear layer
    tensor_t attn_o_out = forward_linear(attn_val_flat, attn_o_w, nullptr);
    
    // 9. Residual connection
    tensor_t residual1 = Tensor::create(input->shape(), input->dtype(), input->deviceType(), input->deviceId());
    // Simplified: should use proper add operation
    memcpy(residual1->data(), input->data(), input->numel() * input->elementSize());
    
    // 10. RMS Norm for MLP
    tensor_t mlp_norm_out = forward_norm(residual1, mlp_norm_w, meta.epsilon);
    
    // 11. MLP
    tensor_t mlp_gate_out = forward_linear(mlp_norm_out, mlp_gate_w, nullptr);
    tensor_t mlp_up_out = forward_linear(mlp_norm_out, mlp_up_w, nullptr);
    
    // SwiGLU activation
    tensor_t mlp_swiglu_out = Tensor::create(mlp_gate_out->shape(), mlp_gate_out->dtype(), mlp_gate_out->deviceType(), mlp_gate_out->deviceId());
    llaisys::ops::swiglu(mlp_swiglu_out, mlp_gate_out, mlp_up_out);
    
    // MLP down projection
    tensor_t mlp_down_out = forward_linear(mlp_swiglu_out, mlp_down_w, nullptr);
    
    // 12. Residual connection
    tensor_t output = Tensor::create(residual1->shape(), residual1->dtype(), residual1->deviceType(), residual1->deviceId());
    // Simplified: should use proper add operation
    memcpy(output->data(), residual1->data(), residual1->numel() * residual1->elementSize());
    
    return output;
}

 tensor_t Qwen2Model::forward_norm(tensor_t input, tensor_t weight, float eps) {
    // Create output tensor
    tensor_t out = Tensor::create(input->shape(), input->dtype(), input->deviceType(), input->deviceId());
    
    // Perform RMS norm
    llaisys::ops::rms_norm(out, input, weight, eps);
    
    return out;
}

 tensor_t Qwen2Model::forward_linear(tensor_t input, tensor_t weight, tensor_t bias) {
    // Calculate output shape: [batch_size, weight_out_features]
    std::vector<size_t> out_shape = {
        input->shape()[0],
        weight->shape()[0]
    };
    
    // Create output tensor
    tensor_t out = Tensor::create(out_shape, input->dtype(), input->deviceType(), input->deviceId());
    
    // Perform linear transformation
    llaisys::ops::linear(out, input, weight, bias);
    
    return out;
}

int64_t Qwen2Model::infer(int64_t* token_ids, size_t ntoken) {
    // 1. Get embedding
    tensor_t emb = forward_embedding(token_ids, ntoken);
    
    // 2. Forward through all layers
    tensor_t hidden = emb;
    for (size_t i = 0; i < meta.nlayer; ++i) {
        hidden = forward_layer(static_cast<int>(i), hidden, ntoken);
    }
    
    // 3. Final norm
    // TODO: Convert llaisysTensor_t to tensor_t
    int device_id = device_ids.empty() ? 0 : device_ids[0];
    tensor_t out_norm_w = Tensor::create({meta.hs}, meta.dtype, device, device_id);
    tensor_t final_norm_out = forward_norm(hidden, out_norm_w, meta.epsilon);
    
    // 4. Output embedding to get logits
    // TODO: Convert llaisysTensor_t to tensor_t
    tensor_t out_embed = Tensor::create({meta.voc, meta.hs}, meta.dtype, device, device_id);
    tensor_t logits = forward_linear(final_norm_out, out_embed, nullptr);
    
    // 5. Get the last token's logits
    std::vector<size_t> last_token_logits_shape = {1, meta.voc};
    tensor_t last_logits = Tensor::create(last_token_logits_shape, logits->dtype(), logits->deviceType(), logits->deviceId());
    
    // Copy last token's logits
    size_t last_token_offset = (ntoken - 1) * meta.voc * logits->elementSize();
    memcpy(last_logits->data(), reinterpret_cast<std::byte*>(reinterpret_cast<uintptr_t>(logits->data()) + last_token_offset), 
           meta.voc * logits->elementSize());
    
    // 6. Argmax to get next token
    tensor_t argmax_out = Tensor::create({1}, LLAISYS_DTYPE_I64, logits->deviceType(), logits->deviceId());
    tensor_t max_val = Tensor::create({1}, logits->dtype(), logits->deviceType(), logits->deviceId());
    
    // Reshape last_logits to 1D tensor
    tensor_t last_logits_1d = last_logits->view({meta.voc});
    llaisys::ops::argmax(argmax_out, max_val, last_logits_1d);
    
    // 7. Get the result
    int64_t* argmax_data = reinterpret_cast<int64_t*>(argmax_out->data());
    int64_t next_token = argmax_data[0];
    
    return next_token;
}

} // namespace llaisys
