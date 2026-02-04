#include "op.hpp"
#include "../../utils/types.hpp"
#include <cmath>
#include <algorithm>

namespace llaisys::ops {

namespace {

// Template function to compute self-attention for different data types
template <typename T>
void self_attention_impl(
    T* attn_val_data, const T* q_data, const T* k_data, const T* v_data,
    size_t seqlen, size_t nhead, size_t nkvhead, size_t d, size_t dv, size_t total_len, float scale)
{
    // Calculate heads per kvhead
    size_t heads_per_kv = nhead / nkvhead;
    
    // 1. Compute attention scores: Q * K^T * scale
    // Temporary matrix to store attention scores [seqlen, nhead, total_len]
    std::vector<float> attn_scores(seqlen * nhead * total_len, -std::numeric_limits<float>::infinity());
    
    // For each sequence element in query
    for (size_t i = 0; i < seqlen; ++i) {
        // For each head
        for (size_t h = 0; h < nhead; ++h) {
            // Map head to kvhead (same as PyTorch's repeat_interleave)
            size_t kvh = h / heads_per_kv;
            
            // For each sequence element in key (causal mask: only up to i + (total_len - seqlen))
            // This matches PyTorch's tril(diagonal=S-L) where S=total_len, L=seqlen
            size_t mask_diagonal = total_len - seqlen;
            for (size_t j = 0; j <= i + mask_diagonal; ++j) {
                // Compute dot product between q[i][h] and k[j][kvh]
                float dot = 0.0f;
                for (size_t l = 0; l < d; ++l) {
                    // q shape: [seqlen, nhead, d]
                    size_t q_idx = i * nhead * d + h * d + l;
                    // k shape: [total_len, nkvhead, d]
                    size_t k_idx = j * nkvhead * d + kvh * d + l;
                    
                    float q_val = llaisys::utils::cast<float>(q_data[q_idx]);
                    float k_val = llaisys::utils::cast<float>(k_data[k_idx]);
                    
                    dot += q_val * k_val;
                }
                
                // Apply scale factor
                dot *= scale;
                
                // Store attention score
                size_t score_idx = i * nhead * total_len + h * total_len + j;
                attn_scores[score_idx] = dot;
            }
        }
    }
    
    // 2. Apply softmax to attention scores
    // For each sequence element in query
    for (size_t i = 0; i < seqlen; ++i) {
        // For each head
        for (size_t h = 0; h < nhead; ++h) {
            // Find max value for numerical stability
            float max_val = -std::numeric_limits<float>::infinity();
            size_t mask_diagonal = total_len - seqlen;
            for (size_t j = 0; j <= i + mask_diagonal; ++j) {
                size_t score_idx = i * nhead * total_len + h * total_len + j;
                if (attn_scores[score_idx] > max_val) {
                    max_val = attn_scores[score_idx];
                }
            }
            
            // Compute exp and sum
            float sum_exp = 0.0f;
            for (size_t j = 0; j <= i + mask_diagonal; ++j) {
                size_t score_idx = i * nhead * total_len + h * total_len + j;
                float exp_val = std::exp(attn_scores[score_idx] - max_val);
                attn_scores[score_idx] = exp_val;
                sum_exp += exp_val;
            }
            
            // Normalize
            for (size_t j = 0; j <= i + mask_diagonal; ++j) {
                size_t score_idx = i * nhead * total_len + h * total_len + j;
                if (sum_exp > 0.0f) {
                    attn_scores[score_idx] /= sum_exp;
                }
            }
        }
    }
    
    // 3. Compute attention output: softmax(A) * V
    // For each sequence element in query
    for (size_t i = 0; i < seqlen; ++i) {
        // For each head
        for (size_t h = 0; h < nhead; ++h) {
            // Map head to kvhead (same as PyTorch's repeat_interleave)
            size_t kvh = h / heads_per_kv;
            
            // For each dimension in value
            for (size_t l = 0; l < dv; ++l) {
                float val = 0.0f;
                
                // For each sequence element in key/value (only up to i + (total_len - seqlen))
                size_t mask_diagonal = total_len - seqlen;
                for (size_t j = 0; j <= i + mask_diagonal; ++j) {
                    // Get attention score
                    size_t score_idx = i * nhead * total_len + h * total_len + j;
                    float score = attn_scores[score_idx];
                    
                    // Get value
                    // v shape: [total_len, nkvhead, dv]
                    size_t v_idx = j * nkvhead * dv + kvh * dv + l;
                    float v_val = llaisys::utils::cast<float>(v_data[v_idx]);
                    
                    // Accumulate
                    val += score * v_val;
                }
                
                // Store result
                // attn_val shape: [seqlen, nhead, dv]
                size_t out_idx = i * nhead * dv + h * dv + l;
                attn_val_data[out_idx] = llaisys::utils::cast<T>(val);
            }
        }
    }
}

} // namespace

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // Validate input shapes
    if (attn_val->ndim() != 3) {
        throw std::runtime_error("Self-Attention: attn_val must be a 3D tensor");
    }
    if (q->ndim() != 3) {
        throw std::runtime_error("Self-Attention: q must be a 3D tensor");
    }
    if (k->ndim() != 3) {
        throw std::runtime_error("Self-Attention: k must be a 3D tensor");
    }
    if (v->ndim() != 3) {
        throw std::runtime_error("Self-Attention: v must be a 3D tensor");
    }
    
    // Get shapes
    auto attn_val_shape = attn_val->shape();
    auto q_shape = q->shape();
    auto k_shape = k->shape();
    auto v_shape = v->shape();
    
    size_t seqlen = q_shape[0];
    size_t nhead = q_shape[1];
    size_t d = q_shape[2];
    
    size_t total_len = k_shape[0];
    size_t nkvhead = k_shape[1];
    size_t k_d = k_shape[2];
    
    size_t v_total_len = v_shape[0];
    size_t v_nkvhead = v_shape[1];
    size_t dv = v_shape[2];
    
    size_t attn_val_seqlen = attn_val_shape[0];
    size_t attn_val_nhead = attn_val_shape[1];
    size_t attn_val_dv = attn_val_shape[2];
    
    // Validate shape consistency
    if (attn_val_seqlen != seqlen) {
        throw std::runtime_error("Self-Attention: attn_val seqlen must match q seqlen");
    }
    if (attn_val_nhead != nhead) {
        throw std::runtime_error("Self-Attention: attn_val nhead must match q nhead");
    }
    if (attn_val_dv != dv) {
        throw std::runtime_error("Self-Attention: attn_val dv must match v dv");
    }
    if (k_d != d) {
        throw std::runtime_error("Self-Attention: k d must match q d");
    }
    if (v_total_len != total_len) {
        throw std::runtime_error("Self-Attention: v total_len must match k total_len");
    }
    if (v_nkvhead != nkvhead) {
        throw std::runtime_error("Self-Attention: v nkvhead must match k nkvhead");
    }
    if (nhead % nkvhead != 0) {
        throw std::runtime_error("Self-Attention: nhead must be divisible by nkvhead");
    }
    
    // Validate data types
    if (q->dtype() != attn_val->dtype()) {
        throw std::runtime_error("Self-Attention: q and attn_val must have the same dtype");
    }
    if (k->dtype() != attn_val->dtype()) {
        throw std::runtime_error("Self-Attention: k and attn_val must have the same dtype");
    }
    if (v->dtype() != attn_val->dtype()) {
        throw std::runtime_error("Self-Attention: v and attn_val must have the same dtype");
    }
    
    // Get data pointers
    std::byte* attn_val_data = attn_val->data();
    const std::byte* q_data = q->data();
    const std::byte* k_data = k->data();
    const std::byte* v_data = v->data();
    
    // Call implementation based on data type
    llaisysDataType_t dtype = attn_val->dtype();
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            self_attention_impl(
                reinterpret_cast<float*>(attn_val_data),
                reinterpret_cast<const float*>(q_data),
                reinterpret_cast<const float*>(k_data),
                reinterpret_cast<const float*>(v_data),
                seqlen, nhead, nkvhead, d, dv, total_len, scale);
            break;
        }
        case LLAISYS_DTYPE_F16: {
            self_attention_impl(
                reinterpret_cast<fp16_t*>(attn_val_data),
                reinterpret_cast<const fp16_t*>(q_data),
                reinterpret_cast<const fp16_t*>(k_data),
                reinterpret_cast<const fp16_t*>(v_data),
                seqlen, nhead, nkvhead, d, dv, total_len, scale);
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            self_attention_impl(
                reinterpret_cast<bf16_t*>(attn_val_data),
                reinterpret_cast<const bf16_t*>(q_data),
                reinterpret_cast<const bf16_t*>(k_data),
                reinterpret_cast<const bf16_t*>(v_data),
                seqlen, nhead, nkvhead, d, dv, total_len, scale);
            break;
        }
        default:
            throw std::runtime_error("Self-Attention: unsupported data type");
    }
}

} // namespace llaisys::ops
