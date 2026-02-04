#include "op.hpp"
#include "../../utils/types.hpp"
#include <cmath>

namespace llaisys::ops {

namespace {

// Template function to compute RoPE for different data types
template <typename T>
void rope_impl(
    T* out_data, const T* in_data, const int64_t* pos_ids_data,
    size_t seqlen, size_t nhead, size_t d, float theta)
{
    // Calculate d_half = d / 2
    size_t d_half = d / 2;
    
    // For each sequence element
    for (size_t i = 0; i < seqlen; ++i) {
        // Get position id for current element
        int64_t p_i = pos_ids_data[i];
        
        // For each head
        for (size_t h = 0; h < nhead; ++h) {
            // For each dimension pair (j, j + d_half)
            for (size_t j = 0; j < d_half; ++j) {
                // Calculate index in the tensor
                size_t idx = i * nhead * d + h * d + j;
                size_t idx_b = idx + d_half;
                
                // Get input values a and b
                T a = in_data[idx];
                T b = in_data[idx_b];
                
                // Calculate angle phi = p_i / (theta^(2j/d))
                float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(d);
                float theta_j = std::pow(theta, exponent);
                float phi = static_cast<float>(p_i) / theta_j;
                
                // Calculate cos and sin of phi
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);
                
                // Convert to appropriate type for calculation
                float a_f = llaisys::utils::cast<float>(a);
                float b_f = llaisys::utils::cast<float>(b);
                
                // Calculate rotated values
                float a_prime = a_f * cos_phi - b_f * sin_phi;
                float b_prime = b_f * cos_phi + a_f * sin_phi;
                
                // Store result
                out_data[idx] = llaisys::utils::cast<T>(a_prime);
                out_data[idx_b] = llaisys::utils::cast<T>(b_prime);
            }
        }
    }
}

} // namespace

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // Validate input shapes
    if (out->ndim() != 3) {
        throw std::runtime_error("RoPE: out must be a 3D tensor");
    }
    if (in->ndim() != 3) {
        throw std::runtime_error("RoPE: in must be a 3D tensor");
    }
    if (pos_ids->ndim() != 1) {
        throw std::runtime_error("RoPE: pos_ids must be a 1D tensor");
    }
    
    // Get shapes
    auto out_shape = out->shape();
    auto in_shape = in->shape();
    auto pos_ids_shape = pos_ids->shape();
    
    size_t seqlen = in_shape[0];
    size_t nhead = in_shape[1];
    size_t d = in_shape[2];
    
    // Validate shape consistency
    if (out_shape[0] != seqlen || out_shape[1] != nhead || out_shape[2] != d) {
        throw std::runtime_error("RoPE: out shape must match in shape");
    }
    if (pos_ids_shape[0] != seqlen) {
        throw std::runtime_error("RoPE: pos_ids length must match seqlen");
    }
    if (d % 2 != 0) {
        throw std::runtime_error("RoPE: d must be even");
    }
    if (pos_ids->dtype() != LLAISYS_DTYPE_I64) {
        throw std::runtime_error("RoPE: pos_ids must be int64 dtype");
    }
    
    // Validate data types
    if (in->dtype() != out->dtype()) {
        throw std::runtime_error("RoPE: in and out must have the same dtype");
    }
    
    // Get data pointers
    std::byte* out_data = out->data();
    const std::byte* in_data = in->data();
    const std::byte* pos_ids_data = pos_ids->data();
    
    // Call implementation based on data type
    llaisysDataType_t dtype = out->dtype();
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            rope_impl(
                reinterpret_cast<float*>(out_data),
                reinterpret_cast<const float*>(in_data),
                reinterpret_cast<const int64_t*>(pos_ids_data),
                seqlen, nhead, d, theta);
            break;
        }
        case LLAISYS_DTYPE_F16: {
            rope_impl(
                reinterpret_cast<fp16_t*>(out_data),
                reinterpret_cast<const fp16_t*>(in_data),
                reinterpret_cast<const int64_t*>(pos_ids_data),
                seqlen, nhead, d, theta);
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            rope_impl(
                reinterpret_cast<bf16_t*>(out_data),
                reinterpret_cast<const bf16_t*>(in_data),
                reinterpret_cast<const int64_t*>(pos_ids_data),
                seqlen, nhead, d, theta);
            break;
        }
        default:
            throw std::runtime_error("RoPE: unsupported data type");
    }
}

} // namespace llaisys::ops
