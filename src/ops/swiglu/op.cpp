#include "op.hpp"
#include "../../utils/types.hpp"
#include <cmath>

namespace llaisys::ops {

namespace {

// Template function to compute SwiGLU for different data types
template <typename T>
void swiglu_impl(T* out_data, const T* gate_data, const T* up_data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        // Convert to float for calculation
        float gate_val = llaisys::utils::cast<float>(gate_data[i]);
        float up_val = llaisys::utils::cast<float>(up_data[i]);
        
        // Compute sigmoid(gate_val) = 1 / (1 + exp(-gate_val))
        float sigmoid = 1.0f / (1.0f + std::exp(-gate_val));
        
        // Compute SwiGLU: up * gate * sigmoid
        float result = up_val * gate_val * sigmoid;
        
        // Store result
        out_data[i] = llaisys::utils::cast<T>(result);
    }
}

} // namespace

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // Validate input shapes
    if (out->ndim() != 2) {
        throw std::runtime_error("SwiGLU: out must be a 2D tensor");
    }
    if (gate->ndim() != 2) {
        throw std::runtime_error("SwiGLU: gate must be a 2D tensor");
    }
    if (up->ndim() != 2) {
        throw std::runtime_error("SwiGLU: up must be a 2D tensor");
    }
    
    // Validate shape consistency
    auto out_shape = out->shape();
    auto gate_shape = gate->shape();
    auto up_shape = up->shape();
    
    if (out_shape[0] != gate_shape[0] || out_shape[1] != gate_shape[1]) {
        throw std::runtime_error("SwiGLU: out and gate must have the same shape");
    }
    if (out_shape[0] != up_shape[0] || out_shape[1] != up_shape[1]) {
        throw std::runtime_error("SwiGLU: out and up must have the same shape");
    }
    
    // Validate data types
    if (out->dtype() != gate->dtype()) {
        throw std::runtime_error("SwiGLU: out and gate must have the same dtype");
    }
    if (out->dtype() != up->dtype()) {
        throw std::runtime_error("SwiGLU: out and up must have the same dtype");
    }
    
    // Calculate total size
    size_t size = out_shape[0] * out_shape[1];
    
    // Get data pointers
    std::byte* out_data = out->data();
    const std::byte* gate_data = gate->data();
    const std::byte* up_data = up->data();
    
    // Call implementation based on data type
    llaisysDataType_t dtype = out->dtype();
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            swiglu_impl(
                reinterpret_cast<float*>(out_data),
                reinterpret_cast<const float*>(gate_data),
                reinterpret_cast<const float*>(up_data),
                size);
            break;
        }
        case LLAISYS_DTYPE_F16: {
            swiglu_impl(
                reinterpret_cast<fp16_t*>(out_data),
                reinterpret_cast<const fp16_t*>(gate_data),
                reinterpret_cast<const fp16_t*>(up_data),
                size);
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            swiglu_impl(
                reinterpret_cast<bf16_t*>(out_data),
                reinterpret_cast<const bf16_t*>(gate_data),
                reinterpret_cast<const bf16_t*>(up_data),
                size);
            break;
        }
        default:
            throw std::runtime_error("SwiGLU: unsupported data type");
    }
}

} // namespace llaisys::ops
