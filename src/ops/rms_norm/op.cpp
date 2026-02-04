#include "op.hpp"
#include "../../utils/types.hpp"

namespace llaisys::ops {

namespace {

// Template function to compute RMS norm for different data types
template <typename T>
void rms_norm_impl(
    T* out_data, const T* in_data, const T* weight_data,
    size_t batch_size, size_t d, float eps)
{
    // Compute RMS norm for each row
    for (size_t i = 0; i < batch_size; ++i) {
        // Compute sum of squares
        float sum_sq = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            size_t idx = i * d + j;
            float x = llaisys::utils::cast<float>(in_data[idx]);
            sum_sq += x * x;
        }
        
        // Compute RMS: sqrt((1/d) * sum_sq) + eps
        float rms = std::sqrt(sum_sq / static_cast<float>(d)) + eps;
        
        // Normalize and apply weight
        for (size_t j = 0; j < d; ++j) {
            size_t idx = i * d + j;
            float x = llaisys::utils::cast<float>(in_data[idx]);
            float w = llaisys::utils::cast<float>(weight_data[j]);
            float y = w * (x / rms);
            out_data[idx] = llaisys::utils::cast<T>(y);
        }
    }
}

} // namespace

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // Validate input shapes
    if (out->ndim() != 2) {
        throw std::runtime_error("RMS Norm: out must be a 2D tensor");
    }
    if (in->ndim() != 2) {
        throw std::runtime_error("RMS Norm: in must be a 2D tensor");
    }
    if (weight->ndim() != 1) {
        throw std::runtime_error("RMS Norm: weight must be a 1D tensor");
    }
    
    // Get shapes
    size_t batch_size = in->shape()[0];
    size_t d = in->shape()[1];
    size_t out_batch_size = out->shape()[0];
    size_t out_d = out->shape()[1];
    size_t weight_d = weight->shape()[0];
    
    // Validate shape consistency
    if (out_batch_size != batch_size) {
        throw std::runtime_error("RMS Norm: out batch size must match in batch size");
    }
    if (out_d != d) {
        throw std::runtime_error("RMS Norm: out feature size must match in feature size");
    }
    if (weight_d != d) {
        throw std::runtime_error("RMS Norm: weight size must match in feature size");
    }
    
    // Validate data types
    if (in->dtype() != out->dtype()) {
        throw std::runtime_error("RMS Norm: in and out must have the same dtype");
    }
    if (weight->dtype() != out->dtype()) {
        throw std::runtime_error("RMS Norm: weight and out must have the same dtype");
    }
    
    // Get data pointers
    std::byte* out_data = out->data();
    const std::byte* in_data = in->data();
    const std::byte* weight_data = weight->data();
    
    // Call implementation based on data type
    llaisysDataType_t dtype = out->dtype();
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            rms_norm_impl(
                reinterpret_cast<float*>(out_data),
                reinterpret_cast<const float*>(in_data),
                reinterpret_cast<const float*>(weight_data),
                batch_size, d, eps);
            break;
        }
        case LLAISYS_DTYPE_F16: {
            rms_norm_impl(
                reinterpret_cast<fp16_t*>(out_data),
                reinterpret_cast<const fp16_t*>(in_data),
                reinterpret_cast<const fp16_t*>(weight_data),
                batch_size, d, eps);
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            rms_norm_impl(
                reinterpret_cast<bf16_t*>(out_data),
                reinterpret_cast<const bf16_t*>(in_data),
                reinterpret_cast<const bf16_t*>(weight_data),
                batch_size, d, eps);
            break;
        }
        default:
            throw std::runtime_error("RMS Norm: unsupported data type");
    }
}

} // namespace llaisys::ops
