#include "op.hpp"
#include "../../utils/types.hpp"

namespace llaisys::ops {

namespace {

// Linear implementation for different data types
template <typename T>
void linear_impl(
    T* out_data, const T* in_data, const T* weight_data, const T* bias_data,
    size_t batch_size, size_t in_features, size_t out_features)
{
    // For each batch element
    for (size_t i = 0; i < batch_size; ++i) {
        // For each output feature
        for (size_t j = 0; j < out_features; ++j) {
            // Calculate dot product: sum(in[i][k] * weight[j][k])
            float dot_product = 0.0f;
            for (size_t k = 0; k < in_features; ++k) {
                // Get input and weight values
                size_t in_idx = i * in_features + k;
                size_t weight_idx = j * in_features + k;
                
                T in_val = in_data[in_idx];
                T weight_val = weight_data[weight_idx];
                
                // Convert to float for calculation
                float in_f = llaisys::utils::cast<float>(in_val);
                float weight_f = llaisys::utils::cast<float>(weight_val);
                
                // Accumulate dot product
                dot_product += in_f * weight_f;
            }
            
            // Add bias if provided
            if (bias_data != nullptr) {
                T bias_val = bias_data[j];
                float bias_f = llaisys::utils::cast<float>(bias_val);
                dot_product += bias_f;
            }
            
            // Store result
            size_t out_idx = i * out_features + j;
            out_data[out_idx] = llaisys::utils::cast<T>(dot_product);
        }
    }
}

} // namespace

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // Validate input shapes
    if (out->ndim() != 2) {
        throw std::runtime_error("Linear: out must be a 2D tensor");
    }
    if (in->ndim() != 2) {
        throw std::runtime_error("Linear: in must be a 2D tensor");
    }
    if (weight->ndim() != 2) {
        throw std::runtime_error("Linear: weight must be a 2D tensor");
    }
    
    // Get shapes
    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    size_t weight_in_features = weight->shape()[1];
    size_t out_batch_size = out->shape()[0];
    size_t out_out_features = out->shape()[1];
    
    // Validate shape consistency
    if (out_batch_size != batch_size) {
        throw std::runtime_error("Linear: out batch size must match in batch size");
    }
    if (out_out_features != out_features) {
        throw std::runtime_error("Linear: out features must match weight out features");
    }
    if (weight_in_features != in_features) {
        throw std::runtime_error("Linear: weight in features must match in features");
    }
    
    // Validate data types
    if (in->dtype() != out->dtype()) {
        throw std::runtime_error("Linear: in and out must have the same dtype");
    }
    if (weight->dtype() != out->dtype()) {
        throw std::runtime_error("Linear: weight and out must have the same dtype");
    }
    if (bias != nullptr && bias->dtype() != out->dtype()) {
        throw std::runtime_error("Linear: bias and out must have the same dtype");
    }
    if (bias != nullptr && bias->ndim() != 1) {
        throw std::runtime_error("Linear: bias must be a 1D tensor");
    }
    if (bias != nullptr && bias->shape()[0] != out_features) {
        throw std::runtime_error("Linear: bias size must match out features");
    }
    
    // Get data pointers
    std::byte* out_data = out->data();
    const std::byte* in_data = in->data();
    const std::byte* weight_data = weight->data();
    const std::byte* bias_data = nullptr;
    if (bias != nullptr) {
        bias_data = bias->data();
    }
    
    // Call implementation based on data type
    llaisysDataType_t dtype = out->dtype();
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            linear_impl(
                reinterpret_cast<float*>(out_data),
                reinterpret_cast<const float*>(in_data),
                reinterpret_cast<const float*>(weight_data),
                reinterpret_cast<const float*>(bias_data),
                batch_size, in_features, out_features);
            break;
        }
        case LLAISYS_DTYPE_F16: {
            linear_impl(
                reinterpret_cast<fp16_t*>(out_data),
                reinterpret_cast<const fp16_t*>(in_data),
                reinterpret_cast<const fp16_t*>(weight_data),
                reinterpret_cast<const fp16_t*>(bias_data),
                batch_size, in_features, out_features);
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            linear_impl(
                reinterpret_cast<bf16_t*>(out_data),
                reinterpret_cast<const bf16_t*>(in_data),
                reinterpret_cast<const bf16_t*>(weight_data),
                reinterpret_cast<const bf16_t*>(bias_data),
                batch_size, in_features, out_features);
            break;
        }
        default:
            throw std::runtime_error("Linear: unsupported data type");
    }
}

} // namespace llaisys::ops
