#include "op.hpp"
#include "../../utils/types.hpp"

namespace llaisys::ops {

namespace {

// Argmax implementation for different data types
template <typename T>
void argmax_impl(
    int64_t* max_idx_data, T* max_val_data, const T* vals_data, size_t size)
{
    if (size == 0) {
        return;
    }
    
    int64_t max_idx = 0;
    T max_val = vals_data[0];
    
    // Find maximum value and its index
    for (size_t i = 1; i < size; ++i) {
        T val = vals_data[i];
        if (llaisys::utils::cast<float>(val) > llaisys::utils::cast<float>(max_val)) {
            max_val = val;
            max_idx = static_cast<int64_t>(i);
        }
    }
    
    // Store results
    max_idx_data[0] = max_idx;
    max_val_data[0] = max_val;
}

} // namespace

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // Validate input shapes
    if (max_idx->ndim() != 1 || max_idx->shape()[0] != 1) {
        throw std::runtime_error("Argmax: max_idx must be a 1D tensor with shape (1,)");
    }
    if (max_val->ndim() != 1 || max_val->shape()[0] != 1) {
        throw std::runtime_error("Argmax: max_val must be a 1D tensor with shape (1,)");
    }
    if (vals->ndim() != 1) {
        throw std::runtime_error("Argmax: vals must be a 1D tensor");
    }
    
    // Validate data types
    if (max_idx->dtype() != LLAISYS_DTYPE_I64) {
        throw std::runtime_error("Argmax: max_idx must be int64");
    }
    if (max_val->dtype() != vals->dtype()) {
        throw std::runtime_error("Argmax: max_val and vals must have the same dtype");
    }
    
    // Get data pointers
    std::byte* max_idx_data = max_idx->data();
    std::byte* max_val_data = max_val->data();
    const std::byte* vals_data = vals->data();
    
    // Get size
    size_t size = vals->shape()[0];
    
    // Call implementation based on data type
    llaisysDataType_t dtype = vals->dtype();
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            argmax_impl(
                reinterpret_cast<int64_t*>(max_idx_data),
                reinterpret_cast<float*>(max_val_data),
                reinterpret_cast<const float*>(vals_data),
                size);
            break;
        }
        case LLAISYS_DTYPE_F16: {
            argmax_impl(
                reinterpret_cast<int64_t*>(max_idx_data),
                reinterpret_cast<fp16_t*>(max_val_data),
                reinterpret_cast<const fp16_t*>(vals_data),
                size);
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            argmax_impl(
                reinterpret_cast<int64_t*>(max_idx_data),
                reinterpret_cast<bf16_t*>(max_val_data),
                reinterpret_cast<const bf16_t*>(vals_data),
                size);
            break;
        }
        default:
            throw std::runtime_error("Argmax: unsupported data type");
    }
}

} // namespace llaisys::ops
