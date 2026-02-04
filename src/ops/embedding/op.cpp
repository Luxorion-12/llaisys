#include "op.hpp"
#include "../../utils/types.hpp"

namespace llaisys::ops {

namespace {

// Embedding implementation for different data types
template <typename T>
void embedding_impl(
    T* out_data, const int64_t* index_data, const T* weight_data,
    size_t index_size, size_t weight_rows, size_t weight_cols)
{
    for (size_t i = 0; i < index_size; ++i) {
        // Get the index value
        int64_t idx = index_data[i];
        // Validate index
        if (idx < 0 || idx >= static_cast<int64_t>(weight_rows)) {
            throw std::runtime_error("Embedding: index out of bounds");
        }
        // Calculate source and destination offsets
        size_t src_offset = static_cast<size_t>(idx) * weight_cols;
        size_t dst_offset = i * weight_cols;
        // Copy row from weight to output
        for (size_t j = 0; j < weight_cols; ++j) {
            out_data[dst_offset + j] = weight_data[src_offset + j];
        }
    }
}

} // namespace

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // Validate input shapes
    if (out->ndim() != 2) {
        throw std::runtime_error("Embedding: out must be a 2D tensor");
    }
    if (index->ndim() != 1) {
        throw std::runtime_error("Embedding: index must be a 1D tensor");
    }
    if (weight->ndim() != 2) {
        throw std::runtime_error("Embedding: weight must be a 2D tensor");
    }
    
    // Get shapes
    size_t index_size = index->shape()[0];
    size_t weight_rows = weight->shape()[0];
    size_t weight_cols = weight->shape()[1];
    size_t out_rows = out->shape()[0];
    size_t out_cols = out->shape()[1];
    
    // Validate shape consistency
    if (out_rows != index_size) {
        throw std::runtime_error("Embedding: out rows must match index size");
    }
    if (out_cols != weight_cols) {
        throw std::runtime_error("Embedding: out cols must match weight cols");
    }
    
    // Validate data types
    if (index->dtype() != LLAISYS_DTYPE_I64) {
        throw std::runtime_error("Embedding: index must be int64");
    }
    if (out->dtype() != weight->dtype()) {
        throw std::runtime_error("Embedding: out and weight must have the same dtype");
    }
    
    // Get data pointers
    std::byte* out_data = out->data();
    const std::byte* index_data = index->data();
    const std::byte* weight_data = weight->data();
    
    // Call implementation based on data type
    llaisysDataType_t dtype = out->dtype();
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            embedding_impl(
                reinterpret_cast<float*>(out_data),
                reinterpret_cast<const int64_t*>(index_data),
                reinterpret_cast<const float*>(weight_data),
                index_size, weight_rows, weight_cols);
            break;
        }
        case LLAISYS_DTYPE_F16: {
            embedding_impl(
                reinterpret_cast<fp16_t*>(out_data),
                reinterpret_cast<const int64_t*>(index_data),
                reinterpret_cast<const fp16_t*>(weight_data),
                index_size, weight_rows, weight_cols);
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            embedding_impl(
                reinterpret_cast<bf16_t*>(out_data),
                reinterpret_cast<const int64_t*>(index_data),
                reinterpret_cast<const bf16_t*>(weight_data),
                index_size, weight_rows, weight_cols);
            break;
        }
        default:
            throw std::runtime_error("Embedding: unsupported data type");
    }
}

} // namespace llaisys::ops
