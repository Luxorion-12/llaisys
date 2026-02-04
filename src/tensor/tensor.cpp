#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}
//<任务1.2>检查张量的形状(shapes)和步长(strides)，判断它在内存中是否连续。
bool Tensor::isContiguous() const {
    size_t ndim = this->ndim();
    if (ndim <= 1) {
        return true;
    }
    
    const auto& strides = this->strides();
    const auto& shape = this->shape();
    
    ptrdiff_t expected_stride = 1;
    for (size_t i = ndim - 1; i > 0; i--) {
        if (strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= shape[i];
    }
    
    if (strides[0] != expected_stride) {
        return false;
    }
    
    return true;
}
//<任务1.4>创建一个新张量，改变原始张量维度的顺序。转置可以通过这个函数实现，而无需移动数据
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t tensor_ndim = ndim();
    if (order.size() != tensor_ndim) {
        throw std::runtime_error("Order size must match tensor dimensions");
    }
    
    // 检查order数组的有效性
    std::vector<bool> seen(tensor_ndim, false);
    for (size_t dim : order) {
        if (dim >= tensor_ndim || seen[dim]) {
            throw std::runtime_error("Invalid permutation order");
        }
        seen[dim] = true;
    }
    
    TensorMeta new_meta = _meta;
    new_meta.shape.resize(tensor_ndim);
    new_meta.strides.resize(tensor_ndim);
    
    // 计算新的形状和步长
    for (size_t i = 0; i < tensor_ndim; i++) {
        new_meta.shape[i] = _meta.shape[order[i]];
        new_meta.strides[i] = _meta.strides[order[i]];
    }
    
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}
//<任务1.3>
//创建一个新张量，-->new 一个Tensor 通过拆分或合并原始维度将原始张量重塑为给定形状。不涉及数据传输。-->使用引用
// 例如，通过合并最后两个维度，将形状为(2, 3, 5)的张量更改为(2, 15)。
// 这个函数不是简单地改变张量的形状那么简单，尽管测试会通过。如果新视图与
// 原始张量不兼容，它应该引发错误。想想一个形状为(2, 3, 5)、步长为(30, 10, 1)
// 的张量。你还能在不传输数据的情况下将其重塑为(2, 15)吗？
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t original_numel = numel();
    size_t new_numel = 1;
    for (size_t s : shape) {
        new_numel *= s;
    }
    
    if (original_numel != new_numel) {
        throw std::runtime_error("Shape mismatch: new shape has different number of elements");
    }
    
    TensorMeta new_meta = _meta;
    new_meta.shape = shape;
    
    // 计算新的步长
    size_t ndim = shape.size();
    new_meta.strides.resize(ndim);
    if (ndim > 0) {
        new_meta.strides[ndim - 1] = 1;
        for (size_t i = ndim - 2; i < ndim; i--) {
            new_meta.strides[i] = new_meta.strides[i + 1] * shape[i + 1];
        }
    }
    
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}
//<任务1.5>创建一个新张量，沿给定维度，start（包含）和end（不包含）索引对原始张量进行切片操作。
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    size_t tensor_ndim = ndim();
    if (dim >= tensor_ndim) {
        throw std::runtime_error("Dimension out of range");
    }
    
    if (start >= end || end > shape()[dim]) {
        throw std::runtime_error("Invalid slice indices");
    }
    
    TensorMeta new_meta = _meta;
    new_meta.shape[dim] = end - start;
    
    // 计算新的偏移量
    size_t new_offset = _offset + start * strides()[dim] * elementSize();
    
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}
// <任务1.1>
// 将主机（cpu）数据加载到张量（可以在设备上）。查看构造函数了解如何获取当前设备
// 上下文的运行时API，并执行从主机到设备的内存复制。
void Tensor::load(const void *src_) {
    size_t total_elems = numel();
    size_t dtype_size = elementSize();
    
    if (deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(data(), src_, total_elems * dtype_size);
    } else {
        core::context().setDevice(deviceType(), deviceId());
        core::context().runtime().api()->memcpy_sync(
            data(),
            src_,
            total_elems * dtype_size,
            LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    if (isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    
    // 创建一个新的连续张量
    size_t total_elems = numel();
    tensor_t new_tensor = create(shape(), dtype(), deviceType(), deviceId());
    
    // 复制数据
    if (deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(new_tensor->data(), data(), total_elems * elementSize());
    } else {
        core::context().setDevice(deviceType(), deviceId());
        core::context().runtime().api()->memcpy_sync(
            new_tensor->data(),
            data(),
            total_elems * elementSize(),
            LLAISYS_MEMCPY_D2D);
    }
    
    return new_tensor;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    size_t original_numel = numel();
    size_t new_numel = 1;
    for (size_t s : shape) {
        new_numel *= s;
    }
    
    if (original_numel != new_numel) {
        throw std::runtime_error("Shape mismatch: new shape has different number of elements");
    }
    
    TensorMeta new_meta = _meta;
    new_meta.shape = shape;
    
    // 计算新的步长
    size_t ndim = shape.size();
    new_meta.strides.resize(ndim);
    if (ndim > 0) {
        new_meta.strides[ndim - 1] = 1;
        for (size_t i = ndim - 2; i < ndim; i--) {
            new_meta.strides[i] = new_meta.strides[i + 1] * shape[i + 1];
        }
    }
    
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    if (deviceType() == device_type && deviceId() == device) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    
    // 创建目标设备上的新张量
    tensor_t new_tensor = create(shape(), dtype(), device_type, device);
    
    // 复制数据
    size_t total_elems = numel();
    size_t elem_size = elementSize();
    
    if (deviceType() == LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        std::memcpy(new_tensor->data(), data(), total_elems * elem_size);
    } else if (deviceType() == LLAISYS_DEVICE_CPU && device_type != LLAISYS_DEVICE_CPU) {
        core::context().setDevice(device_type, device);
        core::context().runtime().api()->memcpy_sync(
            new_tensor->data(),
            data(),
            total_elems * elem_size,
            LLAISYS_MEMCPY_H2D);
    } else if (deviceType() != LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        core::context().setDevice(deviceType(), deviceId());
        core::context().runtime().api()->memcpy_sync(
            new_tensor->data(),
            data(),
            total_elems * elem_size,
            LLAISYS_MEMCPY_D2H);
    } else {
        core::context().setDevice(deviceType(), deviceId());
        core::context().runtime().api()->memcpy_sync(
            new_tensor->data(),
            data(),
            total_elems * elem_size,
            LLAISYS_MEMCPY_D2D);
    }
    
    return new_tensor;
}

} // namespace llaisys
