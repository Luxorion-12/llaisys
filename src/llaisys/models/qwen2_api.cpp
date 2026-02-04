#include "qwen2.hpp"

using namespace llaisys;

__C {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    // Create C++ model
    Qwen2Model *model = new Qwen2Model(meta, device, device_ids, ndevice);
    return reinterpret_cast<struct LlaisysQwen2Model *>(model);
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (model) {
        Qwen2Model *cpp_model = reinterpret_cast<Qwen2Model *>(model);
        delete cpp_model;
    }
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (model) {
        Qwen2Model *cpp_model = reinterpret_cast<Qwen2Model *>(model);
        return cpp_model->get_weights();
    }
    return nullptr;
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (model) {
        Qwen2Model *cpp_model = reinterpret_cast<Qwen2Model *>(model);
        return cpp_model->infer(token_ids, ntoken);
    }
    return -1;
}

} // __C
