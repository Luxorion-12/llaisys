from ctypes import *
from .llaisys_types import llaisysDeviceType_t
from .tensor import llaisysTensor_t


# Qwen2 Model Types
class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", c_int),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]

class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", POINTER(llaisysTensor_t)),
        ("out_embed", POINTER(llaisysTensor_t)),
        ("out_norm_w", POINTER(llaisysTensor_t)),
        ("attn_norm_w", POINTER(POINTER(llaisysTensor_t))),
        ("attn_q_w", POINTER(POINTER(llaisysTensor_t))),
        ("attn_q_b", POINTER(POINTER(llaisysTensor_t))),
        ("attn_k_w", POINTER(POINTER(llaisysTensor_t))),
        ("attn_k_b", POINTER(POINTER(llaisysTensor_t))),
        ("attn_v_w", POINTER(POINTER(llaisysTensor_t))),
        ("attn_v_b", POINTER(POINTER(llaisysTensor_t))),
        ("attn_o_w", POINTER(POINTER(llaisysTensor_t))),
        ("mlp_norm_w", POINTER(POINTER(llaisysTensor_t))),
        ("mlp_gate_w", POINTER(POINTER(llaisysTensor_t))),
        ("mlp_up_w", POINTER(POINTER(llaisysTensor_t))),
        ("mlp_down_w", POINTER(POINTER(llaisysTensor_t))),
    ]

class LlaisysQwen2Model(Structure):
    pass


# Load Qwen2 Model Functions
def load_qwen2_model(lib):
    # llaisysQwen2ModelCreate
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = POINTER(LlaisysQwen2Model)

    # llaisysQwen2ModelDestroy
    lib.llaisysQwen2ModelDestroy.argtypes = [POINTER(LlaisysQwen2Model)]
    lib.llaisysQwen2ModelDestroy.restype = None

    # llaisysQwen2ModelWeights
    lib.llaisysQwen2ModelWeights.argtypes = [POINTER(LlaisysQwen2Model)]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    # llaisysQwen2ModelInfer
    lib.llaisysQwen2ModelInfer.argtypes = [
        POINTER(LlaisysQwen2Model),
        POINTER(c_int64),
        c_size_t,
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64
